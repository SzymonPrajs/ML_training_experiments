from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm
import yaml

from mnist.config import load_config
from mnist.data import build_dataloaders
from mnist.models import build_model, count_trainable_params
from mnist.utils import (
    EpochTimer,
    ensure_unique_run_dir,
    get_env_summary,
    resolve_device,
    seed_everything,
    write_json,
)


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module, loader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        batch = y.shape[0]
        loss_sum += float(loss.item()) * batch
        acc_sum += _accuracy(logits, y) * batch
        n += batch
    return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1)}


def train_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch = y.shape[0]
        loss_sum += float(loss.item()) * batch
        acc_sum += _accuracy(logits, y) * batch
        n += batch
        pbar.set_postfix(loss=loss_sum / max(n, 1), acc=acc_sum / max(n, 1))

    return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1)}


def build_optimizer(cfg: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "sgd")).lower().strip()
    lr = float(cfg.get("lr", 0.01))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    params = model.parameters()
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", True))
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if name == "adamw":
        betas = cfg.get("betas", (0.9, 0.999))
        if isinstance(betas, (list, tuple)) and len(betas) == 2:
            betas = (float(betas[0]), float(betas[1]))
        else:
            betas = (0.9, 0.999)
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "rmsprop":
        momentum = float(cfg.get("momentum", 0.0))
        alpha = float(cfg.get("alpha", 0.99))
        return torch.optim.RMSprop(
            params, lr=lr, alpha=alpha, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    cfg: dict[str, Any], optimizer: torch.optim.Optimizer, total_epochs: int
):
    name = str(cfg.get("name", "none")).lower().strip()
    if name in {"none", "", "null"}:
        return None

    if name == "cosine":
        warmup_epochs = int(cfg.get("warmup_epochs", 0))
        min_lr = float(cfg.get("min_lr", 0.0))
        if warmup_epochs <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=min_lr
            )
        if warmup_epochs >= total_epochs:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=total_epochs
            )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    if name == "step":
        step_size = int(cfg.get("step_size", 10))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    raise ValueError(f"Unknown scheduler: {name}")


def _write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny MNIST model.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under runs/ (defaults to config stem).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dot paths, e.g. --set optimizer.lr=0.03",
    )
    return p.parse_args()


def train_run(
    cfg: dict[str, Any], *, run_name: str, run_root: Path = Path("runs")
) -> Path:
    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "auto")))

    run_dir = ensure_unique_run_dir(run_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(run_dir / "env.json", get_env_summary(device))

    data_cfg = dict(cfg["data"])
    if device.type == "mps":
        data_cfg["pin_memory"] = False
    train_loader, test_loader = build_dataloaders(data_cfg)
    model = build_model(cfg["model"]).to(device)
    param_count = count_trainable_params(model)

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))
    )
    optimizer = build_optimizer(cfg["optimizer"], model)
    scheduler = build_scheduler(
        cfg.get("scheduler", {}), optimizer, int(cfg["train"]["epochs"])
    )

    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best.pt"
    best_acc = -1.0
    best_epoch = -1

    epochs = int(cfg["train"]["epochs"])
    grad_clip_norm = cfg["train"].get("grad_clip_norm", None)
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    for epoch in range(1, epochs + 1):
        with EpochTimer() as timer:
            train_stats = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, grad_clip_norm
            )
            test_stats = evaluate(model, test_loader, loss_fn, device)

        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "seconds": timer.seconds,
        }
        _write_csv_row(metrics_path, row)

        if test_stats["acc"] > best_acc:
            best_acc = float(test_stats["acc"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "test_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )

        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"lr {lr:.6g} | "
            f"train {train_stats['loss']:.4f}/{train_stats['acc']*100:.2f}% | "
            f"test {test_stats['loss']:.4f}/{test_stats['acc']*100:.2f}% | "
            f"{timer.seconds:.1f}s"
        )

    final_test = evaluate(model, test_loader, loss_fn, device)
    write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "param_count": param_count,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "final_test_acc": float(final_test["acc"]),
            "final_test_loss": float(final_test["loss"]),
        },
    )

    # lightweight "latest" pointer for convenience
    (run_root / "latest").unlink(missing_ok=True)
    try:
        (run_root / "latest").symlink_to(run_dir.name)
    except Exception:
        pass

    return run_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=list(args.set or []))
    run_name = args.run_name or Path(args.config).stem
    train_run(cfg, run_name=run_name, run_root=Path("runs"))


if __name__ == "__main__":
    main()

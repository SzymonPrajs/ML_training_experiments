from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm
import yaml

from mnist.config import load_config
from mnist.data import build_dataloaders
from mnist.ff import (
    LabelEncoder,
    add_noise,
    ff_loss,
    goodness_from_acts,
    permute_no_fixed_points,
    pick_wrong_labels,
)
from mnist.models import build_model, count_trainable_params
from mnist.utils import (
    EpochTimer,
    ensure_unique_run_dir,
    get_env_summary,
    get_git_summary,
    now_iso,
    resolve_device,
    seed_everything,
    sync_device,
    stable_sha256,
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
    data_seconds = 0.0
    compute_seconds = 0.0
    end_compute = time.perf_counter()
    for x, y in loader:
        data_seconds += time.perf_counter() - end_compute
        sync_device(device)
        compute_start = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        sync_device(device)
        compute_seconds += time.perf_counter() - compute_start
        batch = y.shape[0]
        loss_sum += float(loss.item()) * batch
        acc_sum += _accuracy(logits, y) * batch
        n += batch
        end_compute = time.perf_counter()
    return {
        "loss": loss_sum / max(n, 1),
        "acc": acc_sum / max(n, 1),
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
    }


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
    data_seconds = 0.0
    compute_seconds = 0.0

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    end_compute = time.perf_counter()
    for x, y in pbar:
        data_seconds += time.perf_counter() - end_compute
        sync_device(device)
        compute_start = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        sync_device(device)
        compute_seconds += time.perf_counter() - compute_start

        batch = y.shape[0]
        loss_sum += float(loss.item()) * batch
        acc_sum += _accuracy(logits, y) * batch
        n += batch
        pbar.set_postfix(loss=loss_sum / max(n, 1), acc=acc_sum / max(n, 1))
        end_compute = time.perf_counter()

    return {
        "loss": loss_sum / max(n, 1),
        "acc": acc_sum / max(n, 1),
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
    }


def build_label_encoder(
    ff_cfg: dict[str, Any],
    *,
    sample_x: torch.Tensor,
    device: torch.device,
) -> LabelEncoder:
    mode = str(ff_cfg.get("label_mode", "overlay")).lower()
    if mode in {"fourier", "spatial-fourier", "spatial_fourier"}:
        mode = "spatial_fourier"
    if mode in {"morph", "morphological", "spatial-morph", "spatial_morph"}:
        mode = "spatial_morph"
    if mode in {"none", "off", "disabled"}:
        mode = "none"
    return LabelEncoder(
        mode=mode,
        num_classes=int(ff_cfg.get("num_classes", 10)),
        image_shape=tuple(sample_x.shape[1:4]),
        device=device,
        dtype=sample_x.dtype,
        label_scale=float(ff_cfg.get("label_scale", 0.6)),
        overlay_size=int(ff_cfg.get("overlay_size", 10)),
        fourier_terms=int(ff_cfg.get("fourier_terms", 4)),
        fourier_max_freq=int(ff_cfg.get("fourier_max_freq", 6)),
        fourier_seed=int(ff_cfg.get("fourier_seed", 0)),
        morph_seed=int(ff_cfg.get("morph_seed", 0)),
        morph_dilate=int(ff_cfg.get("morph_dilate", 1)),
        morph_erode=int(ff_cfg.get("morph_erode", 0)),
        morph_shift=int(ff_cfg.get("morph_shift", 0)),
        morph_scale=float(ff_cfg.get("morph_scale", 1.0)),
    )


def _ff_prepare_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    ff_cfg: dict[str, Any],
    label_encoder: LabelEncoder,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = int(ff_cfg.get("num_classes", 10))
    neg_mode = str(ff_cfg.get("neg_mode", "wrong_label")).lower()
    concat_pairs = neg_mode == "scff"
    x = add_noise(x, noise_std)
    if neg_mode == "scff":
        perm = permute_no_fixed_points(x.shape[0], device=x.device)
        x_other = x[perm]
        y_other = y[perm]
        x_base = label_encoder.encode(x, y)
        x_other = label_encoder.encode(x_other, y_other)
        x_pos = torch.cat([x_base, x_base], dim=1) if concat_pairs else x_base
        x_neg = torch.cat([x_base, x_other], dim=1) if concat_pairs else x_other
        return x_pos, x_neg

    y_neg = pick_wrong_labels(y, num_classes)
    x_pos = label_encoder.encode(x, y)
    x_neg = label_encoder.encode(x, y_neg)
    return x_pos, x_neg


@torch.no_grad()
def _ff_predict(
    model: nn.Module,
    x: torch.Tensor,
    *,
    ff_cfg: dict[str, Any],
    label_encoder: LabelEncoder,
) -> torch.Tensor:
    num_classes = int(ff_cfg.get("num_classes", 10))
    goodness_mode = str(ff_cfg.get("goodness", "sum_sq")).lower()
    neg_mode = str(ff_cfg.get("neg_mode", "wrong_label")).lower()
    concat_pairs = neg_mode == "scff"

    scores: list[torch.Tensor] = []
    for label in range(num_classes):
        y_label = torch.full(
            (x.shape[0],), label, device=x.device, dtype=torch.long
        )
        x_enc = label_encoder.encode(x, y_label)
        if concat_pairs:
            x_enc = torch.cat([x_enc, x_enc], dim=1)
        _, acts = model.forward_with_activations(x_enc, detach_between=False)
        scores.append(goodness_from_acts(acts, goodness_mode))
    stacked = torch.stack(scores, dim=1)
    return torch.argmax(stacked, dim=1)


def train_one_epoch_ff(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    ff_cfg: dict[str, Any],
    label_encoder: LabelEncoder,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    data_seconds = 0.0
    compute_seconds = 0.0

    threshold = float(ff_cfg.get("threshold", 2.0))
    beta = float(ff_cfg.get("loss_beta", 1.0))
    goodness_mode = str(ff_cfg.get("goodness", "sum_sq")).lower()
    noise_std = float(ff_cfg.get("noise_std", 0.0))
    compute_train_acc = bool(ff_cfg.get("compute_train_acc", False))

    pbar = tqdm(loader, desc="train_ff", leave=False, dynamic_ncols=True)
    end_compute = time.perf_counter()
    for x, y in pbar:
        data_seconds += time.perf_counter() - end_compute
        sync_device(device)
        compute_start = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_pos, x_neg = _ff_prepare_inputs(
            x, y, ff_cfg=ff_cfg, label_encoder=label_encoder, noise_std=noise_std
        )

        optimizer.zero_grad(set_to_none=True)
        _, acts_pos = model.forward_with_activations(x_pos, detach_between=True)
        _, acts_neg = model.forward_with_activations(x_neg, detach_between=True)
        loss = ff_loss(
            acts_pos,
            acts_neg,
            threshold=threshold,
            beta=beta,
            goodness_mode=goodness_mode,
        )
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        if compute_train_acc:
            preds = _ff_predict(
                model, x, ff_cfg=ff_cfg, label_encoder=label_encoder
            )
            acc_sum += float((preds == y).float().sum().item())

        sync_device(device)
        compute_seconds += time.perf_counter() - compute_start

        batch = y.shape[0]
        loss_sum += float(loss.item()) * batch
        n += batch
        if compute_train_acc:
            pbar.set_postfix(loss=loss_sum / max(n, 1), acc=acc_sum / max(n, 1))
        else:
            pbar.set_postfix(loss=loss_sum / max(n, 1))
        end_compute = time.perf_counter()

    train_acc = acc_sum / max(n, 1) if compute_train_acc else float("nan")
    return {
        "loss": loss_sum / max(n, 1),
        "acc": train_acc,
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
    }


@torch.no_grad()
def evaluate_ff(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    ff_cfg: dict[str, Any],
    label_encoder: LabelEncoder,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    data_seconds = 0.0
    compute_seconds = 0.0

    threshold = float(ff_cfg.get("threshold", 2.0))
    beta = float(ff_cfg.get("loss_beta", 1.0))
    goodness_mode = str(ff_cfg.get("goodness", "sum_sq")).lower()

    end_compute = time.perf_counter()
    for x, y in loader:
        data_seconds += time.perf_counter() - end_compute
        sync_device(device)
        compute_start = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_pos, x_neg = _ff_prepare_inputs(
            x, y, ff_cfg=ff_cfg, label_encoder=label_encoder, noise_std=0.0
        )
        _, acts_pos = model.forward_with_activations(x_pos, detach_between=False)
        _, acts_neg = model.forward_with_activations(x_neg, detach_between=False)
        loss = ff_loss(
            acts_pos,
            acts_neg,
            threshold=threshold,
            beta=beta,
            goodness_mode=goodness_mode,
        )

        preds = _ff_predict(model, x, ff_cfg=ff_cfg, label_encoder=label_encoder)
        sync_device(device)
        compute_seconds += time.perf_counter() - compute_start

        batch = y.shape[0]
        loss_sum += float(loss.item()) * batch
        acc_sum += float((preds == y).float().sum().item())
        n += batch
        end_compute = time.perf_counter()

    return {
        "loss": loss_sum / max(n, 1),
        "acc": acc_sum / max(n, 1),
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
    }


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
    cfg: dict[str, Any],
    *,
    run_name: str,
    run_root: Path = Path("runs"),
    meta: dict[str, Any] | None = None,
) -> Path:
    meta_out: dict[str, Any] = dict(meta or {})
    meta_out["run_name"] = run_name
    meta_out["started_at"] = now_iso()
    started_perf = time.perf_counter()
    meta_out["config_hash"] = stable_sha256(cfg)[:12]
    meta_out["git"] = get_git_summary()

    seed_everything(int(cfg["seed"]))
    device = resolve_device(str(cfg.get("device", "auto")))
    meta_out["device"] = str(device)

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

    train_mode = str(cfg.get("train", {}).get("mode", "backprop")).lower()
    if train_mode in {"ff", "forward_forward"}:
        train_mode = "forward_forward"
    else:
        train_mode = "backprop"

    ff_cfg: dict[str, Any] = dict(cfg.get("ff", {}))
    label_encoder: LabelEncoder | None = None
    loss_fn: nn.Module | None = None
    if train_mode == "forward_forward":
        if not hasattr(model, "forward_with_activations"):
            raise RuntimeError("Model does not support forward_with_activations for FF.")
        sample_x, _ = next(iter(train_loader))
        label_encoder = build_label_encoder(ff_cfg, sample_x=sample_x, device=device)
    else:
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
    total_train_data = 0.0
    total_train_compute = 0.0
    total_eval_data = 0.0
    total_eval_compute = 0.0

    for epoch in range(1, epochs + 1):
        with EpochTimer() as timer:
            if train_mode == "forward_forward":
                if label_encoder is None:
                    raise RuntimeError("Label encoder not initialized for FF training.")
                train_stats = train_one_epoch_ff(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    ff_cfg=ff_cfg,
                    label_encoder=label_encoder,
                    grad_clip_norm=grad_clip_norm,
                )
                test_stats = evaluate_ff(
                    model,
                    test_loader,
                    device,
                    ff_cfg=ff_cfg,
                    label_encoder=label_encoder,
                )
            else:
                if loss_fn is None:
                    raise RuntimeError("Loss function not initialized for backprop training.")
                train_stats = train_one_epoch(
                    model, train_loader, loss_fn, optimizer, device, grad_clip_norm
                )
                test_stats = evaluate(model, test_loader, loss_fn, device)

        lr = float(optimizer.param_groups[0]["lr"])
        data_seconds = float(train_stats.get("data_seconds", 0.0)) + float(
            test_stats.get("data_seconds", 0.0)
        )
        compute_seconds = float(train_stats.get("compute_seconds", 0.0)) + float(
            test_stats.get("compute_seconds", 0.0)
        )
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "seconds": timer.seconds,
            "data_seconds": data_seconds,
            "compute_seconds": compute_seconds,
            "train_data_seconds": float(train_stats.get("data_seconds", 0.0)),
            "train_compute_seconds": float(train_stats.get("compute_seconds", 0.0)),
            "eval_data_seconds": float(test_stats.get("data_seconds", 0.0)),
            "eval_compute_seconds": float(test_stats.get("compute_seconds", 0.0)),
        }
        _write_csv_row(metrics_path, row)

        total_train_data += float(train_stats.get("data_seconds", 0.0))
        total_train_compute += float(train_stats.get("compute_seconds", 0.0))
        total_eval_data += float(test_stats.get("data_seconds", 0.0))
        total_eval_compute += float(test_stats.get("compute_seconds", 0.0))

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

    if train_mode == "forward_forward":
        if label_encoder is None:
            raise RuntimeError("Label encoder not initialized for FF training.")
        final_test = evaluate_ff(
            model, test_loader, device, ff_cfg=ff_cfg, label_encoder=label_encoder
        )
    else:
        if loss_fn is None:
            raise RuntimeError("Loss function not initialized for backprop training.")
        final_test = evaluate(model, test_loader, loss_fn, device)
    meta_out["finished_at"] = now_iso()
    meta_out["duration_seconds"] = time.perf_counter() - started_perf
    write_json(run_dir / "meta.json", meta_out)
    write_json(
        run_dir / "summary.json",
        {
            "run_id": str(meta_out.get("run_id", run_dir.name)),
            "batch_id": str(meta_out.get("batch_id", "")),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "started_at": str(meta_out.get("started_at", "")),
            "finished_at": str(meta_out.get("finished_at", "")),
            "config_path": str(meta_out.get("config_path", "")),
            "config_hash": str(meta_out.get("config_hash", "")),
            "git_commit": str(meta_out.get("git", {}).get("commit", "")),
            "git_dirty": bool(meta_out.get("git", {}).get("dirty", False)),
            "device": str(meta_out.get("device", "")),
            "torch": str(torch.__version__),
            "overrides": meta_out.get("overrides", []),
            "train_mode": train_mode,
            "ff_label_mode": str(ff_cfg.get("label_mode", ""))
            if train_mode == "forward_forward"
            else "",
            "ff_neg_mode": str(ff_cfg.get("neg_mode", ""))
            if train_mode == "forward_forward"
            else "",
            "ff_goodness": str(ff_cfg.get("goodness", ""))
            if train_mode == "forward_forward"
            else "",
            "ff_threshold": float(ff_cfg.get("threshold", 0.0))
            if train_mode == "forward_forward"
            else None,
            "param_count": param_count,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "final_test_acc": float(final_test["acc"]),
            "final_test_loss": float(final_test["loss"]),
            "duration_seconds": float(meta_out.get("duration_seconds", 0.0)),
            "total_data_seconds": float(total_train_data + total_eval_data),
            "total_compute_seconds": float(total_train_compute + total_eval_compute),
            "train_data_seconds": float(total_train_data),
            "train_compute_seconds": float(total_train_compute),
            "eval_data_seconds": float(total_eval_data),
            "eval_compute_seconds": float(total_eval_compute),
        },
    )

    # lightweight "latest_run" pointer for convenience
    (run_root / "latest_run").unlink(missing_ok=True)
    try:
        (run_root / "latest_run").symlink_to(run_dir.name)
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

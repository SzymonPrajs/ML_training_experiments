from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torchvision import datasets, transforms

from mnist.data import MNIST_MEAN, MNIST_STD
from mnist.models import build_model


def _make_mosaic(images: list[np.ndarray], *, rows: int, cols: int, pad: int = 1) -> np.ndarray:
    if not images:
        raise ValueError("No images provided")
    h, w = images[0].shape
    canvas_h = rows * h + (rows - 1) * pad
    canvas_w = cols * w + (cols - 1) * pad
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = img
    return canvas


def _save_heatmap(
    array_2d: np.ndarray,
    *,
    path: Path,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(array_2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_feature_maps(
    tensor: torch.Tensor,
    *,
    path: Path,
    title: str,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W) tensor, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError("Expected batch size 1 for visualization")
    c = int(tensor.shape[1])
    h = int(tensor.shape[2])
    w = int(tensor.shape[3])
    rows = int(math.floor(math.sqrt(c)))
    cols = int(math.ceil(c / max(rows, 1)))
    rows = int(math.ceil(c / max(cols, 1)))

    imgs = [tensor[0, i].detach().cpu().float().numpy() for i in range(c)]
    mosaic = _make_mosaic(imgs, rows=rows, cols=cols, pad=1)
    if symmetric:
        absmax = float(np.percentile(np.abs(mosaic), 99.0))
        if not np.isfinite(absmax) or absmax == 0.0:
            absmax = float(np.max(np.abs(mosaic))) if mosaic.size else 1.0
        if absmax == 0.0:
            absmax = 1.0
        lo = -absmax
        hi = absmax
    else:
        lo = float(np.percentile(mosaic, 1.0))
        hi = float(np.percentile(mosaic, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.min(mosaic))
            hi = float(np.max(mosaic))

    _save_heatmap(
        mosaic,
        path=path,
        title=f"{title}  (C={c}, H={h}, W={w})",
        cmap=cmap,
        vmin=lo,
        vmax=hi,
    )


def _save_kernel_grid(
    weight: torch.Tensor,
    *,
    path: Path,
    title: str,
    cmap: str = "coolwarm",
) -> None:
    if weight.ndim != 4:
        raise ValueError(f"Expected conv weight (O,I,k,k), got shape={tuple(weight.shape)}")
    out_ch, in_ch, k, _ = weight.shape
    imgs = [
        weight[o, i].detach().cpu().float().numpy() for o in range(out_ch) for i in range(in_ch)
    ]
    mosaic = _make_mosaic(imgs, rows=int(out_ch), cols=int(in_ch), pad=1)
    absmax = float(np.max(np.abs(mosaic))) if mosaic.size else 1.0
    if absmax == 0:
        absmax = 1.0

    _save_heatmap(
        mosaic,
        path=path,
        title=f"{title}  (out={out_ch}, in={in_ch}, k={k})",
        cmap=cmap,
        vmin=-absmax,
        vmax=absmax,
    )


def _load_cfg_from_run(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")
    return yaml.safe_load(cfg_path.read_text())


def _load_best_state_dict(run_dir: Path) -> dict[str, Any]:
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best.pt in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", None)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")
    return state


def _get_run_name(run_dir: Path) -> str:
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            import json

            return str(json.loads(summary.read_text()).get("run_name", run_dir.name))
        except Exception:
            return run_dir.name
    return run_dir.name


def _load_test_sample(*, data_root: Path, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    raw_ds = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    raw_x, y = raw_ds[int(index)]
    if not isinstance(y, int):
        y = int(y)

    mean = torch.tensor([MNIST_MEAN]).view(1, 1, 1)
    std = torch.tensor([MNIST_STD]).view(1, 1, 1)
    x_norm = (raw_x - mean) / std
    return raw_x, x_norm, y


def generate_kernel_plots(model: nn.Module, *, out_dir: Path, run_name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    convs: list[tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            convs.append((name, module))

    written: list[Path] = []
    for idx, (name, conv) in enumerate(convs, start=1):
        weight = conv.weight.detach().cpu()
        path = out_dir / f"conv{idx}_kernels.png"
        _loadable_name = name.replace(".", "/")
        _save_kernel_grid(weight, path=path, title=f"{run_name} | conv{idx} ({_loadable_name})")
        written.append(path)
    return written


def generate_activation_plots(
    model: nn.Module,
    *,
    data_root: Path,
    out_dir: Path,
    run_name: str,
    sample_index: int = 0,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_x, x_norm, y = _load_test_sample(data_root=data_root, index=sample_index)
    x = x_norm.unsqueeze(0)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(raw_x[0].cpu().numpy(), cmap="gray", interpolation="nearest")
    ax.set_title(f"test[{sample_index}] label={y}")
    ax.axis("off")
    input_path = out_dir / "input.png"
    fig.tight_layout()
    fig.savefig(input_path, dpi=200)
    plt.close(fig)

    captures: list[tuple[str, torch.Tensor]] = []
    hooks: list[Any] = []

    def _capture(name: str):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                captures.append((name, out.detach().cpu()))

        return fn

    # Capture conv outputs (per-kernel feature maps), pooling outputs, then avgpool.
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_capture(f"conv:{name}")))
        elif isinstance(module, nn.MaxPool2d):
            hooks.append(module.register_forward_hook(_capture(f"maxpool:{name}")))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            hooks.append(module.register_forward_hook(_capture(f"avgpool:{name}")))

    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    written: list[Path] = [input_path]
    conv_i = 0
    pool_i = 0
    for kind_name, t in captures:
        if t.ndim != 4:
            continue
        if kind_name.startswith("conv:"):
            conv_i += 1
            path = out_dir / f"conv{conv_i}_convolved.png"
            _save_feature_maps(
                t,
                path=path,
                title=f"{run_name} | conv{conv_i} ({kind_name.split(':',1)[1]})",
                cmap="coolwarm",
                symmetric=True,
            )
            written.append(path)
            continue

        if kind_name.startswith("maxpool:"):
            pool_i += 1
            path = out_dir / f"maxpool{pool_i}_output.png"
            _save_feature_maps(
                t,
                path=path,
                title=f"{run_name} | maxpool{pool_i} ({kind_name.split(':',1)[1]})",
                cmap="coolwarm",
                symmetric=True,
            )
            written.append(path)
            continue

        if kind_name.startswith("avgpool:"):
            # (1,C,1,1): show as a simple bar chart.
            if t.shape[0] != 1 or t.shape[2] != 1 or t.shape[3] != 1:
                continue
            vals = t[0, :, 0, 0].float().numpy()
            fig, ax = plt.subplots(figsize=(7.5, 2.8))
            ax.bar(np.arange(vals.shape[0]), vals)
            ax.set_title(f"{run_name} | avgpool ({kind_name.split(':',1)[1]})")
            ax.set_xlabel("channel")
            ax.set_ylabel("value")
            ax.grid(True, axis="y", alpha=0.3)
            path = out_dir / "avgpool_values.png"
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)

    return written


def _save_class_weight_maps(
    weight: torch.Tensor,
    *,
    size: int,
    path: Path,
    title: str,
    cmap: str = "coolwarm",
) -> None:
    if weight.ndim != 2:
        raise ValueError(f"Expected (C,D) weight matrix, got shape={tuple(weight.shape)}")
    num_classes = int(weight.shape[0])
    if int(weight.shape[1]) != int(size) * int(size):
        raise ValueError(
            f"Expected D={int(size)*int(size)} for size={size}, got D={int(weight.shape[1])}"
        )
    imgs = [
        weight[c].detach().cpu().float().reshape(int(size), int(size)).numpy()
        for c in range(num_classes)
    ]
    rows = 2 if num_classes == 10 else int(math.floor(math.sqrt(num_classes)))
    cols = int(math.ceil(num_classes / max(rows, 1)))
    mosaic = _make_mosaic(imgs, rows=rows, cols=cols, pad=1)
    absmax = float(np.max(np.abs(mosaic))) if mosaic.size else 1.0
    if absmax == 0.0:
        absmax = 1.0
    _save_heatmap(
        mosaic,
        path=path,
        title=title,
        cmap=cmap,
        vmin=-absmax,
        vmax=absmax,
    )


def generate_manual_feature_plots(
    model: nn.Module,
    *,
    data_root: Path,
    out_dir: Path,
    run_name: str,
    sample_index: int = 0,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_x, x_norm, y = _load_test_sample(data_root=data_root, index=sample_index)
    x = x_norm.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.imshow(raw_x[0].cpu().numpy(), cmap="gray", interpolation="nearest")
    ax.set_title(f"test[{sample_index}] label={y} pred={pred}")
    ax.axis("off")
    input_path = out_dir / "input.png"
    fig.tight_layout()
    fig.savefig(input_path, dpi=200)
    plt.close(fig)

    parts: dict[str, torch.Tensor] = {}
    if hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "extract_parts"):
        with torch.no_grad():
            parts = model.feature_extractor.extract_parts(x)

    written: list[Path] = [input_path]

    # Visualize pooled patches as small heatmaps.
    for name, t in parts.items():
        if not name.startswith("avg"):
            continue
        try:
            size = int(name.replace("avg", ""))
        except Exception:
            continue
        if t.ndim != 2 or t.shape[0] != 1 or t.shape[1] != size * size:
            continue
        arr = t[0].detach().cpu().float().reshape(size, size).numpy()
        path = out_dir / f"features_{name}.png"
        _save_heatmap(
            arr,
            path=path,
            title=f"{run_name} | {name} feature map",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        written.append(path)

    # Projection curves (row/col means).
    if "row_mean" in parts and "col_mean" in parts:
        row = parts["row_mean"][0].detach().cpu().float().numpy()
        col = parts["col_mean"][0].detach().cpu().float().numpy()
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.8, 4.2), sharex=False)
        axes[0].plot(np.arange(row.shape[0]), row)
        axes[0].set_title(f"{run_name} | row_mean")
        axes[0].set_xlabel("row")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(np.arange(col.shape[0]), col)
        axes[1].set_title(f"{run_name} | col_mean")
        axes[1].set_xlabel("col")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "features_projections.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    # Moment features as a tiny bar chart.
    if "moments" in parts and parts["moments"].ndim == 2 and parts["moments"].shape[1] == 5:
        vals = parts["moments"][0].detach().cpu().float().numpy()
        labels = ["mass_mean", "x_mean", "y_mean", "x_var", "y_var"]
        fig, ax = plt.subplots(figsize=(7.8, 2.8))
        ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"{run_name} | moments")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / "features_moments.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    # Visualize per-class weights for pooled maps (if available).
    if (
        hasattr(model, "classifier")
        and isinstance(model.classifier, nn.Linear)
        and hasattr(model, "feature_extractor")
        and hasattr(model.feature_extractor, "feature_slices")
    ):
        weight = model.classifier.weight.detach().cpu()
        slices = model.feature_extractor.feature_slices
        for key in slices:
            if not key.startswith("avg"):
                continue
            try:
                size = int(key.replace("avg", ""))
            except Exception:
                continue
            sl = slices[key]
            w_part = weight[:, sl]
            path = out_dir / f"weights_{key}.png"
            _save_class_weight_maps(
                w_part,
                size=size,
                path=path,
                title=f"{run_name} | classifier weights for {key} (class 0..9)",
            )
            written.append(path)

    return written


def generate_run_viz(run_dir: Path, *, sample_index: int = 0) -> Path:
    cfg = _load_cfg_from_run(run_dir)
    run_name = _get_run_name(run_dir)
    model = build_model(cfg["model"])
    model.load_state_dict(_load_best_state_dict(run_dir))

    data_root = Path(cfg["data"]["root"])
    viz_root = run_dir / "viz"
    kernels_dir = viz_root / "kernels"
    activations_dir = viz_root / "activations"

    model_name = str(cfg.get("model", {}).get("name", ""))
    if model_name == "manual_features_linear":
        generate_manual_feature_plots(
            model,
            data_root=data_root,
            out_dir=viz_root / "manual_features",
            run_name=run_name,
            sample_index=sample_index,
        )
    else:
        generate_kernel_plots(model, out_dir=kernels_dir, run_name=run_name)
        generate_activation_plots(
            model,
            data_root=data_root,
            out_dir=activations_dir,
            run_name=run_name,
            sample_index=sample_index,
        )
    return viz_root

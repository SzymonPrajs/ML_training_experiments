from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MorphOp:
    dilate: int
    erode: int
    shift_x: int
    shift_y: int


def build_morph_ops(
    num_classes: int,
    *,
    seed: int,
    max_dilate: int,
    max_erode: int,
    max_shift: int,
) -> list[MorphOp]:
    shifts = list(range(-max_shift, max_shift + 1)) if max_shift > 0 else [0]
    combos = [
        (d, e, sx, sy)
        for d in range(max_dilate + 1)
        for e in range(max_erode + 1)
        for sx in shifts
        for sy in shifts
    ]
    if not combos:
        combos = [(0, 0, 0, 0)]
    rng = random.Random(seed)
    rng.shuffle(combos)
    ops: list[MorphOp] = []
    for i in range(num_classes):
        d, e, sx, sy = combos[i % len(combos)]
        ops.append(MorphOp(dilate=d, erode=e, shift_x=sx, shift_y=sy))
    return ops


def apply_morph(x: torch.Tensor, op: MorphOp) -> torch.Tensor:
    out = x
    for _ in range(op.dilate):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    for _ in range(op.erode):
        out = -F.max_pool2d(-out, kernel_size=3, stride=1, padding=1)
    if op.shift_x != 0 or op.shift_y != 0:
        out = torch.roll(out, shifts=(op.shift_y, op.shift_x), dims=(2, 3))
    return out


def make_fourier_templates(
    num_classes: int,
    *,
    height: int,
    width: int,
    terms: int,
    max_freq: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    templates: list[torch.Tensor] = []
    for label in range(num_classes):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + label * 9973)
        freqs = torch.randint(1, max_freq + 1, (terms, 2), generator=gen, device=device)
        phases = torch.rand((terms,), generator=gen, device=device, dtype=dtype) * 2.0 * math.pi
        pattern = torch.zeros((height, width), device=device, dtype=dtype)
        for k in range(terms):
            fx = float(freqs[k, 0].item())
            fy = float(freqs[k, 1].item())
            pattern = pattern + torch.sin(2.0 * math.pi * (fx * grid_x + fy * grid_y) + phases[k])
        pattern = pattern - pattern.mean()
        pattern = pattern / (pattern.std() + 1e-6)
        templates.append(pattern)
    stacked = torch.stack(templates, dim=0).unsqueeze(1)
    return stacked


class LabelEncoder:
    def __init__(
        self,
        *,
        mode: str,
        num_classes: int,
        image_shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        label_scale: float,
        overlay_size: int,
        fourier_terms: int,
        fourier_max_freq: int,
        fourier_seed: int,
        morph_seed: int,
        morph_dilate: int,
        morph_erode: int,
        morph_shift: int,
        morph_scale: float,
    ) -> None:
        self.mode = mode
        self.num_classes = num_classes
        self.label_scale = float(label_scale)
        self.overlay_size = int(overlay_size)
        self.morph_scale = float(morph_scale)
        self.device = device
        self.dtype = dtype
        self._templates: torch.Tensor | None = None
        self._morph_ops: list[MorphOp] | None = None

        _, height, width = image_shape
        if mode == "spatial_fourier":
            self._templates = make_fourier_templates(
                num_classes,
                height=height,
                width=width,
                terms=fourier_terms,
                max_freq=fourier_max_freq,
                seed=fourier_seed,
                device=device,
                dtype=dtype,
            )
        if mode == "spatial_morph":
            self._morph_ops = build_morph_ops(
                num_classes,
                seed=morph_seed,
                max_dilate=morph_dilate,
                max_erode=morph_erode,
                max_shift=morph_shift,
            )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if self.mode == "overlay":
            return overlay_label(x, y, scale=self.label_scale, size=self.overlay_size)
        if self.mode == "spatial_fourier":
            if self._templates is None:
                return x
            labels = y.long().clamp(0, self.num_classes - 1)
            pattern = self._templates[labels]
            return x + self.label_scale * pattern
        if self.mode == "spatial_morph":
            if self._morph_ops is None:
                return x
            labels = y.long().clamp(0, self.num_classes - 1)
            out = x.clone()
            for i in range(x.shape[0]):
                op = self._morph_ops[int(labels[i])]
                transformed = apply_morph(x[i : i + 1], op)
                if self.morph_scale != 1.0:
                    transformed = x[i : i + 1] + self.morph_scale * (transformed - x[i : i + 1])
                out[i : i + 1] = transformed
            return out
        raise ValueError(f"Unknown label mode: {self.mode}")


def overlay_label(x: torch.Tensor, y: torch.Tensor, *, scale: float, size: int) -> torch.Tensor:
    out = x.clone()
    batch, _, height, width = out.shape
    patch_h = min(size, height)
    patch_w = min(size, width)
    rows = y.long().clamp(0, patch_h - 1)
    cols = torch.arange(patch_w, device=out.device)
    batch_idx = torch.arange(batch, device=out.device)
    out[batch_idx, 0, rows, :patch_w] = out[batch_idx, 0, rows, :patch_w] + scale
    return out


def pick_wrong_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 1:
        return y.clone()
    rand = torch.randint(1, num_classes, size=y.shape, device=y.device)
    return (y + rand) % num_classes


def permute_no_fixed_points(batch: int, *, device: torch.device) -> torch.Tensor:
    if batch <= 1:
        return torch.arange(batch, device=device)
    perm = torch.randperm(batch, device=device)
    fixed = perm == torch.arange(batch, device=device)
    if fixed.any():
        perm = (perm + 1) % batch
    return perm


def goodness_from_acts(acts: list[torch.Tensor], mode: str) -> torch.Tensor:
    total = None
    for act in acts:
        g = layer_goodness(act, mode)
        total = g if total is None else (total + g)
    if total is None:
        raise ValueError("No activations for goodness computation.")
    return total


def layer_goodness(act: torch.Tensor, mode: str) -> torch.Tensor:
    flat = act.view(act.shape[0], -1)
    if mode == "sum_sq":
        return (flat * flat).mean(dim=1)
    if mode == "ed":
        v2 = flat * flat
        denom = v2.sum(dim=1, keepdim=True).clamp_min(1e-8)
        p = v2 / denom
        return 1.0 / (p * p).sum(dim=1).clamp_min(1e-8)
    raise ValueError(f"Unknown goodness mode: {mode}")


def ff_loss(
    acts_pos: list[torch.Tensor],
    acts_neg: list[torch.Tensor],
    *,
    threshold: float,
    beta: float,
    goodness_mode: str,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for pos, neg in zip(acts_pos, acts_neg, strict=False):
        g_pos = layer_goodness(pos, goodness_mode)
        g_neg = layer_goodness(neg, goodness_mode)
        loss_pos = F.softplus(beta * (threshold - g_pos))
        loss_neg = F.softplus(beta * (g_neg - threshold))
        losses.append((loss_pos + loss_neg).mean())
    if not losses:
        raise ValueError("No losses computed (empty activations).")
    return torch.stack(losses).mean()


def add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return x
    return x + noise_std * torch.randn_like(x)

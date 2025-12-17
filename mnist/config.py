from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1,
    "device": "auto",  # auto|mps|cuda|cpu
    "data": {
        "root": "data",
        "batch_size": 256,
        "num_workers": 2,
        "pin_memory": True,
        "augment": {
            "enabled": True,
            "degrees": 10,
            "translate": 0.1,
            "scale": 0.1,
        },
    },
    "model": {
        "name": "tiny_cnn",
        "width1": 10,
        "width2": 20,
        "width3": 40,
        "dropout": 0.0,
        "extra_3x3": True,
    },
    "train": {
        "epochs": 20,
        "label_smoothing": 0.0,
        "grad_clip_norm": 1.0,
    },
    "optimizer": {
        "name": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 5e-4,
    },
    "scheduler": {"name": "cosine", "warmup_epochs": 1, "min_lr": 0.0},
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = deepcopy(base)
    for key, value in update.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def set_by_dotted_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: dict[str, Any] = cfg
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item}")
        key, raw_value = item.split("=", 1)
        set_by_dotted_path(out, key.strip(), _parse_override_value(raw_value.strip()))
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    user_cfg: dict[str, Any] = {}
    if config_path.exists():
        user_cfg = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError(f"Config must be a mapping: {config_path}")

    cfg = deep_merge(DEFAULT_CONFIG, user_cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg

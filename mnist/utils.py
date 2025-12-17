from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device=mps but MPS is not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested}")


def now_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_unique_run_dir(root: Path, run_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_name
    if not candidate.exists():
        return candidate
    suffix = now_timestamp()
    return root / f"{run_name}_{suffix}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@dataclass
class EpochTimer:
    start_time: float | None = None

    def __enter__(self) -> "EpochTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        pass

    @property
    def seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time


def get_env_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }

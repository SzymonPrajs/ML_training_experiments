from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
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


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_unique_run_dir(root: Path, run_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_name
    if not candidate.exists():
        return candidate
    suffix = now_timestamp()
    return root / f"{run_name}_{suffix}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def stable_sha256(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def get_git_summary(cwd: Path | None = None) -> dict[str, Any]:
    try:
        cwd = Path.cwd() if cwd is None else cwd
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(cwd), text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(cwd), text=True
        ).strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {}


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


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
        return
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass

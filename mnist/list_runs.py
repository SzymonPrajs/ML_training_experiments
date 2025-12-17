from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mnist.compare import render_markdown_table


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List available MNIST run batches and runs.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--batch", type=str, default=None, help="Batch id to list runs for.")
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args()


def _list_batches(runs_dir: Path, limit: int) -> None:
    batches_csv = runs_dir / "batches.csv"
    rows: list[dict[str, Any]] = []
    if batches_csv.exists():
        rows = _read_csv(batches_csv)
        rows = rows[::-1]  # newest last appended -> show newest first
    else:
        for child in sorted(runs_dir.iterdir(), reverse=True):
            batch_json = child / "batch.json"
            if child.is_dir() and batch_json.exists():
                meta = _read_json(batch_json)
                rows.append(
                    {
                        "batch_id": str(meta.get("batch_id", child.name)),
                        "started_at": str(meta.get("started_at", "")),
                        "finished_at": str(meta.get("finished_at", "")),
                        "num_runs": str(len(meta.get("runs", []))),
                        "overrides": " ".join(meta.get("overrides", [])),
                    }
                )

    rows = rows[: max(0, int(limit))]
    headers = ["batch_id", "started_at", "finished_at", "duration_seconds", "num_runs", "overrides"]
    table_rows = [[str(r.get(h, "")) for h in headers] for r in rows]
    print(render_markdown_table(headers, table_rows))


def _list_runs(runs_dir: Path, batch_id: str) -> None:
    batch_dir = runs_dir / batch_id if not Path(batch_id).exists() else Path(batch_id)
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch not found: {batch_id}")

    runs_csv = batch_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"Missing runs.csv in batch: {batch_dir}")

    rows = _read_csv(runs_csv)
    headers = [
        "run_name",
        "run_id",
        "started_at",
        "params",
        "best_acc",
        "best_epoch",
        "duration_s",
        "final_acc",
        "final_loss",
        "config",
        "overrides",
    ]
    table_rows: list[list[str]] = []
    for r in rows:
        config = str(r.get("config_path", "")).split("/")[-1]
        overrides = r.get("overrides", [])
        if isinstance(overrides, list):
            overrides = " ".join(str(x) for x in overrides)
        try:
            best_acc = f"{float(r.get('best_test_acc', 0.0)) * 100:.2f}%"
        except Exception:
            best_acc = str(r.get("best_test_acc", ""))
        try:
            final_acc = f"{float(r.get('final_test_acc', 0.0)) * 100:.2f}%"
        except Exception:
            final_acc = str(r.get("final_test_acc", ""))
        table_rows.append(
            [
                str(r.get("run_name", "")),
                str(r.get("run_id", "")),
                str(r.get("started_at", "")),
                str(r.get("param_count", "")),
                best_acc,
                str(r.get("best_epoch", "")),
                f"{float(r.get('duration_seconds', 0.0)):.2f}",
                final_acc,
                str(r.get("final_test_loss", "")),
                config,
                str(overrides),
            ]
        )
    print(render_markdown_table(headers, table_rows))


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if args.batch:
        _list_runs(runs_dir, str(args.batch))
    else:
        _list_batches(runs_dir, int(args.limit))


if __name__ == "__main__":
    main()

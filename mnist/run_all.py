from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from mnist.compare import compare_runs, load_summary, plot_time_to_acc
from mnist.config import load_config
from mnist.train import train_run
from mnist.utils import get_git_summary, now_iso, now_timestamp, write_json
from mnist.viz import generate_run_viz


def _discover_configs(configs_dir: Path, pattern: str) -> list[Path]:
    return sorted(p for p in configs_dir.glob(pattern) if p.is_file())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all configs in a folder sequentially, then print a summary table."
    )
    p.add_argument("--configs-dir", type=str, default="configs")
    p.add_argument("--pattern", type=str, default="*.yaml")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Optional label appended to the batch id.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete runs/ and reports/ contents before running.",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config for every run with dot paths, e.g. --set train.epochs=5",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining configs if one fails.",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip per-run kernel/activation visualization plots.",
    )
    p.add_argument(
        "--viz-sample-index",
        type=int,
        default=0,
        help="MNIST test-set index to visualize (default: 0).",
    )
    return p.parse_args()


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _sanitize_label(label: str) -> str:
    label = label.strip().replace(" ", "_")
    return "".join(ch for ch in label if ch.isalnum() or ch in {"_", "-", "."})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return

    # If new keys appear, rewrite the file with a union header to keep it readable.
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_headers = list(reader.fieldnames or [])
        existing_rows = list(reader)

    new_headers = list(dict.fromkeys(existing_headers + list(row.keys())))
    if new_headers == existing_headers:
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=existing_headers)
            w.writerow(row)
        return

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_headers)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        w.writerow(row)


def main() -> None:
    args = parse_args()
    configs_dir = Path(args.configs_dir)
    runs_dir = Path(args.runs_dir)
    reports_dir = Path(args.reports_dir)

    configs = _discover_configs(configs_dir, args.pattern)
    if not configs:
        raise FileNotFoundError(f"No configs found in {configs_dir} (pattern={args.pattern})")

    if args.clean:
        _safe_rmtree(runs_dir)
        _safe_rmtree(reports_dir)

    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    batch_id = now_timestamp()
    if args.batch_name:
        label = _sanitize_label(args.batch_name)
        if label:
            batch_id = f"{batch_id}_{label}"

    batch_dir = runs_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)

    batch_started_at = now_iso()
    batch_started_perf = time.perf_counter()
    git = get_git_summary()

    print(f"Found {len(configs)} config(s) in {configs_dir}")
    print(f"Batch: {batch_id}")
    run_dirs: list[Path] = []
    failures: list[dict[str, Any]] = []
    viz_failures: list[dict[str, Any]] = []

    for i, config_path in enumerate(configs, start=1):
        run_name = config_path.stem
        print(f"\n[{i}/{len(configs)}] {run_name}  ({config_path})")
        try:
            overrides = list(args.set or [])
            cfg = load_config(config_path, overrides=overrides)
            run_id = f"{batch_id}/{run_name}"
            run_dir = train_run(
                cfg,
                run_name=run_name,
                run_root=batch_dir,
                meta={
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "config_path": str(config_path),
                    "overrides": overrides,
                },
            )
            run_dirs.append(run_dir)

            summary = load_summary(run_dir / "summary.json")
            best_acc = float(summary.get("best_test_acc", 0.0)) * 100.0
            params = int(summary.get("param_count", 0))
            print(f"Done: {run_dir} | best_acc={best_acc:.2f}% | params={params}")

            if not args.no_viz:
                try:
                    viz_root = generate_run_viz(
                        run_dir, sample_index=int(args.viz_sample_index)
                    )
                    dest = reports_dir / "viz" / run_name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(viz_root, dest)
                    print(f"Viz: {dest}")
                except Exception as exc:
                    viz_failures.append(
                        {
                            "run_name": run_name,
                            "run_dir": str(run_dir),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"VIZ FAILED: {run_name} -> {exc}", file=sys.stderr)
        except Exception as exc:
            failures.append(
                {
                    "config": str(config_path),
                    "run_name": run_name,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAILED: {run_name} ({config_path})", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if not args.continue_on_error:
                break

    batch_finished_at: str | None = None
    if run_dirs:
        curve_plot = batch_dir / "compare.png"
        pareto_plot = batch_dir / "pareto.png"
        time_plot = batch_dir / "time_to_acc.png"
        time_plot_compute = batch_dir / "time_to_acc_compute.png"
        print("\n=== Summary ===")
        table = compare_runs(
            run_dirs,
            metrics=["test_acc", "test_loss", "train_loss", "lr"],
            plot_path=curve_plot,
            pareto_path=pareto_plot,
            label_key="run_name",
        )
        plot_time_to_acc(run_dirs, time_plot, label_key="run_name")
        plot_time_to_acc(
            run_dirs, time_plot_compute, label_key="run_name", time_key="compute_seconds"
        )
        print(table)

        batch_finished_at = now_iso()
        summary_md = (
            f"# MNIST batch `{batch_id}`\n\n"
            f"- started_at: `{batch_started_at}`\n"
            f"- finished_at: `{batch_finished_at}`\n"
            f"- configs_dir: `{configs_dir}`\n"
            f"- pattern: `{args.pattern}`\n"
            f"- overrides: `{ ' '.join(list(args.set or [])) }`\n"
            f"- viz: enabled={str(not args.no_viz).lower()} sample_index={int(args.viz_sample_index)} (see `reports/viz/`)\n"
            f"- git: `{git.get('commit', '')}` dirty={git.get('dirty', False)}\n\n"
            f"{table}\n"
        )
        (batch_dir / "summary.md").write_text(summary_md)
        shutil.copyfile(curve_plot, reports_dir / "compare.png")
        shutil.copyfile(pareto_plot, reports_dir / "pareto.png")
        shutil.copyfile(time_plot, reports_dir / "time_to_acc.png")
        if time_plot_compute.exists():
            shutil.copyfile(time_plot_compute, reports_dir / "time_to_acc_compute.png")
        (reports_dir / "summary.md").write_text(summary_md)

        summaries = [load_summary(r / "summary.json") for r in run_dirs]
        _write_csv(batch_dir / "runs.csv", summaries)
        shutil.copyfile(batch_dir / "runs.csv", reports_dir / "runs.csv")

    if failures:
        print("\n=== Failures ===", file=sys.stderr)
        for f in failures:
            print(f"- {f['run_name']}: {f['config']} -> {f['error']}", file=sys.stderr)
        raise SystemExit(1)

    if batch_finished_at is None:
        batch_finished_at = now_iso()
    batch_meta = {
        "batch_id": batch_id,
        "batch_name": str(args.batch_name or ""),
        "started_at": batch_started_at,
        "finished_at": batch_finished_at,
        "duration_seconds": time.perf_counter() - batch_started_perf,
        "configs_dir": str(configs_dir),
        "pattern": str(args.pattern),
        "overrides": list(args.set or []),
        "git": git,
        "argv": list(sys.argv),
        "configs": [str(p) for p in configs],
        "runs": [str(p.relative_to(batch_dir)) for p in run_dirs],
        "viz": {
            "enabled": (not args.no_viz),
            "sample_index": int(args.viz_sample_index),
            "failures": viz_failures,
        },
    }
    write_json(batch_dir / "batch.json", batch_meta)
    shutil.copyfile(batch_dir / "batch.json", reports_dir / "batch.json")

    _append_csv(
        runs_dir / "batches.csv",
        {
            "batch_id": batch_id,
            "started_at": batch_started_at,
            "finished_at": batch_finished_at,
            "duration_seconds": f"{batch_meta['duration_seconds']:.3f}",
            "num_runs": str(len(run_dirs)),
            "overrides": " ".join(list(args.set or [])),
            "batch_name": str(args.batch_name or ""),
            "pattern": str(args.pattern),
            "git_commit": str(git.get("commit", "")),
            "git_dirty": str(bool(git.get("dirty", False))),
        },
    )

    # update runs/latest -> <batch_id>
    latest = runs_dir / "latest"
    latest.unlink(missing_ok=True)
    try:
        latest.symlink_to(batch_id)
    except Exception:
        pass


if __name__ == "__main__":
    main()

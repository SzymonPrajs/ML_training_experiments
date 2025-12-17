from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

from mnist.compare import compare_runs, load_summary
from mnist.config import load_config
from mnist.train import train_run


def _discover_configs(configs_dir: Path, pattern: str) -> list[Path]:
    return sorted(p for p in configs_dir.glob(pattern) if p.is_file())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all configs in a folder sequentially, then print a summary table."
    )
    p.add_argument("--configs-dir", type=str, default="configs")
    p.add_argument("--pattern", type=str, default="*.yaml")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--run-prefix", type=str, default="")
    p.add_argument("--run-suffix", type=str, default="")
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
    p.add_argument("--plot", type=str, default=None, help="Write plot PNG path (optional).")
    p.add_argument(
        "--metric",
        type=str,
        default="test_acc",
        help="Metric to plot from metrics.csv (default: test_acc).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs_dir = Path(args.configs_dir)
    runs_dir = Path(args.runs_dir)

    configs = _discover_configs(configs_dir, args.pattern)
    if not configs:
        raise FileNotFoundError(f"No configs found in {configs_dir} (pattern={args.pattern})")

    print(f"Found {len(configs)} config(s) in {configs_dir}")
    run_dirs: list[Path] = []
    failures: list[dict[str, Any]] = []

    for i, config_path in enumerate(configs, start=1):
        run_name = f"{args.run_prefix}{config_path.stem}{args.run_suffix}"
        print(f"\n[{i}/{len(configs)}] {run_name}  ({config_path})")
        try:
            cfg = load_config(config_path, overrides=list(args.set or []))
            run_dir = train_run(cfg, run_name=run_name, run_root=runs_dir)
            run_dirs.append(run_dir)

            summary = load_summary(run_dir / "summary.json")
            best_acc = float(summary.get("best_test_acc", 0.0)) * 100.0
            params = int(summary.get("param_count", 0))
            print(f"Done: {run_dir} | best_acc={best_acc:.2f}% | params={params}")
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

    if run_dirs:
        plot_path = Path(args.plot) if args.plot else None
        print("\n=== Summary ===")
        print(compare_runs(run_dirs, metric=args.metric, plot_path=plot_path))
        if plot_path:
            print(f"Wrote plot: {plot_path}")

    if failures:
        print("\n=== Failures ===", file=sys.stderr)
        for f in failures:
            print(f"- {f['run_name']}: {f['config']} -> {f['error']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

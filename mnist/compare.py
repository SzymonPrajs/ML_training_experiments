from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append({k: _maybe_float(v) for k, v in row.items()})
        return rows


def _maybe_float(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        if v.strip() == "":
            return v
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except Exception:
        return v


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def compare_runs(
    run_dirs: list[Path],
    *,
    metric: str = "test_acc",
    plot_path: Path | None = None,
) -> str:
    summaries: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json in {run_dir}")
        summaries.append(load_summary(summary_path))
        if metrics_path.exists():
            curves[run_dir.name] = load_metrics(metrics_path)

    headers = ["run", "params", "best_acc", "best_epoch", "final_acc", "final_loss"]
    rows: list[list[str]] = []
    for s in summaries:
        rows.append(
            [
                Path(s.get("run_dir", "")).name or "run",
                str(s.get("param_count", "")),
                f"{float(s.get('best_test_acc', 0.0))*100:.2f}%",
                str(s.get("best_epoch", "")),
                f"{float(s.get('final_test_acc', 0.0))*100:.2f}%",
                f"{float(s.get('final_test_loss', 0.0)):.4f}",
            ]
        )

    table = render_markdown_table(headers, rows)

    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        for run_name, metrics_rows in curves.items():
            if not metrics_rows:
                continue
            xs = [int(m["epoch"]) for m in metrics_rows if "epoch" in m]
            ys = [float(m.get(metric, 0.0)) for m in metrics_rows]
            plt.plot(xs, ys, label=run_name)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} vs epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)

    return table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MNIST runs (table + optional plot).")
    p.add_argument("--runs", nargs="+", required=True, help="Run directories under runs/")
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
    run_dirs = [Path(r) if Path(r).exists() else Path("runs") / r for r in args.runs]
    plot_path = Path(args.plot) if args.plot else None
    table = compare_runs(run_dirs, metric=args.metric, plot_path=plot_path)
    print(table)
    if plot_path:
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()

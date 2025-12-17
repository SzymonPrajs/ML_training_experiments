from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


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
    metrics: list[str] | None = None,
    plot_path: Path | None = None,
    pareto_path: Path | None = None,
    label_key: str = "run_id",
) -> str:
    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary.json in {run_dir}")
        summary = load_summary(summary_path)
        base_label = str(
            summary.get(label_key)
            or summary.get("run_name")
            or summary.get("run_id")
            or Path(summary.get("run_dir", "")).name
            or run_dir.name
        )
        entries.append(
            {
                "run_dir": run_dir,
                "summary": summary,
                "base_label": base_label,
                "metrics": load_metrics(metrics_path) if metrics_path.exists() else None,
            }
        )

    base_counts = Counter(e["base_label"] for e in entries)
    used_labels: set[str] = set()
    for e in entries:
        base = str(e["base_label"])
        summary = e["summary"]
        if base_counts[base] == 1:
            label = base
        else:
            batch_id = str(summary.get("batch_id", "")).strip()
            if batch_id:
                label = f"{base} [{batch_id}]"
            else:
                label = base

        if label in used_labels:
            i = 2
            candidate = f"{label}#{i}"
            while candidate in used_labels:
                i += 1
                candidate = f"{label}#{i}"
            label = candidate
        used_labels.add(label)
        summary["_label"] = label

    summaries = [e["summary"] for e in entries]
    curves: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        label = str(e["summary"].get("_label", "run"))
        if e["metrics"]:
            curves[label] = e["metrics"]

    summaries = sorted(
        summaries, key=lambda s: float(s.get("best_test_acc", 0.0)), reverse=True
    )

    ordered_curves: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        label = str(s.get("_label", "run"))
        if label in curves:
            ordered_curves[label] = curves[label]
    if ordered_curves:
        curves = ordered_curves

    headers = [
        "run",
        "batch",
        "started_at",
        "device",
        "params",
        "best_acc",
        "best_epoch",
        "duration_s",
        "final_acc",
        "final_loss",
    ]
    rows: list[list[str]] = []
    for s in summaries:
        try:
            duration_s = f"{float(s.get('duration_seconds', 0.0)):.1f}"
        except Exception:
            duration_s = str(s.get("duration_seconds", ""))
        rows.append(
            [
                str(s.get("_label") or "run"),
                str(s.get("batch_id", "")),
                str(s.get("started_at", "")),
                str(s.get("device", "")),
                str(s.get("param_count", "")),
                f"{float(s.get('best_test_acc', 0.0))*100:.2f}%",
                str(s.get("best_epoch", "")),
                duration_s,
                f"{float(s.get('final_test_acc', 0.0))*100:.2f}%",
                f"{float(s.get('final_test_loss', 0.0)):.4f}",
            ]
        )

    table = render_markdown_table(headers, rows)

    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_to_plot = metrics or ["test_acc", "test_loss", "lr"]
        _plot_curves(curves, metrics_to_plot, plot_path)

    if pareto_path:
        pareto_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_pareto(summaries, pareto_path, label_key=label_key)

    return table


def plot_time_to_acc(
    run_dirs: list[Path], plot_path: Path, *, label_key: str = "run_id"
) -> None:
    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        summary = load_summary(summary_path)
        base_label = str(
            summary.get(label_key)
            or summary.get("run_name")
            or summary.get("run_id")
            or Path(summary.get("run_dir", "")).name
            or run_dir.name
        )
        rows = load_metrics(metrics_path)
        rows = sorted((r for r in rows if "epoch" in r), key=lambda r: int(r["epoch"]))
        if not rows:
            continue
        entries.append({"summary": summary, "base_label": base_label, "rows": rows})

    if not entries:
        return

    base_counts = Counter(e["base_label"] for e in entries)
    used_labels: set[str] = set()
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for e in entries:
        base = str(e["base_label"])
        summary = e["summary"]
        if base_counts[base] == 1:
            label = base
        else:
            batch_id = str(summary.get("batch_id", "")).strip()
            if batch_id:
                label = f"{base} [{batch_id}]"
            else:
                label = base

        if label in used_labels:
            i = 2
            candidate = f"{label}#{i}"
            while candidate in used_labels:
                i += 1
                candidate = f"{label}#{i}"
            label = candidate
        used_labels.add(label)

        xs: list[float] = []
        ys: list[float] = []
        total = 0.0
        for r in e["rows"]:
            total += float(r.get("seconds", 0.0))
            xs.append(total)
            ys.append(float(r.get("test_acc", 0.0)))
        curves[label] = (xs, ys)

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, linewidth=1.8, label=label)
        ax.plot(xs[-1], ys[-1], marker="o", markersize=5)

    ax.set_title("test_acc vs time")
    ax.set_xlabel("cumulative seconds")
    ax.set_ylabel("test_acc")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = min(3, max(1, len(labels)))
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _plot_curves(curves: dict[str, list[dict[str, Any]]], metrics: list[str], plot_path: Path) -> None:
    metrics = [m.strip() for m in metrics if m.strip()]
    if not metrics:
        return

    n_metrics = len(metrics)
    ncols = 2 if n_metrics > 1 else 1
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        figsize=(8 * ncols, 4.2 * nrows),
    )
    if isinstance(axes, list):
        axes_flat = axes
    else:
        try:
            axes_flat = list(axes.ravel())
        except Exception:
            axes_flat = [axes]

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    run_labels = list(curves.keys())
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = (
        color_cycle.by_key().get("color", []) if color_cycle is not None else []
    )
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(run_labels)}

    all_epochs: list[int] = []
    for run_label, metrics_rows in curves.items():
        if not metrics_rows:
            continue
        sorted_rows = sorted(
            (m for m in metrics_rows if "epoch" in m),
            key=lambda m: int(m["epoch"]),
        )
        xs = [int(m["epoch"]) for m in sorted_rows]
        all_epochs.extend(xs)
        for idx, metric in enumerate(metrics):
            ys = [float(m.get(metric, 0.0)) for m in sorted_rows]
            ax = axes_flat[idx]
            label = run_label if idx == 0 else None
            color = color_map.get(run_label, None)
            if len(xs) <= 1:
                ax.plot(xs, ys, marker="o", linestyle="None", markersize=6, label=label, color=color)
            elif len(xs) <= 3:
                ax.plot(xs, ys, marker="o", linestyle="-", linewidth=1.5, markersize=4, label=label, color=color)
            else:
                ax.plot(xs, ys, linestyle="-", linewidth=1.8, label=label, color=color)

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        ax.grid(True, alpha=0.3)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        metric_lower = metric.lower()
        is_accuracy = metric_lower.endswith("_acc") or metric_lower in {"acc", "accuracy"}
        if is_accuracy:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    # x-axis formatting across visible axes
    if all_epochs:
        min_e = min(all_epochs)
        max_e = max(all_epochs)
        if min_e == max_e:
            for ax in axes_flat[:n_metrics]:
                ax.set_xlim(min_e - 0.5, max_e + 0.5)
                ax.set_xticks([min_e])
        else:
            span = max_e - min_e
            xticks = list(range(min_e, max_e + 1)) if span <= 25 else None
            for ax in axes_flat[:n_metrics]:
                ax.set_xlim(min_e, max_e)
                if xticks is not None:
                    ax.set_xticks(xticks)
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in axes_flat[max(0, n_metrics - ncols): n_metrics]:
        ax.set_xlabel("epoch")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        ncol = min(3, max(1, len(labels)))
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=True)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _plot_pareto(summaries: list[dict[str, Any]], plot_path: Path, *, label_key: str) -> None:
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for s in summaries:
        xs.append(float(s.get("param_count", 0)))
        ys.append(float(s.get("best_test_acc", 0.0)))
        labels.append(
            str(s.get(label_key) or s.get("run_id") or Path(s.get("run_dir", "")).name or "run")
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(xs, ys, s=60)
    for x, y, label in zip(xs, ys, labels, strict=False):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel("best test accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MNIST runs (table + optional plot).")
    p.add_argument("--runs", nargs="+", required=True, help="Run directories under runs/")
    p.add_argument("--plot", type=str, default=None, help="Write plot PNG path (optional).")
    p.add_argument(
        "--metrics",
        type=str,
        default="test_acc,test_loss,train_loss,lr",
        help="Comma-separated metrics to plot from metrics.csv.",
    )
    p.add_argument(
        "--pareto",
        type=str,
        default=None,
        help="Write parameter-vs-accuracy scatter PNG path (optional).",
    )
    p.add_argument(
        "--time-plot",
        type=str,
        default=None,
        help="Write test_acc-vs-time plot PNG path (optional).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.runs, runs_root=Path("runs"))
    plot_path = Path(args.plot) if args.plot else None
    pareto_path = Path(args.pareto) if args.pareto else None
    time_plot_path = Path(args.time_plot) if args.time_plot else None
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    table = compare_runs(run_dirs, metrics=metrics, plot_path=plot_path, pareto_path=pareto_path)
    print(table)
    if plot_path:
        print(f"Wrote plot: {plot_path}")
    if pareto_path:
        print(f"Wrote pareto: {pareto_path}")
    if time_plot_path:
        plot_time_to_acc(run_dirs, time_plot_path)
        print(f"Wrote time plot: {time_plot_path}")


def resolve_run_dirs(args: list[str], *, runs_root: Path = Path("runs")) -> list[Path]:
    out: list[Path] = []
    for raw in args:
        p = Path(raw)
        if not p.exists():
            p = runs_root / raw
        if not p.exists():
            raise FileNotFoundError(f"Run path not found: {raw}")

        if p.is_dir() and (p / "summary.json").exists():
            out.append(p)
            continue

        if p.is_dir():
            children = sorted(
                c
                for c in p.iterdir()
                if c.is_dir() and not c.is_symlink() and (c / "summary.json").exists()
            )
            if not children:
                raise FileNotFoundError(f"No runs found under directory: {p}")
            out.extend(children)
            continue

        raise FileNotFoundError(f"Not a directory: {p}")

    # stable order + de-dup
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


if __name__ == "__main__":
    main()

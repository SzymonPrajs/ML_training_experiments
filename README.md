# Minimal MNIST (PyTorch)

Small MNIST training pipeline geared toward **few trainable parameters** and easy **manual experiment comparison**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Train (downloads MNIST automatically)

Run one experiment at a time (each creates a new folder in `runs/`):

```bash
python -m mnist.train --config configs/cnn_4_8_16_sgd_cosine_30e.yaml --run-name cnn_4_8_16_30e
python -m mnist.train --config configs/tiny_sgd_cosine.yaml --run-name tiny_sgd_cosine
python -m mnist.train --config configs/tiny_adamw.yaml --run-name tiny_adamw
python -m mnist.train --config configs/tiny_rmsprop.yaml --run-name tiny_rmsprop
```

Recommended minimal model: `configs/cnn_4_8_16_sgd_cosine_30e.yaml` (~4k trainable params) reaches ~99%+ test accuracy (30 epochs).

The `tiny_*.yaml` configs use a slightly larger CNN (~24k trainable params) and should reach ~99.3%+ test accuracy in ~20 epochs on MNIST.

By default, the trainer uses `mps` if available (Apple Silicon), otherwise `cuda`, otherwise `cpu`.

Quick overrides for ad-hoc experiments:

```bash
python -m mnist.train --config configs/tiny_sgd_cosine.yaml --run-name lr_test --set optimizer.lr=0.05
```

## Run All Configs

Run every `*.yaml` in `configs/` sequentially and print a final summary table:

```bash
python -m mnist.run_all
```

To run the exploratory configs in `configs/search/`:

```bash
python -m mnist.run_all --configs-dir configs/search
```

This creates a new timestamped batch directory under `runs/` and writes:

- `reports/compare.png` (curves for `test_acc`, `test_loss`, `train_loss`, `lr`)
- `reports/pareto.png` (params vs best accuracy)
- `reports/time_to_acc.png` (`test_acc` vs cumulative seconds)
- `reports/summary.md` (table)
- `reports/runs.csv` (per-run summaries)
- `reports/batch.json` (batch metadata)

List batches and runs:

```bash
python -m mnist.list_runs
python -m mnist.list_runs --batch runs/latest
```

## Compare runs (tables + plots)

```bash
python -m mnist.compare --runs runs/latest --plot reports/compare.png --pareto reports/pareto.png
```

This prints a markdown table to stdout and writes an optional plot.

## What gets saved per run

- `runs/<run>/config.yaml` (resolved config)
- `runs/<run>/metrics.csv` (epoch metrics)
- `runs/<run>/summary.json` (best/final metrics + param count)
- `runs/<run>/best.pt` (best model checkpoint by test accuracy)

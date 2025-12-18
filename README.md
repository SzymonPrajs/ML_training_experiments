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
python -m mnist.train --config configs/cnn4k_sgd_cosine.yaml --run-name cnn4k_sgd
python -m mnist.train --config configs/cnn4k_rmsprop_cosine.yaml --run-name cnn4k_rmsprop
python -m mnist.train --config configs/cnn4k_adamw_cosine.yaml --run-name cnn4k_adamw
```

These configs all use the same tiny CNN (~4k trainable params) and reach ~99% test accuracy in ~20 epochs.

By default, the trainer uses `mps` if available (Apple Silicon), otherwise `cuda`, otherwise `cpu`.

Quick overrides for ad-hoc experiments:

```bash
python -m mnist.train --config configs/cnn4k_sgd_cosine.yaml --run-name lr_test --set optimizer.lr=0.05
```

## Run All Configs

Run every `*.yaml` in `configs/` sequentially and print a final summary table:

```bash
python -m mnist.run_all
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

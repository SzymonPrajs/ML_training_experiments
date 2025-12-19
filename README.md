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
python -m mnist.train --config configs/cnn_sgd_cosine.yaml --run-name cnn_sgd
python -m mnist.train --config configs/cnn_rmsprop_cosine.yaml --run-name cnn_rmsprop
python -m mnist.train --config configs/cnn_adamw_cosine.yaml --run-name cnn_adamw
```

These configs all use the same tiny CNN (~4k trainable params) and target ~99% test accuracy in ~10 epochs.

By default, the trainer uses `mps` if available (Apple Silicon), otherwise `cuda`, otherwise `cpu`.

Quick overrides for ad-hoc experiments:

```bash
python -m mnist.train --config configs/cnn_sgd_cosine.yaml --run-name lr_test --set optimizer.lr=0.05
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
- `reports/viz/<run_name>/...` (conv kernels + single-image activations per run)
- `reports/summary.md` (table)
- `reports/runs.csv` (per-run summaries)
- `reports/batch.json` (batch metadata)
- `reports/time_to_acc_compute.png` (test_acc vs compute-only seconds)

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

## Forward-Forward comparisons

Forward-forward (FF) variants and an AdamW backprop baseline live under `configs/compare_*.yaml`:

```bash
python -m mnist.run_all --pattern "compare_*.yaml" --batch-name ff_compare
```

The summary table includes `compute_s` and `data_s`, and `reports/time_to_acc_compute.png` plots accuracy vs compute-only time.

## What gets saved per run

- `runs/<run>/config.yaml` (resolved config)
- `runs/<run>/metrics.csv` (epoch metrics)
- `runs/<run>/summary.json` (best/final metrics + param count)
- `runs/<run>/best.pt` (best model checkpoint by test accuracy)

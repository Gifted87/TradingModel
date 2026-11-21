# Gold Market Research Models

An experimental Python research project for building and evaluating machine learning models on XAU/USD (spot gold) price data. The code explores short horizon price-change classification and regression, multi-scale features, Transformer and Conformer models, and offline policy learning.

The project contains two related research tracks: the top-level `src/` pipeline for cause-and-effect experiments, and `Model_Trial/` for a separate, phased model and evaluation workflow. Results from either track are experiments; this repository is not a trading platform or financial advice.

## Research workflow

The top-level pipeline uses minute bars as model inputs and predicts a five-minute price change. Configuration files define the symbol, lookback, data paths, and label construction. For example, the default `configs/gold.yaml` uses XAU/USD and a 60-minute cause window. The 45-minute discrete and continuous configurations are in `configs/gold_discrete_45m.yaml` and `configs/gold_continuous_45m.yaml`.

1. Download or provide historical Dukascopy data.
2. Build a time-ordered cause/effect dataset.
3. Train one of the baseline, Transformer, hybrid, or multi-scale models.
4. Evaluate predictions with the corresponding backtest or audit script.

Inputs should come from a consistent data feed. Spot gold and gold futures are different instruments and should not be mixed in one experiment.

## Requirements

- Python 3.10 or newer (Python 3.12 is used for development)
- Packages listed in [`requirements.txt`](requirements.txt)
- PyTorch for neural network training and inference; install a build appropriate for your CPU or CUDA environment using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)

Create an environment and install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch separately if you plan to run the neural network scripts.

## Quick start

Download one-minute bars beginning on the configured start date:

```bash
python src/download_dukascopy.py --config configs/gold.yaml --timeframe 1m --start 2010-01-01
```

Build the cause/effect dataset:

```bash
python src/build_cause_effect_dataset.py --config configs/gold.yaml
```

The dataset builder writes processed data beneath the configured data directory (by default, `data/processed/`). It aligns the cause window before the target interval, derives five-minute effects from the same minute series, and assigns labels from configured return quantiles.

The repository includes several model entry points. Examples include:

```bash
python src/train_baseline.py --config configs/gold.yaml
python src/train_advanced.py --config configs/gold.yaml
python src/train_multiscale_conformer.py --config configs/gold_continuous_45m.yaml
```

Training scripts may require substantial memory, compute time, and local data. Read each script's `--help` output and the selected YAML configuration before starting a long run.

## Evaluation

Backtest scripts accept an assumed round-trip cost in gold price units. For example:

```bash
python src/backtest_transformer.py --config configs/gold.yaml --cost 0.50
python src/backtest_discrete_transformer.py --config configs/gold_discrete_45m.yaml --cost 0.50
```

These scripts provide research diagnostics, not evidence of achievable live returns. Results depend on the data period, feed, execution assumptions, spread, slippage, and market regime. Validate using chronological holdouts, walk-forward analysis, realistic costs, and an independent data source before drawing conclusions. Do not use experimental model output to make financial decisions without independent professional review.

## Repository layout

```text
configs/       Data and experiment configuration
src/           Dataset builders, feature code, models, training, and evaluation
tests/         Unit and semantic checks for the top-level pipeline
Model_Trial/   Separate phased experiments, model code, and evaluation scripts
```

Generated datasets and model checkpoints are expected under the local data directory and are not required to understand the source code. The `Model_Trial/` workflow has its own scripts and dependencies; consult files in that directory before running it.

## Contributing

Keep experiments reproducible: record the configuration, data source and date range, model checkpoint, and evaluation assumptions alongside results. Avoid committing private credentials, locally downloaded market data, or machine-specific artifacts.

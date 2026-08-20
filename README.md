# Gold Price-Change Model

This project builds a leakage-safe dataset for predicting the next 5-minute gold close change from prior 1-minute close changes.

## Quick start

```powershell
python -m pip install -r requirements.txt
python src/download_gold.py --config configs/gold.yaml
python src/build_dataset.py --config configs/gold.yaml
python src/train_baseline.py --config configs/gold.yaml
python src/backtest_transformer.py --config configs/gold.yaml --cost 0.50
```

The default instrument is `GC=F` (gold futures). Change `symbol` in `configs/gold.yaml` if a different gold feed is required.

The target is a classification range, not an iterative search. Each historical 5-minute effect is assigned to exactly one bounded range, and the model will later output probabilities for all ranges. Generated raw data, processed data, and models are stored under `data/`.

Yahoo Finance limits how far back 1-minute bars can be requested. The downloader requests the maximum available history in backward chunks and reports the actual range returned.

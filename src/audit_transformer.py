"""Stress-test the out-of-sample Transformer against causal and placebo checks."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from common import data_path, load_config
from train_advanced import GoldTransformer


def direction_score(signal: np.ndarray, actual: np.ndarray) -> dict:
    active = signal != 0
    if not active.any():
        return {"coverage": 0.0, "accuracy": 0.0, "examples": 0}
    return {"coverage": float(active.mean()), "accuracy": float((signal[active] * actual[active] > 0).mean()), "examples": int(active.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = pd.read_parquet(data_path(config, "processed", config["model_dataset"])).sort_values("target_time").reset_index(drop=True)
    columns = sorted(c for c in data.columns if c.startswith("change_"))
    split = int(len(data) * 0.85)
    train, test = data.iloc[:split], data.iloc[split:].reset_index(drop=True)
    preprocessing = np.load(data_path(config, "models", "gold_transformer_oos_preprocessing.npz"))
    checkpoint = torch.load(data_path(config, "models", "gold_transformer_oos.pt"), map_location="cpu", weights_only=True)
    model = GoldTransformer(checkpoint["sequence_length"], checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    x = (test[columns].to_numpy(dtype=np.float32) - preprocessing["mean"]) / preprocessing["std"]
    with torch.no_grad():
        logits, _ = model(torch.tensor(x, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()
    predicted = probabilities.argmax(axis=1)
    actual_labels = test["label"].to_numpy()
    actual = test["effect"].to_numpy(dtype=float)
    expected = probabilities @ np.array([train[train.label == i].effect.mean() for i in range(checkpoint["classes"])])

    result = {
        "test_examples": len(test),
        "overall_class_accuracy": float((predicted == actual_labels).mean()),
        "top_3_accuracy": float(np.mean([actual_labels[i] in np.argsort(probabilities[i])[-3:] for i in range(len(test))])),
        "direction_by_threshold": {str(t): direction_score(np.where(expected > t, 1, np.where(expected < -t, -1, 0)), actual) for t in (0.0, 0.25, 0.5, 1.0, 1.5)},
    }

    momentum = test[columns].sum(axis=1).to_numpy()
    result["simple_cause_sum_baseline"] = direction_score(np.sign(momentum), actual)

    rng = np.random.default_rng(42)
    permuted_x = x[rng.permutation(len(x))]
    with torch.no_grad():
        shuffled_logits, _ = model(torch.tensor(permuted_x, dtype=torch.float32))
    shuffled_probabilities = torch.softmax(shuffled_logits, dim=1).numpy()
    shuffled_expected = shuffled_probabilities @ np.array([train[train.label == i].effect.mean() for i in range(checkpoint["classes"])])
    result["shuffled_input_placebo"] = direction_score(np.sign(shuffled_expected), actual)

    for shift in (-1, 1):
        shifted = np.roll(actual, shift)
        result[f"target_shift_{shift}_bars"] = direction_score(np.sign(expected), shifted)

    yearly = {}
    years = pd.to_datetime(test["target_time"]).dt.year.to_numpy()
    for year in sorted(np.unique(years)):
        mask = years == year
        yearly[str(year)] = {"examples": int(mask.sum()), "class_accuracy": float((predicted[mask] == actual_labels[mask]).mean()), "direction": direction_score(np.sign(expected[mask]), actual[mask])}
    result["by_year"] = yearly
    output = data_path(config, "reports")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "gold_transformer_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

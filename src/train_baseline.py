"""Train a chronological baseline classifier and report useful diagnostics."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from common import data_path, load_config


def evaluate_top_k(model, x_test: pd.DataFrame, y_test: pd.Series, k: int) -> float:
    probabilities = model.predict_proba(x_test)
    classes = model.classes_
    top = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([actual in classes[row] for actual, row in zip(y_test, top)]))


def add_summary_features(data: pd.DataFrame, raw_columns: list[str]) -> pd.DataFrame:
    features = data[raw_columns].copy()
    values = features.to_numpy(dtype=float)
    for minutes in (5, 15, 30, 60):
        window = values[:, -minutes:]
        features[f"mean_{minutes}m"] = window.mean(axis=1)
        features[f"std_{minutes}m"] = window.std(axis=1)
        features[f"sum_{minutes}m"] = window.sum(axis=1)
        features[f"up_fraction_{minutes}m"] = (window > 0).mean(axis=1)
        features[f"range_{minutes}m"] = window.max(axis=1) - window.min(axis=1)
    features["total_up"] = (values > 0).sum(axis=1)
    features["total_down"] = (values < 0).sum(axis=1)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gold.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    path = data_path(config, "processed", "gold_5m_classification.parquet")
    data = pd.read_parquet(path).sort_values("target_time")
    raw_columns = [column for column in data.columns if column.startswith("change_")]
    features = add_summary_features(data, raw_columns)
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    x_train, x_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = train["label"], test["label"]

    linear_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    tree_model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    models = {"logistic": linear_model, "hist_gradient_boosting": tree_model}
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        known_test = y_test.isin(model.classes_)
        known_probabilities = probabilities[known_test.to_numpy()]
        known_targets = y_test[known_test]
        metrics[name] = {
        "train_examples": len(train),
        "test_examples": len(test),
        "feature_count": x_train.shape[1],
        "train_classes": int(y_train.nunique()),
        "test_classes": int(y_test.nunique()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "top_3_accuracy": evaluate_top_k(model, x_test, y_test, 3),
        "known_test_examples": int(known_test.sum()),
        "log_loss_known_test_classes": float(
            log_loss(known_targets, known_probabilities, labels=model.classes_)
        ) if known_test.any() else None,
        }
    output = data_path(config, "models")
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output / f"gold_{name}.joblib")
    with (output / "gold_baseline_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print("Most common training labels:")
    print(y_train.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

"""
model.py
Trains a logistic regression propensity model and evaluates it.

Primary evaluation metric: top-decile lift — the concentration of actual
converters in the highest-scoring decile. A model can have high AUC but
spread buyers across multiple deciles; for campaign budgeting, buyers need
to be stacked in decile 1.

Secondary metrics: AUC, precision/recall, classification report.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from preprocess import fit_thresholds, apply_thresholds, summarize_thresholds

FEATURE_COLS = [
    "base_propensity_score",
    "purchased_same_period_1yr_ago",
    "purchased_same_period_2yr_ago",
    "purchased_month_minus_1",
    "purchased_month_minus_2",
    "purchased_month_minus_3",
]
TARGET_COL = "purchased_in_campaign"

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "customers.csv")
)
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "propensity_model_v1.joblib")
)


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Measures concentration of actual converters in the top scoring decile.

    Returns a dict with:
    - decile_1_converter_share: % of all converters captured in decile 1
    - decile_1_conversion_rate: conversion rate within decile 1
    - population_conversion_rate: baseline conversion rate
    - lift_multiplier: decile 1 rate / baseline rate
    """
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df["y_score"], q=10, labels=range(10, 0, -1))

    total_converters = df["y_true"].sum()
    decile_1 = df[df["decile"] == 1]  # label 1 = highest scores (top decile)
    decile_1_converters = decile_1["y_true"].sum()
    decile_1_rate = decile_1_converters / len(decile_1)
    pop_rate = total_converters / len(df)

    return {
        "decile_1_converter_share": decile_1_converters / total_converters,
        "decile_1_conversion_rate": decile_1_rate,
        "population_conversion_rate": pop_rate,
        "lift_multiplier": decile_1_rate / pop_rate if pop_rate > 0 else 0,
        "decile_table": (
            df.groupby("decile", observed=True)
            .agg(
                n_customers=("y_true", "count"),
                n_converters=("y_true", "sum"),
            )
            .assign(conversion_rate=lambda d: d["n_converters"] / d["n_customers"])
            .sort_index(ascending=False)
        ),
    }


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    print("Loading data...")
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Conversion rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}\n")

    # ── Outlier handling: winsorize at 2nd/98th percentile ───────────────────
    # Thresholds are fit on X_train only to prevent data leakage.
    # The same thresholds are saved and reapplied at score time.
    thresholds = fit_thresholds(X_train, FEATURE_COLS)
    summarize_thresholds(thresholds)
    X_train = apply_thresholds(X_train, thresholds)
    X_test = apply_thresholds(X_test, thresholds)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ── Primary metric: top-decile lift ──────────────────────────────────────
    lift_stats = top_decile_lift(y_test.values, y_prob)
    print("=" * 60)
    print("PRIMARY METRIC: Top-Decile Lift")
    print("=" * 60)
    print(f"  Decile 1 converter share : {lift_stats['decile_1_converter_share']:.1%}")
    print(f"  Decile 1 conversion rate : {lift_stats['decile_1_conversion_rate']:.1%}")
    print(f"  Population baseline rate : {lift_stats['population_conversion_rate']:.1%}")
    print(f"  Lift multiplier          : {lift_stats['lift_multiplier']:.2f}x")
    print()
    print("Decile breakdown (test set):")
    print(lift_stats["decile_table"].to_string())
    print()

    # ── Secondary metrics ────────────────────────────────────────────────────
    auc = roc_auc_score(y_test, y_prob)
    print("=" * 60)
    print("SECONDARY METRICS: Overall Model Accuracy")
    print("=" * 60)
    print(f"  AUC-ROC: {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"], zero_division=0))

    # ── Feature coefficients ─────────────────────────────────────────────────
    coef_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": model.coef_[0]}
    ).sort_values("coefficient", ascending=False)
    print("Feature coefficients:")
    print(coef_df.to_string(index=False))
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "thresholds": thresholds}, model_path)
    print(f"Model saved -> {model_path}")

    return model, lift_stats


if __name__ == "__main__":
    train()

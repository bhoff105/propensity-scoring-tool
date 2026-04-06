"""
score.py
Loads a trained propensity model, scores a customer CSV, and produces
a ranked output with deciles, subdeciles, and score contribution breakdowns.

Output columns:
  customer_id                   — original ID
  predicted_score               — model probability (0–1)
  model_decile                  — 1 (highest) to 10 (lowest)
  model_subdecile               — e.g. "1_1" through "1_10"
  base_score_contribution_pct   — % of score explained by base propensity
  recent_behavior_contribution_pct — % from same-period purchase history
  momentum_contribution_pct     — % from recent 3-month activity

The contribution breakdown tells marketers *why* a customer scored high —
a customer driven by recent momentum tells a different story than one
whose score is purely historical propensity.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import apply_thresholds

MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "propensity_model_v1.joblib")
)

FEATURE_COLS = [
    "base_propensity_score",
    "purchased_same_period_1yr_ago",
    "purchased_same_period_2yr_ago",
    "purchased_month_minus_1",
    "purchased_month_minus_2",
    "purchased_month_minus_3",
]

# Feature group -> contribution buckets (mirrors real model's 2YA%/SER%/REM%)
BASE_FEATURES = ["base_propensity_score"]
RECENCY_FEATURES = ["purchased_same_period_1yr_ago", "purchased_same_period_2yr_ago"]
MOMENTUM_FEATURES = ["purchased_month_minus_1", "purchased_month_minus_2", "purchased_month_minus_3"]


def _subdecile_labels(decile_num: int) -> list[str]:
    return [f"{decile_num}_{i}" for i in range(1, 11)]


def score(input_df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Score a DataFrame of customers.

    Parameters
    ----------
    input_df : pd.DataFrame
        Must contain all columns in FEATURE_COLS plus 'customer_id'.
    model_path : str
        Path to the saved model artifact.

    Returns
    -------
    pd.DataFrame
        Scored output with decile, subdecile, and contribution columns.
    """
    artifact = joblib.load(model_path)
    model = artifact["model"]
    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    feature_coef_map = dict(zip(FEATURE_COLS, coefs))

    # Apply training-time winsorization thresholds before scoring.
    # This ensures score-time data is treated identically to training data.
    thresholds = artifact.get("thresholds")
    if thresholds:
        input_df = apply_thresholds(input_df, thresholds)

    X = input_df[FEATURE_COLS].values
    linear_scores = X @ coefs + intercept

    # Predicted probability
    predicted_score = 1 / (1 + np.exp(-linear_scores))

    # Contribution of each feature group to the linear score
    def group_contribution(features):
        return np.array(
            [feature_coef_map[f] * input_df[f].values for f in features]
        ).sum(axis=0)

    base_contrib = group_contribution(BASE_FEATURES)
    recency_contrib = group_contribution(RECENCY_FEATURES)
    momentum_contrib = group_contribution(MOMENTUM_FEATURES)

    # Normalize contributions (use abs value so negatives don't produce >100%)
    total_abs = np.abs(base_contrib) + np.abs(recency_contrib) + np.abs(momentum_contrib)
    total_abs = np.where(total_abs == 0, 1, total_abs)  # avoid div/0

    base_pct = np.abs(base_contrib) / total_abs
    recency_pct = np.abs(recency_contrib) / total_abs
    momentum_pct = np.abs(momentum_contrib) / total_abs

    scored = input_df[["customer_id"]].copy()
    scored["predicted_score"] = predicted_score.round(6)

    # Decile 1 = highest scores (label assignment: lowest score bin -> 10, highest -> 1)
    scored = scored.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    scored["model_decile"] = pd.qcut(
        scored["predicted_score"], q=10, labels=range(10, 0, -1)
    ).astype(int)

    # Subdecile within each decile (1_1 = top of top)
    subdecile_parts = []
    for d in range(1, 11):
        mask = scored["model_decile"] == d
        subset = scored[mask].copy()
        if len(subset) >= 10:
            _, bins = pd.qcut(
                subset["predicted_score"], q=10, duplicates="drop", retbins=True
            )
            n_bins = len(bins) - 1
            labels = list(reversed(_subdecile_labels(d)[:n_bins]))
            subset["model_subdecile"] = pd.qcut(
                subset["predicted_score"],
                q=10,
                labels=labels,
                duplicates="drop",
            ).astype(str)
        else:
            subset["model_subdecile"] = f"{d}_1"
        subdecile_parts.append(subset)

    scored = pd.concat(subdecile_parts).sort_values(
        ["model_decile", "predicted_score"], ascending=[True, False]
    )

    # Reattach contribution columns (aligned by customer_id after sort)
    contrib_df = pd.DataFrame(
        {
            "customer_id": input_df["customer_id"].values,
            "base_score_contribution_pct": (base_pct * 100).round(1),
            "recent_behavior_contribution_pct": (recency_pct * 100).round(1),
            "momentum_contribution_pct": (momentum_pct * 100).round(1),
        }
    )
    scored = scored.merge(contrib_df, on="customer_id", how="left")

    # Bring back original feature columns for downstream use
    feature_passthrough = input_df[["customer_id"] + FEATURE_COLS]
    scored = scored.merge(feature_passthrough, on="customer_id", how="left")

    return scored.reset_index(drop=True)


if __name__ == "__main__":
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "customers.csv")
    )
    output_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "scored_output.csv")
    )

    df = pd.read_csv(data_path)
    result = score(df)
    result.to_csv(output_path, index=False)

    print(f"Scored {len(result):,} customers -> {output_path}")
    print("\nDecile distribution:")
    decile_summary = (
        result.groupby("model_decile")
        .agg(
            n_customers=("customer_id", "count"),
            avg_score=("predicted_score", "mean"),
            avg_base_contrib=("base_score_contribution_pct", "mean"),
            avg_recency_contrib=("recent_behavior_contribution_pct", "mean"),
            avg_momentum_contrib=("momentum_contribution_pct", "mean"),
        )
        .round(3)
    )
    print(decile_summary.to_string())

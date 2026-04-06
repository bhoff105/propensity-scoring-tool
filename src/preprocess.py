"""
preprocess.py
Outlier handling via winsorization (2nd/98th percentile capping).

Thresholds are fit on training data only and saved with the model artifact
so the same caps are applied consistently at score time — preventing data
leakage and ensuring production scoring matches training conditions.

Why winsorize here:
  Logistic regression coefficients are sensitive to extreme values. A customer
  with 500 purchases in a population where the 98th percentile is 12 can pull
  the coefficient for that feature significantly. Capping preserves relative
  ordering while preventing outliers from distorting the fit.

  Tree-based models (XGBoost, RF) are rank-based and don't require this step.
  For LR on behavioral features, it's still standard practice.
"""

import numpy as np
import pandas as pd

LOWER_PERCENTILE = 2
UPPER_PERCENTILE = 98


def fit_thresholds(X: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Compute per-feature winsorization bounds from training data.

    Parameters
    ----------
    X : pd.DataFrame
        Training features (X_train only — never fit on test or score-time data).
    feature_cols : list[str]
        Features to compute bounds for.

    Returns
    -------
    dict : {feature_name: (lower_bound, upper_bound)}
    """
    thresholds = {}
    for col in feature_cols:
        lower = float(np.percentile(X[col], LOWER_PERCENTILE))
        upper = float(np.percentile(X[col], UPPER_PERCENTILE))
        thresholds[col] = (lower, upper)
    return thresholds


def apply_thresholds(X: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Apply saved winsorization thresholds to a feature DataFrame.

    Values below the lower bound are raised to the lower bound.
    Values above the upper bound are capped at the upper bound.
    Binary features (0/1) are unaffected — their bounds will be (0, 1).

    Parameters
    ----------
    X : pd.DataFrame
        Input features (training, test, or new data to score).
    thresholds : dict
        Output of fit_thresholds().

    Returns
    -------
    pd.DataFrame : Copy of X with clipped values.
    """
    X = X.copy()
    for col, (lower, upper) in thresholds.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lower, upper=upper)
    return X


def summarize_thresholds(thresholds: dict) -> None:
    """Print a human-readable summary of winsorization bounds."""
    print("Winsorization thresholds (2nd / 98th percentile, fit on training data):")
    print(f"  {'Feature':<35} {'Lower':>8} {'Upper':>8}  Note")
    print(f"  {'-'*35} {'-'*8} {'-'*8}  ----")
    for col, (lower, upper) in thresholds.items():
        note = "binary — no-op" if lower == 0.0 and upper == 1.0 else ""
        print(f"  {col:<35} {lower:>8.4f} {upper:>8.4f}  {note}")
    print()

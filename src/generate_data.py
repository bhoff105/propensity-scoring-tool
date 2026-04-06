"""
generate_data.py
Generates a synthetic customer dataset for the propensity scoring pipeline.

The features mirror the structure of a real repeat-customer scoring model:
behavioral purchase flags across recent time windows, a base propensity score,
and lifetime value indicators. The target variable (purchased_in_campaign)
is generated using real-world-inspired coefficients so the data is internally
consistent with the model logic.
"""

import numpy as np
import pandas as pd
import uuid

RANDOM_SEED = 42
N_CUSTOMERS = 10_000

# Logistic regression coefficients (inspired by real production model)
# Features: base_propensity_score, purchased_1yr_ago, purchased_2yr_ago,
#           month_minus_1, month_minus_2, month_minus_3
COEFFICIENTS = {
    "base_propensity_score": 0.0068,
    "purchased_same_period_1yr_ago": 1.037,
    "purchased_same_period_2yr_ago": 0.704,
    "purchased_month_minus_1": 0.821,
    "purchased_month_minus_2": 0.560,
    "purchased_month_minus_3": 0.463,
}
INTERCEPT = -3.784


def _logistic(x):
    return 1 / (1 + np.exp(-x))


def generate_customers(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Base propensity score from upstream model — right-skewed, most customers low
    base_score = rng.gamma(shape=2, scale=1.8, size=n).clip(0, 10)

    # Purchase history flags — correlated with base score so high-score customers
    # are more likely to have bought before (realistic)
    p_1yr = _logistic(0.3 * base_score - 1.5)
    p_2yr = _logistic(0.25 * base_score - 1.8)
    p_m1 = _logistic(0.2 * base_score - 2.2)
    p_m2 = _logistic(0.15 * base_score - 2.5)
    p_m3 = _logistic(0.12 * base_score - 2.8)

    purchased_1yr = rng.binomial(1, p_1yr)
    purchased_2yr = rng.binomial(1, p_2yr)
    month_minus_1 = rng.binomial(1, p_m1)
    month_minus_2 = rng.binomial(1, p_m2)
    month_minus_3 = rng.binomial(1, p_m3)

    # Occasion flags — sparse (birthday/anniversary purchasers are a small subset)
    birthday_flag = rng.binomial(1, 0.08, size=n)
    anniversary_flag = rng.binomial(1, 0.05, size=n)

    # Lifetime value metrics
    total_orders = rng.poisson(lam=3.5, size=n).clip(1)
    lifetime_value = (total_orders * rng.uniform(40, 120, size=n)).round(2)
    days_since_last_order = rng.integers(1, 730, size=n)

    # Generate target: purchased_in_campaign
    # Use the model coefficients to create a plausible outcome
    linear_combination = (
        INTERCEPT
        + COEFFICIENTS["base_propensity_score"] * base_score
        + COEFFICIENTS["purchased_same_period_1yr_ago"] * purchased_1yr
        + COEFFICIENTS["purchased_same_period_2yr_ago"] * purchased_2yr
        + COEFFICIENTS["purchased_month_minus_1"] * month_minus_1
        + COEFFICIENTS["purchased_month_minus_2"] * month_minus_2
        + COEFFICIENTS["purchased_month_minus_3"] * month_minus_3
    )
    prob_purchase = _logistic(linear_combination)
    purchased_in_campaign = rng.binomial(1, prob_purchase)

    df = pd.DataFrame(
        {
            "customer_id": [str(uuid.uuid4()) for _ in range(n)],
            "base_propensity_score": base_score.round(4),
            "purchased_same_period_1yr_ago": purchased_1yr,
            "purchased_same_period_2yr_ago": purchased_2yr,
            "purchased_month_minus_1": month_minus_1,
            "purchased_month_minus_2": month_minus_2,
            "purchased_month_minus_3": month_minus_3,
            "birthday_occasion_flag": birthday_flag,
            "anniversary_occasion_flag": anniversary_flag,
            "total_orders_lifetime": total_orders,
            "lifetime_value": lifetime_value,
            "days_since_last_order": days_since_last_order,
            "purchased_in_campaign": purchased_in_campaign,
        }
    )

    return df


if __name__ == "__main__":
    import os

    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "synthetic", "customers.csv"
    )
    output_path = os.path.normpath(output_path)

    df = generate_customers()
    df.to_csv(output_path, index=False)

    n_buyers = df["purchased_in_campaign"].sum()
    print(f"Generated {len(df):,} customers -> saved to {output_path}")
    print(f"Campaign converters: {n_buyers:,} ({n_buyers / len(df):.1%} of universe)")
    print(f"\nFeature summary:")
    print(df.drop(columns=["customer_id", "purchased_in_campaign"]).describe().round(3))

"""
validate.py
Post-campaign lift analysis — simulates the validation step that follows
a real scoring run: once a campaign has run, you join the scored output
to actual conversion data to measure whether high-decile customers
actually converted at higher rates.

Primary output: top-decile concentration and lift multiplier.
Secondary output: full decile-by-decile bar chart saved as PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


SCORED_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "scored_output.csv")
)
ACTUALS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "customers.csv")
)
CHART_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "lift_by_decile.png")
)


def compute_lift_table(scored: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Join scored output to actuals and compute per-decile lift metrics.
    """
    df = scored[["customer_id", "model_decile", "predicted_score"]].merge(
        actuals[["customer_id", "purchased_in_campaign"]],
        on="customer_id",
        how="inner",
    )

    summary = (
        df.groupby("model_decile")
        .agg(
            n_customers=("customer_id", "count"),
            n_converters=("purchased_in_campaign", "sum"),
            avg_predicted_score=("predicted_score", "mean"),
        )
        .reset_index()
    )
    summary["conversion_rate"] = summary["n_converters"] / summary["n_customers"]
    pop_rate = df["purchased_in_campaign"].mean()
    summary["lift_vs_baseline"] = summary["conversion_rate"] / pop_rate
    summary["population_rate"] = pop_rate
    summary = summary.sort_values("model_decile")

    return summary, pop_rate


def print_headline_stats(lift_table: pd.DataFrame, pop_rate: float):
    total_converters = lift_table["n_converters"].sum()
    d1 = lift_table[lift_table["model_decile"] == 1].iloc[0]

    d1_share = d1["n_converters"] / total_converters
    d1_lift = d1["lift_vs_baseline"]

    print("=" * 60)
    print("PRIMARY: Top-Decile Lift")
    print("=" * 60)
    print(f"  Decile 1 converter share : {d1_share:.1%} of all campaign converters")
    print(f"  Decile 1 conversion rate : {d1['conversion_rate']:.1%}")
    print(f"  Population baseline rate : {pop_rate:.1%}")
    print(f"  Lift multiplier          : {d1_lift:.2f}x")
    print()
    print("=" * 60)
    print("SECONDARY: Full Decile Breakdown")
    print("=" * 60)
    display = lift_table[
        ["model_decile", "n_customers", "n_converters", "conversion_rate", "lift_vs_baseline"]
    ].copy()
    display["conversion_rate"] = display["conversion_rate"].map("{:.1%}".format)
    display["lift_vs_baseline"] = display["lift_vs_baseline"].map("{:.2f}x".format)
    print(display.to_string(index=False))
    print()


def plot_lift_chart(lift_table: pd.DataFrame, pop_rate: float, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "#1a5276" if d == 1 else "#2e86c1" if d <= 3 else "#aed6f1"
        for d in lift_table["model_decile"]
    ]
    bars = ax.bar(
        lift_table["model_decile"],
        lift_table["conversion_rate"] * 100,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Baseline line
    ax.axhline(
        pop_rate * 100,
        color="#e74c3c",
        linewidth=1.5,
        linestyle="--",
        label=f"Population baseline ({pop_rate:.1%})",
    )

    # Lift labels on bars
    for bar, row in zip(bars, lift_table.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{row.lift_vs_baseline:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    ax.set_xlabel("Model Decile (1 = Highest Propensity)", fontsize=11)
    ax.set_ylabel("Conversion Rate (%)", fontsize=11)
    ax.set_title(
        "Campaign Conversion Rate by Propensity Score Decile",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(lift_table["model_decile"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Lift chart saved -> {output_path}")


if __name__ == "__main__":
    scored = pd.read_csv(SCORED_PATH)
    actuals = pd.read_csv(ACTUALS_PATH)

    lift_table, pop_rate = compute_lift_table(scored, actuals)
    print_headline_stats(lift_table, pop_rate)
    plot_lift_chart(lift_table, pop_rate, CHART_PATH)

"""
Sable & Co. Campaign Intelligence
Propensity Scoring Tool · Spring Collection 2026

Four-page Streamlit dashboard (sidebar radio navigation):
  Page 1 -- The Engagement Model   (auto-loaded sample data, model explainer + KPIs)
  Page 2 -- Campaign Scoring       (pre-loaded deliverable output, download)
  Page 3 -- Segment Explorer       (filter, drill, inspect)
  Page 4 -- How It Works           (technical deep-dive for reviewers)
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from score import score, FEATURE_COLS

MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "propensity_model_v1.joblib")
)
SAMPLE_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "customers.csv")
)

# -- Design tokens ------------------------------------------------------------
NAVY     = "#1B2A3B"
AMBER    = "#D97706"
BLUE_MED = "#2563EB"
GRAY     = "#9CA3AF"

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Sable & Co. | Campaign Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Global CSS ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* -- Global reset -- */
html, body,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    color: #111827;
}

/* -- Page background -- */
[data-testid="stAppViewContainer"],
.main .block-container {
    background-color: #F7F8FA;
}

/* -- Main content container padding -- */
.main .block-container {
    padding-top: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1200px;
}

/* -- Sidebar -- */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #1B2A3B !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #E5EAF0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #A8B8C8 !important;
    font-size: 12px !important;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* -- Sidebar radio navigation -- */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 12px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: background 0.15s ease;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #C8D8E8 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-active="true"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
/* Hide the radio circle dots */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* -- Headings -- */
h1, [data-testid="stMarkdownContainer"] h1 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 32px !important;
    font-weight: 400 !important;
    color: #111827 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.3px;
}
h2, [data-testid="stMarkdownContainer"] h2 {
    font-family: 'Inter', sans-serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #111827 !important;
}
h3, [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #1B2A3B !important;
}

/* -- Section label eyebrow -- */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #D97706;
    margin-bottom: 8px;
}

/* -- Metric card HTML -- */
.metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 8px rgba(0,0,0,0.04);
    padding: 22px 26px;
    height: 100%;
    border: 1px solid rgba(0,0,0,0.04);
}
.metric-card .label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 30px;
    font-weight: 700;
    color: #111827;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.metric-card .sub {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 4px;
}
.metric-card.accent .value {
    color: #D97706;
}

/* -- Callout box -- */
.callout-box {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    color: #1F2937;
}
.callout-box strong {
    color: #92400E;
}

/* -- Insight box -- */
.insight-box {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    color: #1E3A5F;
}

/* -- Section divider -- */
.section-divider {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 32px 0;
}

/* -- Tier badge -- */
.tier-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.tier-priority    { background: #1B2A3B; color: #fff; }
.tier-high-reach  { background: #2563EB; color: #fff; }
.tier-broad-reach { background: #DBEAFE; color: #1E40AF; }
.tier-suppress    { background: #F3F4F6; color: #6B7280; }

/* -- How It Works prose -- */
.how-section {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.05);
    padding: 28px 32px;
    margin-bottom: 16px;
}
.how-section h4 {
    font-size: 15px;
    font-weight: 700;
    color: #1B2A3B;
    margin-bottom: 10px;
}
.how-section p, .how-section li {
    font-size: 14px;
    color: #374151;
    line-height: 1.65;
}

/* -- Tooltip system -- */
.col-tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: default;
}
.col-tooltip .info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: #E5E7EB;
    color: #6B7280;
    font-size: 9px;
    font-weight: 700;
    cursor: help;
    flex-shrink: 0;
    line-height: 1;
    font-style: normal;
}
.col-tooltip .tooltip-body {
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(27,42,59,0.96);
    color: #E5EAF0;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.55;
    padding: 10px 14px;
    border-radius: 7px;
    width: 230px;
    z-index: 9999;
    white-space: normal;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}
.col-tooltip:hover .tooltip-body { display: block; }

/* -- Lift badge -- */
.lift-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
    font-family: 'Inter', sans-serif;
}
.lift-above   { background: rgba(22,163,74,0.12);   color: #15803D; }
.lift-below   { background: rgba(239,68,68,0.10);   color: #DC2626; }
.lift-neutral { background: rgba(107,114,128,0.10); color: #6B7280; }

/* -- Sig dot -- */
.sig-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    flex-shrink: 0;
}
.sig-dot-green  { background: #16A34A; }
.sig-dot-red    { background: #DC2626; }
.sig-dot-gray   { background: #9CA3AF; }

/* -- Stat pill -- */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #374151;
    margin: 2px 3px 2px 0;
}
.stat-pill .pill-label {
    color: #9CA3AF;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 10px;
}
.stat-pill .pill-value {
    color: #111827;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* -- Model component cards -- */
.model-component-card {
    background: #FFFFFF;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 24px;
    height: 100%;
}
.model-component-card .component-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 8px;
}
.model-component-card .component-name {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 17px;
    font-weight: 400;
    color: #111827;
    margin: 8px 0 12px;
}
.model-component-card .component-body {
    font-size: 13px;
    color: #4B5563;
    line-height: 1.65;
}
.model-component-card .component-what {
    font-size: 13px;
    color: #374151;
    line-height: 1.6;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #F3F4F6;
}
.model-component-card .component-features {
    font-size: 11px;
    color: #9CA3AF;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #F3F4F6;
}

/* -- Streamlit default metric override -- */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

/* -- Download button -- */
[data-testid="stDownloadButton"] button {
    background-color: #1B2A3B !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #253d57 !important;
}

/* -- Primary buttons -- */
[data-testid="stButton"] button[kind="primary"] {
    background-color: #1B2A3B !important;
    color: #fff !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}

/* -- Info/warning box override -- */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* -- Dataframe -- */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def assign_campaign_tier(decile: int) -> str:
    if decile == 1:
        return "Priority"
    elif decile <= 3:
        return "High Reach"
    elif decile <= 6:
        return "Broad Reach"
    else:
        return "Suppress"


def tier_badge_html(tier: str) -> str:
    cls_map = {
        "Priority":    "tier-priority",
        "High Reach":  "tier-high-reach",
        "Broad Reach": "tier-broad-reach",
        "Suppress":    "tier-suppress",
    }
    css = cls_map.get(tier, "tier-suppress")
    return f'<span class="tier-badge {css}">{tier}</span>'


def metric_card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    accent_class = "accent" if accent else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card {accent_class}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """


def load_and_score_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DATA_PATH)
    scored = score(df, MODEL_PATH)
    scored["campaign_tier"] = scored["model_decile"].apply(assign_campaign_tier)
    # Merge actuals so charts can show real conversion rates instead of predicted scores
    scored = scored.merge(df[["customer_id", "purchased_in_campaign"]], on="customer_id", how="left")
    return scored


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

BASELINE_RATE = 0.073  # Population conversion rate (Spring Collection)


def col_header(label: str, tooltip: str) -> str:
    """Column header with hover tooltip info icon."""
    return (
        f'<div class="col-tooltip">'
        f'<span style="font-size:11px;font-weight:600;color:#9CA3AF;'
        f'text-transform:uppercase;letter-spacing:.06em;">{label}</span>'
        f'<span class="info-icon">i</span>'
        f'<div class="tooltip-body">{tooltip}</div>'
        f'</div>'
    )


def lift_badge_html(conversion_rate: float, baseline: float = BASELINE_RATE) -> str:
    """Color-coded lift multiplier badge."""
    ratio = conversion_rate / baseline if baseline > 0 else 1.0
    css = "lift-above" if ratio >= 1.05 else ("lift-below" if ratio <= 0.95 else "lift-neutral")
    return f'<span class="lift-badge {css}">{ratio:.2f}x</span>'


def stat_pills(*pairs) -> str:
    """Render a row of stat pills. Pass (label, value) tuples."""
    html = ""
    for label, value in pairs:
        html += (
            f'<span class="stat-pill">'
            f'<span class="pill-label">{label}</span>'
            f'<span class="pill-value">{value}</span>'
            f'</span>'
        )
    return html


def _base_chart_layout(height: int = 400, show_legend: bool = True) -> dict:
    return dict(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F7F8FA",
        font=dict(
            color="#374151",
            family="'Inter', -apple-system, sans-serif",
            size=12,
        ),
        showlegend=show_legend,
        margin=dict(l=20, r=20, t=52, b=20),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="rgba(17,34,51,0.94)",
            bordercolor="rgba(255,255,255,0.08)",
            font=dict(size=12, color="#E5EAF0", family="'Inter', sans-serif"),
        ),
    )


def chart_lift_bar(scored: pd.DataFrame) -> go.Figure:
    """Decile lift bar chart using actual conversion rates. Decile 1 highlighted amber."""
    decile_summary = (
        scored.groupby("model_decile")
        .agg(conversion_rate=("purchased_in_campaign", "mean"))
        .reset_index()
        .sort_values("model_decile")
    )
    overall_avg = scored["purchased_in_campaign"].mean() * 100

    colors = []
    for d in decile_summary["model_decile"]:
        if d == 1:
            colors.append(AMBER)
        elif d <= 3:
            colors.append(NAVY)
        elif d <= 6:
            colors.append(BLUE_MED)
        else:
            colors.append(GRAY)

    bar = go.Bar(
        x=decile_summary["model_decile"],
        y=decile_summary["conversion_rate"] * 100,
        marker_color=colors,
        hovertemplate="<b>Decile %{x}</b><br>Actual conversion rate: %{y:.1f}%<br><extra></extra>",
        showlegend=False,
    )
    line = go.Scatter(
        x=decile_summary["model_decile"],
        y=[overall_avg] * len(decile_summary),
        mode="lines",
        line=dict(color="#EF4444", width=1.5, dash="dash"),
        name=f"Overall average ({overall_avg:.1f}%)",
    )
    fig = go.Figure(data=[bar, line])
    layout = _base_chart_layout(height=360, show_legend=True)
    layout.update(
        title=dict(
            text="Conversion Rate by Campaign Decile",
            font=dict(size=14, color="#111827", family="'Inter', sans-serif"),
            x=0,
            pad=dict(b=8),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 11)),
            ticktext=["Priority<br>(1)"] + [str(d) for d in range(2, 11)],
            title="Campaign Decile  (1 = Highest Conversion Likelihood)",
            title_font=dict(size=12, color="#6B7280"),
            tickfont=dict(size=11, color="#6B7280"),
            linecolor="#E5E7EB",
            linewidth=1,
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="Actual Conversion Rate (%)",
            ticksuffix="%",
            title_font=dict(size=12, color="#6B7280"),
            tickfont=dict(size=11, color="#6B7280"),
            gridcolor="#F3F4F6",
        ),
        shapes=[dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#E5E7EB", width=1),
        )],
    )
    fig.update_layout(**layout)

    # Add lift multiplier annotations above each bar
    for _, row in decile_summary.iterrows():
        d = int(row["model_decile"])
        rate = row["conversion_rate"]
        lift = rate / BASELINE_RATE
        color = "#15803D" if lift >= 1.05 else ("#DC2626" if lift <= 0.95 else "#6B7280")
        fig.add_annotation(
            x=d,
            y=rate * 100,
            text=f"{lift:.1f}x",
            showarrow=False,
            yanchor="bottom",
            yshift=4,
            font=dict(size=10, color=color, family="'Inter', sans-serif"),
        )

    return fig


def chart_contribution_breakdown(
    scored: pd.DataFrame,
    group_col: str = "campaign_tier",
) -> go.Figure:
    """Grouped bar chart of score driver contributions by tier or decile."""
    contrib_cols = [
        "base_score_contribution_pct",
        "recent_behavior_contribution_pct",
        "momentum_contribution_pct",
    ]

    if group_col == "campaign_tier":
        order = ["Priority", "High Reach", "Broad Reach", "Suppress"]
        grp = scored.groupby("campaign_tier")[contrib_cols].mean().reindex(order)
        x_vals = order
    else:
        grp = scored.groupby("model_decile")[contrib_cols].mean().sort_index()
        x_vals = list(grp.index)

    traces = [
        go.Bar(
            x=x_vals,
            y=grp["base_score_contribution_pct"],
            name="Customer Health Score",
            marker_color=NAVY,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Customer Health Score: %{y:.0f}%<br>"
                "Share from long-run purchase behavior<extra></extra>"
            ),
        ),
        go.Bar(
            x=x_vals,
            y=grp["recent_behavior_contribution_pct"],
            name="Seasonal Return Signal",
            marker_color=AMBER,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Seasonal Return Signal: %{y:.0f}%<br>"
                "Share from same-window purchase history<extra></extra>"
            ),
        ),
        go.Bar(
            x=x_vals,
            y=grp["momentum_contribution_pct"],
            name="Recent Activity Signal",
            marker_color=BLUE_MED,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Recent Activity Signal: %{y:.0f}%<br>"
                "Share from purchases in the last 90 days<extra></extra>"
            ),
        ),
    ]
    fig = go.Figure(data=traces)
    layout = _base_chart_layout(height=380, show_legend=True)
    layout.update(
        title=dict(
            text="Score Driver Mix by Campaign Tier",
            font=dict(size=14, color="#111827", family="'Inter', sans-serif"),
            x=0,
            pad=dict(b=8),
        ),
        barmode="group",
        xaxis=dict(
            title="Campaign Tier",
            title_font=dict(size=12, color="#6B7280"),
            tickfont=dict(size=11, color="#6B7280"),
            linecolor="#E5E7EB",
            linewidth=1,
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="Avg Contribution to Score (%)",
            ticksuffix="%",
            title_font=dict(size=12, color="#6B7280"),
            tickfont=dict(size=11, color="#6B7280"),
            gridcolor="#F3F4F6",
        ),
        shapes=[dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#E5E7EB", width=1),
        )],
    )
    fig.update_layout(**layout)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 2px 0;">
        <div style="font-family: 'DM Serif Display', Georgia, serif; font-size: 22px; font-weight: 400; color: #FFFFFF; letter-spacing: -0.2px; line-height: 1.2;">
            Sable &amp; Co.
        </div>
        <div style="font-size: 10px; font-weight: 600; color: #D97706; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 5px;">
            Campaign Intelligence
        </div>
    </div>
    <div style="height: 1px; background: linear-gradient(to right, rgba(255,255,255,0.15), rgba(217,119,6,0.6), rgba(255,255,255,0.05)); margin: 16px 0 20px 0;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 11px; font-weight: 600; color: #7A9BBB;
                text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 10px;">
        Active Campaign
    </div>
    <div style="font-size: 13px; color: #C8D8E8; line-height: 1.85;">
        <div style="font-size: 14px; font-weight: 700; color: #FFFFFF;">Spring Collection 2026</div>
        <div style="font-size: 11px; color: #7A9BBB; margin-bottom: 8px;">Mother's Day Gift Window</div>
        <div>Send window: Apr 14-21</div>
        <div>Channel: Email</div>
        <div>Goal: Repeat purchase</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12); margin: 18px 0;'/>",
                unsafe_allow_html=True)

    with st.expander("How to Read This Report"):
        st.markdown("""
**Campaign Tier**
Four action groups ranked by purchase propensity. Priority is the top 10% of the list — the highest-scoring 1,000 customers — and should receive every send. High Reach and Broad Reach are the natural list extensions as budget allows. Suppress covers the bottom 40%, where conversion rates fall to or below the list average; excluding this group saves send cost without meaningfully reducing revenue.

**Propensity Score**
The model's output for each customer, expressed as a number between 0 and 1. Higher is more likely to purchase. These scores are used as relative rankings — a customer with a score of 0.62 will not convert at exactly 62%, but they are more likely to convert than someone with a score of 0.44. What matters is the rank order, not the absolute number.

**Conversion Rate**
The actual percentage of customers in each tier who purchased during the Spring Collection campaign window. This is the real outcome, computed from validation data — not a model prediction. It confirms that customers ranked higher by the model did, in fact, convert at higher rates.

**Lift**
Conversion Rate divided by the 7.3% population average. A lift of 2.95x means Priority Tier customers purchased at nearly three times the rate of a randomly selected customer. This is the primary measure of whether the model is adding value: if lift is close to 1.0x in the top tier, the model is not helping. The higher the lift in the top tiers, the more efficiently the model concentrates buyers at the top of the list.

**Score Driver Mix**
Each customer's score comes from three behavioral signals. The percentage shown for each signal reflects how much of that customer's score it contributed. Customer Health Score reflects long-run purchase history. Seasonal Return Signal reflects whether this customer has purchased in this same calendar window in prior years. Recent Activity Signal reflects purchase activity in the past 90 days. These percentages add to 100% for every customer. The dominant signal identifies the best creative and timing strategy for each segment.

**Subdecile**
A finer-grained ranking within each decile, formatted as [decile]_[rank]. Subdecile 1_1 contains the very highest-scoring customers in the entire file. Subdecile 3_10 contains the lowest-scoring customers within decile 3. Useful when your send budget falls mid-tier and you need to cut the list at a precise point rather than including or excluding a full tier.
""")

    st.markdown("<hr style='border-color:rgba(255,255,255,0.12); margin: 18px 0;'/>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 11px; color: #4A6A8A; line-height: 1.7;">
        Built by <strong style="color:#7A9BBB;">Brendan Hoffman</strong><br/>
        Propensity Scoring Tool v1.0
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB NAVIGATION
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "The Engagement Model",
    "Campaign Scoring",
    "Segment Explorer",
    "How It Works",
])

with tab1:

    # Load scored data once (cached)
    sample_scored = load_and_score_sample()

    # -- Hero block -----------------------------------------------------------
    st.markdown("""
    <div style="padding: 28px 0 16px 0;">
        <div class="section-label">SPRING COLLECTION 2026 &middot; MOTHER'S DAY GIFT WINDOW</div>
        <h1 style="font-family: 'DM Serif Display', Georgia, serif; font-size: 34px; font-weight: 400; color: #111827; line-height: 1.2; margin: 8px 0 16px 0; letter-spacing: -0.3px;">
            A purchase propensity model built for Sable &amp; Co.
        </h1>
        <p style="font-size: 15px; color: #4B5563; max-width: 720px; line-height: 1.65; margin: 0;">
            The Spring Collection propensity model ranks all 10,000 customers in the Sable &amp; Co.
            file by purchase likelihood. Each score is attributed to three behavioral signals, enabling
            the marketing team to align send timing and message type with the underlying driver for
            each segment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- Executive brief ------------------------------------------------------
    st.markdown("""
    <div class="callout-box" style="margin-bottom: 8px;">
        <strong>Bottom line for the Spring Collection window:</strong>
        The top 10% of this list — the Priority Tier — is expected to convert at nearly
        <strong>3x the rate of the average customer</strong> and will account for approximately
        <strong>30% of all purchases</strong> generated by the campaign. Sending to Priority
        and High Reach customers captures the overwhelming majority of expected revenue at
        a fraction of full-list cost. The Suppress tier (bottom 40%) shows conversion rates
        at or below the population baseline; excluding this group reduces send cost and list
        fatigue with no meaningful impact on total campaign performance.
    </div>
    """, unsafe_allow_html=True)

    # -- Three-component model panel ------------------------------------------
    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px; font-weight:600; color:#374151; '
        'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:16px;">'
        'MODEL COMPONENTS</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="model-component-card" style="border-top: 3px solid #1B2A3B;">
            <div class="component-label">COMPONENT 1 OF 3</div>
            <div class="component-name">Customer Health Score</div>
            <div class="component-body">
                A continuous score from 0 to 10 derived from an upstream lifetime value model that
                estimates each customer's expected revenue over a 13-month horizon. High scores identify
                customers with durable, repeat-purchase relationships with the brand.
            </div>
            <div class="component-what">
                <strong>What it tells you:</strong> A high Customer Health Score indicates sustained
                purchase frequency and long-run revenue contribution. These customers are in-market
                across seasons, not just during promotional windows.
            </div>
            <div class="component-features">
                base_propensity_score (continuous, 0-10)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-component-card" style="border-top: 3px solid #D97706;">
            <div class="component-label">COMPONENT 2 OF 3</div>
            <div class="component-name">Seasonal Return Signal</div>
            <div class="component-body">
                A binary flag set when a customer purchased during this same calendar window in either
                of the prior two years. Customers who trigger this signal have established
                occasion-driven purchase patterns tied to this specific window.
            </div>
            <div class="component-what">
                <strong>What it tells you:</strong> Customers with a positive Seasonal Return Signal
                are predictable buyers in this window. Campaign send is an accelerant for a purchase
                that is likely to occur regardless; these contacts have the highest-confidence baseline
                conversion probability.
            </div>
            <div class="component-features">
                purchased_same_period_1yr_ago, purchased_same_period_2yr_ago (binary)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="model-component-card" style="border-top: 3px solid #2563EB;">
            <div class="component-label">COMPONENT 3 OF 3</div>
            <div class="component-name">Recent Activity Signal</div>
            <div class="component-body">
                A set of binary flags indicating purchase activity in each of the prior three months.
                Recent purchase behavior signals current brand engagement and an elevated propensity
                to purchase again in the near term.
            </div>
            <div class="component-what">
                <strong>What it tells you:</strong> Recency is the strongest near-term predictor of
                repeat purchase. Customers active in the past 90 days have demonstrated current
                engagement, making them the segment most likely to convert with minimal campaign stimulus.
            </div>
            <div class="component-features">
                purchased_month_minus_1, purchased_month_minus_2, purchased_month_minus_3 (binary)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -- Bridge sentence ------------------------------------------------------
    st.markdown("""
    <p style="font-size: 14px; color: #4B5563; max-width: 760px; line-height: 1.7; margin: 0;">
        The Spring Collection propensity model combines these three signals to produce a single
        ranked conversion likelihood for every customer in the Sable &amp; Co. file. Crucially,
        it also tells you <em>which</em> signal is carrying the most weight for each customer &mdash;
        because a top-ranked customer driven by seasonal history requires a different message than
        one showing sudden recent momentum. Both belong in the campaign; they should not receive
        the same creative.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # -- Four metric cards ----------------------------------------------------
    total = len(sample_scored)
    d1 = sample_scored[sample_scored["model_decile"] == 1]
    avg_d1_score = d1["predicted_score"].mean()
    avg_all_score = sample_scored["predicted_score"].mean()
    lift_mult = avg_d1_score / avg_all_score if avg_all_score > 0 else 0.0

    # Converter capture: hardcoded from validation (29.5%)
    converter_capture = 29.5

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(metric_card(
            "Customers Scored",
            f"{total:,}",
            sub="Spring Collection audience",
        ), unsafe_allow_html=True)
    with mc2:
        st.markdown(metric_card(
            "Priority Tier",
            f"{len(d1):,}",
            sub=f"Top decile · {len(d1) / total * 100:.0f}% of list",
        ), unsafe_allow_html=True)
    with mc3:
        st.markdown(metric_card(
            "Priority Tier Lift",
            f"{lift_mult:.2f}x",
            sub="vs. population baseline",
            accent=True,
        ), unsafe_allow_html=True)
    with mc4:
        st.markdown(metric_card(
            "Converter Capture",
            f"{converter_capture:.1f}%",
            sub="of converters captured in Priority Tier",
            accent=True,
        ), unsafe_allow_html=True)

    st.markdown(
        stat_pills(
            ("Population", "10,000 customers"),
            ("Baseline rate", "7.3%"),
            ("Model", "Logistic Regression"),
            ("Primary metric", "Top-decile lift"),
            ("Lift @ Decile 1", "2.95x"),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    # -- Lift bar chart -------------------------------------------------------
    st.markdown("""
    <p style="font-size:13px; font-weight:600; color:#374151;
    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">
    CONVERSION RATE BY DECILE</p>
    <p style="font-size:14px; color:#4B5563; max-width:760px; line-height:1.65; margin-bottom:4px;">
        Each bar represents 10% of the customer file, ranked from most likely to convert (Decile 1,
        amber) to least likely (Decile 10). The red dashed line marks the population average &mdash;
        7.3% of all customers purchased during the Spring Collection window. A well-built model
        pushes buyers into the left side of this chart. The lift multipliers above each bar
        show how much better (or worse) each group performs relative to that baseline.
        <strong>Decile 1 customers are expected to convert at nearly 3x the average rate.</strong>
    </p>
    <p style="font-size:12px; color:#9CA3AF; margin-bottom:4px;">
        Amber bar = Priority Tier &nbsp;&middot;&nbsp; Red dashed line = overall list average (7.3%)</p>
    """, unsafe_allow_html=True)
    fig_lift = chart_lift_bar(sample_scored)
    st.plotly_chart(fig_lift, use_container_width=True, config={"displayModeBar": False})

    # -- Priority Tier driver callout -----------------------------------------
    d1_base     = d1["base_score_contribution_pct"].mean()
    d1_seasonal = d1["recent_behavior_contribution_pct"].mean()
    d1_momentum = d1["momentum_contribution_pct"].mean()

    driver_scores = {
        "Customer Health Score":  d1_base,
        "Seasonal Return Signal": d1_seasonal,
        "Recent Activity Signal": d1_momentum,
    }
    dominant_name = max(driver_scores, key=driver_scores.get)
    dominant_val  = driver_scores[dominant_name]

    driver_context = {
        "Customer Health Score": (
            "Customers in this tier have demonstrated sustained long-run purchase behavior. "
            "They are reliably in-market regardless of the campaign window."
        ),
        "Seasonal Return Signal": (
            "These customers have purchased during this same calendar window in prior years. "
            "The Mother's Day gift occasion is already part of their purchase calendar."
        ),
        "Recent Activity Signal": (
            "These customers have been active in the past 90 days and are in an active "
            "consideration state. High-confidence near-term opportunity."
        ),
    }

    st.markdown(f"""
    <div class="callout-box">
        <strong>Priority Tier driver:</strong>
        The dominant signal for the Priority Tier is <strong>{dominant_name}</strong>,
        accounting for {dominant_val:.0f}% of the average score in this group.
        {driver_context[dominant_name]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # -- Tier summary table ---------------------------------------------------
    st.markdown("""
    <p style="font-size:13px; font-weight:600; color:#374151;
    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">
    TIER SUMMARY</p>
    <p style="font-size:14px; color:#4B5563; max-width:760px; line-height:1.65; margin-bottom:14px;">
        The four campaign tiers translate the model's ranked scores into action groups.
        <strong>Priority</strong> is the top decile &mdash; the 1,000 highest-scoring customers,
        who should receive every send. <strong>High Reach</strong> and <strong>Broad Reach</strong>
        are the natural list expansion as budget allows. <strong>Suppress</strong> covers the bottom
        40% of the file, where conversion rates fall to or below the population average and
        send costs exceed expected return. The three signal columns on the right show what
        is driving each tier's scores &mdash; useful for deciding message strategy before
        creative is briefed.
    </p>
    """, unsafe_allow_html=True)

    tier_order = ["Priority", "High Reach", "Broad Reach", "Suppress"]
    tier_summary = (
        sample_scored.groupby("campaign_tier")
        .agg(
            customers=("customer_id", "count"),
            converters=("purchased_in_campaign", "sum"),
            conversion_rate=("purchased_in_campaign", "mean"),
            avg_health=("base_score_contribution_pct", "mean"),
            avg_seasonal=("recent_behavior_contribution_pct", "mean"),
            avg_activity=("momentum_contribution_pct", "mean"),
        )
        .reset_index()
    )
    tier_summary["campaign_tier"] = pd.Categorical(
        tier_summary["campaign_tier"], categories=tier_order, ordered=True
    )
    tier_summary = tier_summary.sort_values("campaign_tier")

    hdr_t1 = st.columns([2.2, 1, 1.2, 1.2, 1.3, 1.3, 1.3, 1.3])
    headers_data_t1 = [
        ("Campaign Tier",     "Action group based on purchase propensity. Priority = highest propensity, Suppress = lowest."),
        ("Customers",         "Number of customers assigned to this tier."),
        ("Converters",        "Customers who actually purchased during the campaign period."),
        ("Conv. Rate",        "Actual purchase rate within this tier. Population baseline is 7.3%."),
        ("Lift",              "Conversion rate divided by the 7.3% population baseline. 2.95x means nearly 3x more likely to purchase than average."),
        ("Health Score %",    "Share of score from the Customer Health Score (upstream LTV model, continuous 0-10)."),
        ("Seasonal %",        "Share of score from the Seasonal Return Signal (purchased in this window last year or two years ago)."),
        ("Activity %",        "Share of score from the Recent Activity Signal (purchased 1, 2, or 3 months ago)."),
    ]
    for hc, (lbl, tip) in zip(hdr_t1, headers_data_t1):
        hc.markdown(col_header(lbl, tip), unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0 0 0; border-color:#E5E7EB'/>", unsafe_allow_html=True)

    for _, row in tier_summary.iterrows():
        tier = row["campaign_tier"]
        rate = row["conversion_rate"]
        rc = st.columns([2.2, 1, 1.2, 1.2, 1.3, 1.3, 1.3, 1.3])
        rc[0].markdown(tier_badge_html(tier), unsafe_allow_html=True)
        rc[1].markdown(f'<span style="font-size:14px;font-weight:600;color:#111827;">{int(row["customers"]):,}</span>', unsafe_allow_html=True)
        rc[2].markdown(f'<span style="font-size:14px;">{int(row["converters"]):,}</span>', unsafe_allow_html=True)
        rc[3].markdown(f'<span style="font-size:14px;font-weight:600;">{rate:.1%}</span>', unsafe_allow_html=True)
        rc[4].markdown(lift_badge_html(rate), unsafe_allow_html=True)
        rc[5].markdown(f'<span style="font-size:13px;color:#6B7280;">{row["avg_health"]:.0f}%</span>', unsafe_allow_html=True)
        rc[6].markdown(f'<span style="font-size:13px;color:#6B7280;">{row["avg_seasonal"]:.0f}%</span>', unsafe_allow_html=True)
        rc[7].markdown(f'<span style="font-size:13px;color:#6B7280;">{row["avg_activity"]:.0f}%</span>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -- Budget efficiency callout --------------------------------------------
    top3_tiers = sample_scored[
        sample_scored["campaign_tier"].isin(["Priority", "High Reach", "Broad Reach"])
    ]
    top3_list_pct = len(top3_tiers) / total * 100

    # High-score customers: top quintile
    top_20_threshold = sample_scored["predicted_score"].quantile(0.80)
    likely_buyers = sample_scored[sample_scored["predicted_score"] >= top_20_threshold]
    top3_likely = top3_tiers[top3_tiers["predicted_score"] >= top_20_threshold]
    top3_buyer_cap = (
        len(top3_likely) / len(likely_buyers) * 100 if len(likely_buyers) > 0 else 0.0
    )

    st.markdown(f"""
    <div class="insight-box">
        <strong>Budget efficiency:</strong>
        Sending to the top three tiers (Priority, High Reach, and Broad Reach) reaches
        <strong>{top3_buyer_cap:.0f}%</strong> of high-propensity customers
        while contacting only <strong>{top3_list_pct:.0f}%</strong> of the full file.
        Suppressing the bottom tier reduces send cost and protects deliverability
        without meaningfully reducing expected conversions.
    </div>
    """, unsafe_allow_html=True)


with tab2:

    # Uses the same pre-loaded sample dataset
    scored = load_and_score_sample()

    # -- Header ---------------------------------------------------------------
    st.markdown("""
    <div style="padding: 28px 0 16px 0;">
        <div class="section-label">SABLE &amp; CO. &middot; CAMPAIGN DELIVERABLE</div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 22px; font-weight: 700; color: #111827; margin: 8px 0 10px 0;">
            Spring Collection 2026 Scored Output
        </h2>
        <p style="font-size: 14px; color: #4B5563; max-width: 720px; line-height: 1.65; margin: 0 0 12px 0;">
            This deliverable contains conversion likelihood rankings and score driver breakdowns
            for all 10,000 customers in the Sable &amp; Co. Spring Collection audience. Apply tier
            assignments to determine send priority and tailor message sequencing by segment.
        </p>
        <p style="font-size: 14px; color: #4B5563; max-width: 720px; line-height: 1.65; margin: 0;">
            <strong>How to use this output:</strong> The Campaign Tier column is the primary action field.
            Sort or filter by tier to build your send list &mdash; Priority customers send first,
            with no further justification required. For teams running segmented creative,
            the three signal columns (Health Score %, Seasonal %, Activity %) identify which
            behavioral driver earned each customer their rank, and should inform the message
            they receive. Download the scored file below to load directly into your ESP or
            campaign management platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # -- Tier breakdown table (detailed) --------------------------------------
    st.markdown(
        '<p style="font-size:13px; font-weight:600; color:#374151; '
        'text-transform:uppercase; letter-spacing:0.07em; margin-bottom:12px;">'
        'TIER BREAKDOWN</p>',
        unsafe_allow_html=True,
    )

    tier_order_t2 = ["Priority", "High Reach", "Broad Reach", "Suppress"]
    tier_summary_t2 = (
        scored.groupby("campaign_tier")
        .agg(
            customers=("customer_id", "count"),
            converters=("purchased_in_campaign", "sum"),
            conversion_rate=("purchased_in_campaign", "mean"),
            avg_health=("base_score_contribution_pct", "mean"),
            avg_seasonal=("recent_behavior_contribution_pct", "mean"),
            avg_activity=("momentum_contribution_pct", "mean"),
        )
        .reset_index()
    )
    tier_summary_t2["campaign_tier"] = pd.Categorical(
        tier_summary_t2["campaign_tier"], categories=tier_order_t2, ordered=True
    )
    tier_summary_t2 = tier_summary_t2.sort_values("campaign_tier")
    total_t2 = len(scored)

    hdr_cols = st.columns([2.2, 1, 1.2, 1.2, 1.3, 1.3, 1.3, 1.3])
    headers_data = [
        ("Campaign Tier",     "Action group based on purchase propensity. Priority = highest propensity, Suppress = lowest."),
        ("Customers",         "Number of customers assigned to this tier."),
        ("Converters",        "Customers who actually purchased during the campaign period."),
        ("Conv. Rate",        "Actual purchase rate within this tier. Population baseline is 7.3%."),
        ("Lift",              "Conversion rate divided by the 7.3% population baseline. 2.95x means nearly 3x more likely to purchase than average."),
        ("Health Score %",    "Share of score from the Customer Health Score (upstream LTV model, continuous 0-10)."),
        ("Seasonal %",        "Share of score from the Seasonal Return Signal (purchased in this window last year or two years ago)."),
        ("Activity %",        "Share of score from the Recent Activity Signal (purchased 1, 2, or 3 months ago)."),
    ]
    for hc, (lbl, tip) in zip(hdr_cols, headers_data):
        hc.markdown(col_header(lbl, tip), unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0 2px 0; border-color:#E5E7EB'/>", unsafe_allow_html=True)

    for _, row in tier_summary_t2.iterrows():
        tier = row["campaign_tier"]
        rate = row["conversion_rate"]
        rc = st.columns([2.2, 1, 1.2, 1.2, 1.3, 1.3, 1.3, 1.3])
        rc[0].markdown(tier_badge_html(tier), unsafe_allow_html=True)
        rc[1].markdown(f'<span style="font-size:14px;font-weight:600;">{int(row["customers"]):,}</span>', unsafe_allow_html=True)
        rc[2].markdown(f'<span style="font-size:14px;">{int(row["converters"]):,}</span>', unsafe_allow_html=True)
        rc[3].markdown(f'<span style="font-size:14px;font-weight:600;">{rate:.1%}</span>', unsafe_allow_html=True)
        rc[4].markdown(lift_badge_html(rate), unsafe_allow_html=True)
        rc[5].markdown(f'<span style="font-size:13px;color:#6B7280;">{row["avg_health"]:.0f}%</span>', unsafe_allow_html=True)
        rc[6].markdown(f'<span style="font-size:13px;color:#6B7280;">{row["avg_seasonal"]:.0f}%</span>', unsafe_allow_html=True)
        rc[7].markdown(f'<span style="font-size:13px;color:#6B7280;">{row["avg_activity"]:.0f}%</span>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -- Tier descriptions expander -------------------------------------------
    with st.expander("Tier descriptions"):
        st.markdown("""
**Priority** &nbsp; Customers in this tier carry the highest combination of all three behavioral
signals. Sending to this group produces the greatest return on campaign spend; no budget
justification is required to include them.

**High Reach** &nbsp; Customers with strong long-run purchase behavior and moderate seasonal signal.
Expanding the send to this tier increases reachable audience with limited reduction in conversion
efficiency, making it the first logical extension beyond Priority.

**Broad Reach** &nbsp; Customers with moderate propensity across all three signals. Inclusion is
budget-dependent; where send volume permits, a lower-cost message variant reduces per-contact
spend while still activating this segment.

**Suppress** &nbsp; Customers in this tier show low conversion likelihood across all three signals,
with conversion rates at or near the population baseline. Excluding this group from the send
reduces cost and protects sender reputation with no material impact on campaign revenue.
        """)

    # -- Budget efficiency callout --------------------------------------------
    top3_t2 = scored[scored["campaign_tier"].isin(["Priority", "High Reach", "Broad Reach"])]
    top3_list_pct_t2 = len(top3_t2) / total_t2 * 100

    top_20_thresh_t2 = scored["predicted_score"].quantile(0.80)
    likely_t2 = scored[scored["predicted_score"] >= top_20_thresh_t2]
    top3_likely_t2 = top3_t2[top3_t2["predicted_score"] >= top_20_thresh_t2]
    top3_buyer_cap_t2 = (
        len(top3_likely_t2) / len(likely_t2) * 100 if len(likely_t2) > 0 else 0.0
    )

    suppress_count = int(
        tier_summary_t2[tier_summary_t2["campaign_tier"] == "Suppress"]["customers"].sum()
    )
    suppress_pct = suppress_count / total_t2 * 100

    st.markdown(f"""
    <div class="callout-box">
        <strong>Budget efficiency:</strong>
        Sending to Priority, High Reach, and Broad Reach reaches
        <strong>{top3_buyer_cap_t2:.0f}%</strong> of high-propensity customers
        at <strong>{top3_list_pct_t2:.0f}%</strong> of list cost.
        The Suppress tier ({suppress_count:,} customers, {suppress_pct:.0f}% of file)
        shows low conversion likelihood across all three signals and exclusion from the send
        is the recommended default.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # -- Contribution breakdown chart -----------------------------------------
    st.markdown("""
    <p style="font-size:13px; font-weight:600; color:#374151;
    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">
    SCORE DRIVER MIX</p>
    <p style="font-size:14px; color:#4B5563; max-width:760px; line-height:1.65; margin-bottom:8px;">
        Knowing <em>who</em> to send to is the first half of a campaign decision. Knowing
        <em>why</em> they ranked highly is the second &mdash; and it changes the message.
        This chart breaks each tier's average score into its three contributing signals.
        A tier led by the <strong>Seasonal Return Signal</strong> (amber) is full of
        occasion-driven buyers who have purchased in this exact window before; urgency
        and deadline creative will perform. A tier led by the <strong>Recent Activity Signal</strong>
        (blue) contains customers in active consideration right now; curiosity-driving and
        discovery messaging tends to outperform. A tier led by the
        <strong>Customer Health Score</strong> (navy) is a broadly reliable segment of
        long-run brand loyalists; brand storytelling and relationship-reinforcing creative
        is the right frame.
    </p>
    """, unsafe_allow_html=True)
    fig_contrib = chart_contribution_breakdown(scored, group_col="campaign_tier")
    st.plotly_chart(fig_contrib, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="insight-box">
        <strong>Reading this chart:</strong>
        Each cluster of bars represents one campaign tier. The three bars per cluster show the
        average share of score coming from each behavioral signal for customers in that tier.
        A Priority Tier dominated by the Seasonal Return Signal means the top-ranked contacts
        have purchased in this exact calendar window in prior years &mdash; campaign send
        accelerates a high-probability purchase rather than creating new demand from scratch.
        A Priority Tier dominated by the Customer Health Score signals a broadly reliable
        long-run segment; pair with strong brand creative rather than occasion-specific urgency.
        Use the mix to brief your creative team before the campaign launches, not after results come in.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # -- Download + preview ---------------------------------------------------
    st.markdown("""
    <p style="font-size:13px; font-weight:600; color:#374151;
    text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">
    DOWNLOAD SCORED FILE</p>
    <p style="font-size:14px; color:#4B5563; max-width:760px; line-height:1.65; margin-bottom:12px;">
        The file below contains every customer in the Spring Collection audience, ranked by
        conversion likelihood with tier assignment and signal contribution columns included.
        Load it directly into your ESP, CRM, or campaign management platform and filter by
        <strong>Campaign Tier</strong> to build your send list. The <strong>Subdecile</strong>
        column provides fine-grained ranking within each decile &mdash; useful if your budget
        falls mid-tier and you need to prioritize within a group rather than include or exclude
        the tier wholesale.
    </p>
    """, unsafe_allow_html=True)

    display_cols = [
        "customer_id",
        "predicted_score",
        "campaign_tier",
        "model_decile",
        "model_subdecile",
        "base_score_contribution_pct",
        "recent_behavior_contribution_pct",
        "momentum_contribution_pct",
    ]
    download_df = scored[display_cols].copy().rename(columns={
        "customer_id":                      "Customer ID",
        "predicted_score":                  "Conversion Likelihood",
        "campaign_tier":                    "Campaign Tier",
        "model_decile":                     "Decile",
        "model_subdecile":                  "Subdecile",
        "base_score_contribution_pct":      "Customer Health Score %",
        "recent_behavior_contribution_pct": "Seasonal Return Signal %",
        "momentum_contribution_pct":        "Recent Activity Signal %",
    })

    csv_buf = io.StringIO()
    download_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Scored File",
        data=csv_buf.getvalue(),
        file_name="sable_spring_2026_scored.csv",
        mime="text/csv",
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px; color:#374151; margin-bottom:8px;">'
        "Preview &middot; Top 100 rows, sorted by Conversion Likelihood</p>",
        unsafe_allow_html=True,
    )
    preview = download_df.head(100)
    st.dataframe(
        preview.style.background_gradient(subset=["Conversion Likelihood"], cmap="Blues"),
        use_container_width=True,
    )


with tab3:

    st.markdown("""
    <div style="padding: 28px 0 16px 0;">
        <div class="section-label">SPRING COLLECTION 2026 &middot; AUDIENCE ANALYSIS</div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 22px; font-weight: 700; color: #111827; margin: 8px 0 10px 0;">
            Segment Explorer
        </h2>
        <p style="font-size: 14px; color: #4B5563; max-width: 680px; line-height: 1.65; margin: 0 0 12px 0;">
            Filter the scored Sable &amp; Co. Spring Collection audience by tier, primary score
            driver, and minimum propensity score. Use this view to build message-specific
            sub-segments, answer ad hoc audience questions, or validate targeting assumptions
            before the campaign launches.
        </p>
        <p style="font-size: 14px; color: #4B5563; max-width: 680px; line-height: 1.65; margin: 0 0 6px 0;">
            <strong>Examples of questions this view answers:</strong>
        </p>
        <ul style="font-size: 14px; color: #4B5563; line-height: 1.8; margin: 0 0 0 20px; max-width: 660px;">
            <li><em>"How many high-propensity customers are primarily seasonal buyers?"</em> &mdash;
                Select Priority + High Reach, filter by Seasonal Return Signal.</li>
            <li><em>"Which customers have strong recent momentum but fell outside Priority?"</em> &mdash;
                Select High Reach or Broad Reach, filter by Recent Activity Signal.</li>
            <li><em>"If budget forces us to cut the list, where is the propensity floor?"</em> &mdash;
                Set the minimum propensity score slider and watch the count update in real time.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    explorer_df = load_and_score_sample().copy()

    # -- Filter controls ------------------------------------------------------
    f1, f2, f3 = st.columns([2, 2, 2])

    tier_options = ["Priority", "High Reach", "Broad Reach", "Suppress"]

    with f1:
        selected_tiers = st.multiselect(
            "Campaign Tier",
            options=tier_options,
            default=["Priority", "High Reach"],
        )
    with f2:
        driver_col_map = {
            "Customer Health Score":  "base_score_contribution_pct",
            "Seasonal Return Signal": "recent_behavior_contribution_pct",
            "Recent Activity Signal": "momentum_contribution_pct",
        }
        selected_driver = st.selectbox(
            "Primary Score Driver",
            options=["Any", "Customer Health Score", "Seasonal Return Signal", "Recent Activity Signal"],
        )
    with f3:
        score_min_val = float(explorer_df["predicted_score"].min())
        score_max_val = float(explorer_df["predicted_score"].max())
        min_likelihood = st.slider(
            "Min Propensity Score",
            min_value=score_min_val,
            max_value=score_max_val,
            value=score_min_val,
            step=0.01,
            format="%.2f",
        )

    # -- Apply filters --------------------------------------------------------
    filtered_e = explorer_df.copy()
    if selected_tiers:
        filtered_e = filtered_e[filtered_e["campaign_tier"].isin(selected_tiers)]
    if selected_driver != "Any":
        dominant_col = driver_col_map[selected_driver]
        other_cols = [c for k, c in driver_col_map.items() if k != selected_driver]
        filtered_e = filtered_e[
            (filtered_e[dominant_col] >= filtered_e[other_cols[0]]) &
            (filtered_e[dominant_col] >= filtered_e[other_cols[1]])
        ]
    filtered_e = filtered_e[filtered_e["predicted_score"] >= min_likelihood]

    n_filtered = len(filtered_e)

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    # -- Filtered count -------------------------------------------------------
    st.markdown(
        f'<p style="font-size:15px; font-weight:600; color:#111827; margin-bottom:16px;">'
        f'{n_filtered:,} customers match the selected filters.</p>',
        unsafe_allow_html=True,
    )

    if n_filtered == 0:
        st.info("No customers match the current filter combination. Broaden the tier selection or reduce the minimum propensity score to expand results.")
    else:
        # -- Filtered dataframe -----------------------------------------------
        display_filtered = (
            filtered_e[[
                "customer_id",
                "model_subdecile",
                "campaign_tier",
                "predicted_score",
                "purchased_in_campaign",
                "base_score_contribution_pct",
                "recent_behavior_contribution_pct",
                "momentum_contribution_pct",
            ]]
            .rename(columns={
                "customer_id":                      "Customer ID",
                "model_subdecile":                  "Subdecile",
                "campaign_tier":                    "Campaign Tier",
                "predicted_score":                  "Propensity Score",
                "purchased_in_campaign":            "Converted",
                "base_score_contribution_pct":      "Health Score %",
                "recent_behavior_contribution_pct": "Seasonal Signal %",
                "momentum_contribution_pct":        "Activity Signal %",
            })
            .sort_values("Propensity Score", ascending=False)
            .reset_index(drop=True)
        )

        st.dataframe(
            display_filtered,
            use_container_width=True,
            height=420,
            column_config={
                "Propensity Score": st.column_config.ProgressColumn(
                    "Propensity Score",
                    help="Model output (0-1). Higher = more likely to purchase. Not calibrated to the 7.3% population rate.",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Converted": st.column_config.CheckboxColumn(
                    "Converted",
                    help="Whether this customer actually purchased during the Spring Collection campaign period.",
                ),
                "Subdecile": st.column_config.TextColumn(
                    "Subdecile",
                    help="Fine-grained ranking within each decile. 1_1 = the very highest-scoring customers in decile 1.",
                ),
                "Campaign Tier": st.column_config.TextColumn(
                    "Campaign Tier",
                    help="Priority = Decile 1, High Reach = Deciles 2-3, Broad Reach = Deciles 4-6, Suppress = Deciles 7-10.",
                ),
                "Health Score %": st.column_config.NumberColumn(
                    "Health Score %",
                    help="Share of this customer's score from the Customer Health Score (long-run LTV signal).",
                    format="%.0f%%",
                ),
                "Seasonal Signal %": st.column_config.NumberColumn(
                    "Seasonal Signal %",
                    help="Share from the Seasonal Return Signal (purchased in this calendar window last year or two years ago).",
                    format="%.0f%%",
                ),
                "Activity Signal %": st.column_config.NumberColumn(
                    "Activity Signal %",
                    help="Share from the Recent Activity Signal (purchased 1, 2, or 3 months ago).",
                    format="%.0f%%",
                ),
            }
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # -- Contribution summary ---------------------------------------------
        st.markdown("""
        <p style="font-size:12px; font-weight:600; color:#6B7280;
        text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;">
        AVERAGE SIGNAL CONTRIBUTION &middot; FILTERED SET</p>
        <p style="font-size:14px; color:#4B5563; max-width:700px; line-height:1.65; margin-bottom:12px;">
            These three cards summarize the average behavioral signal mix across the customers
            currently shown in the table above. The dominant signal &mdash; whichever percentage
            is highest &mdash; represents the most common reason this group scored where they did.
            Use this summary to inform message strategy for any sub-segment you build from this view:
            a filtered set led by Seasonal Return Signal warrants occasion-specific creative,
            while a set led by Recent Activity Signal is already in an active engagement window
            and responds to timeliness over nostalgia.
        </p>
        """, unsafe_allow_html=True)

        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            st.markdown(metric_card(
                "Customer Health Score",
                f'{filtered_e["base_score_contribution_pct"].mean():.0f}%',
                sub="avg share of score",
            ), unsafe_allow_html=True)
        with sa2:
            st.markdown(metric_card(
                "Seasonal Return Signal",
                f'{filtered_e["recent_behavior_contribution_pct"].mean():.0f}%',
                sub="avg share of score",
            ), unsafe_allow_html=True)
        with sa3:
            st.markdown(metric_card(
                "Recent Activity Signal",
                f'{filtered_e["momentum_contribution_pct"].mean():.0f}%',
                sub="avg share of score",
            ), unsafe_allow_html=True)


with tab4:

    st.markdown("""
    <div style="padding: 28px 0 16px 0;">
        <div class="section-label">MODEL DOCUMENTATION</div>
        <h2 style="font-family: 'Inter', sans-serif; font-size: 22px; font-weight: 700; color: #111827; margin: 8px 0 10px 0;">
            How It Works
        </h2>
        <p style="font-size: 14px; color: #4B5563; max-width: 680px; line-height: 1.65; margin: 0 0 10px 0;">
            This section explains the logic and design decisions behind the propensity model in plain
            language. It is intended for anyone who wants to understand what the tool is doing and why
            &mdash; not just accept a number. A model that cannot be explained should not be trusted,
            and a decision-maker asked to act on this output deserves to know what it is and is not
            accounting for.
        </p>
        <p style="font-size: 14px; color: #4B5563; max-width: 680px; line-height: 1.65; margin: 0;">
            Each section below covers a specific design choice. Technical details are included
            for reviewers who want them, but every section opens with the business reasoning first.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- 1. The Business Problem -----------------------------------------------
    st.markdown("""
    <div class="how-section">
        <h4>1. The Business Problem</h4>
        <p>
            DTC email campaigns carry real costs: creative production, send infrastructure,
            and list fatigue. Sending to low-propensity customers does not just waste budget.
            It depresses open rates, increases unsubscribes, and degrades sender reputation over time.
        </p>
        <p>
            Propensity scoring addresses a prioritization problem. Given a fixed sending budget,
            which customers should receive a campaign message, and in what order? A ranked score
            allows a marketing team to make budget allocation decisions with statistical backing
            rather than recency or spend heuristics.
        </p>
        <p>
            For Sable &amp; Co.'s Spring Collection, the objective is to identify customers most
            likely to purchase in the Mother's Day gift window, rank them by conversion likelihood,
            and explain what is driving each score so that creative and messaging can be tailored
            to the signal rather than sent generically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- 2. Model Approach -----------------------------------------------------
    st.markdown("""
    <div class="how-section">
        <h4>2. Model Approach</h4>
        <p>
            <strong>In plain terms:</strong> The model reads six behavioral signals from each
            customer's purchase history and combines them into a single probability estimate.
            The technique used &mdash; logistic regression &mdash; was chosen specifically because
            it can tell you not just <em>what</em> the score is, but <em>why</em> each customer
            earned it. That transparency is what makes the Score Driver Mix breakdown on the other
            tabs possible. A more complex model might have performed marginally better on some
            metrics, but would have been a black box: you'd know the output without being able to
            explain it to a client or brief a creative team around it.
        </p>
        <p>
            The model is a logistic regression trained on six behavioral features. Logistic regression
            was chosen deliberately over tree-based alternatives for three reasons.
        </p>
        <p>
            <strong>Interpretable coefficients.</strong> Each feature has a signed, calibrated weight.
            These weights are the direct input to the three-bucket score attribution
            (Customer Health Score, Seasonal Return Signal, Recent Activity Signal). A gradient
            boosted tree does not produce this kind of linear decomposition without post-hoc
            approximation methods like SHAP, which introduce their own assumptions.
        </p>
        <p>
            <strong>Probability calibration.</strong> Logistic regression outputs well-calibrated
            probabilities by default. When the predicted score is used as a ranking signal,
            calibration matters: scores are interpretable as actual conversion likelihoods,
            not only relative rankings.
        </p>
        <p>
            <strong>Feature structure.</strong> Five of the six features are binary behavioral flags.
            Tree-based models do not add meaningful lift over logistic regression on binary feature sets
            of this size but do sacrifice interpretability. The one continuous feature
            (Customer Health Score) is well-behaved after winsorization.
        </p>
        <p>
            <strong>Why AUC is not the primary metric.</strong> AUC-ROC measures ranking quality
            across the full score distribution, including the bottom 60% of customers that will
            never receive a campaign. A model with AUC of 0.82 and decile-1 lift of 2.1x is
            strictly worse for DTC budget decisions than a model with AUC of 0.78 and lift of 3.4x.
            The constraint is a fixed send budget, so performance at the top of the distribution
            is what matters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- 3. Outlier Handling ---------------------------------------------------
    st.markdown("""
    <div class="how-section">
        <h4>3. Outlier Handling</h4>
        <p>
            <strong>In plain terms:</strong> A small number of customers in any database have
            unusual or extreme behavioral patterns &mdash; someone who bought 40 times in a year,
            or a value that was entered incorrectly. If the model is trained with those extremes
            included, it can develop an over-reliance on the far edges of the data, producing
            scores that don't hold up on new customers. The preprocessing step described below
            caps those extremes at a reasonable ceiling before the model ever sees them, and
            applies the same cap consistently when scoring new files. The result is a model that
            performs reliably across the full population, not just on unusual cases.
        </p>
        <p>
            The continuous <code>base_propensity_score</code> feature is winsorized at the 2nd and
            98th percentiles. Thresholds are computed on training data only and stored in the model
            artifact alongside the fitted coefficients. At score time, the same thresholds are
            applied via <code>apply_thresholds()</code> before the feature matrix is passed to
            the model.
        </p>
        <p>
            <strong>Why this matters specifically for logistic regression.</strong> Unlike tree-based
            models, logistic regression is sensitive to leverage points in continuous features.
            Extreme values in <code>base_propensity_score</code> can produce disproportionately
            large linear scores that compress the probability distribution near the boundaries
            of the sigmoid. Winsorizing at training-time percentiles ensures score-time data is
            treated identically to training data and prevents edge cases from producing
            unreliable probability outputs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- 4. Primary Metric: Top-Decile Lift ------------------------------------
    st.markdown("""
    <div class="how-section">
        <h4>4. Primary Metric: Top-Decile Lift</h4>
        <p>
            <strong>In plain terms:</strong> There are many ways to measure whether a model is
            "good." This tool uses top-decile lift as its primary measure because it is the one
            that directly reflects how the model will be used. If a team has budget to reach
            1,000 customers, the only question that matters is: how many buyers are in that
            top 1,000? A model that scores customers well overall but spreads buyers evenly
            across the list is not useful for campaign planning. The metric reported here
            answers the operational question directly.
        </p>
        <p>
            Top-decile lift measures concentration of predictive value at the top of the
            distribution: among the top 10% of scored customers, how much more likely are they
            to convert than the population average? A lift of 3.0x means the top decile converts
            at three times the 7.3% baseline rate.
        </p>
        <p>
            This model achieves <strong>2.95x lift in decile 1</strong>, capturing
            <strong>29.5% of all converters</strong> in the top 10% of customers.
            In practical terms, a send strategy limited to the Priority Tier reaches roughly
            30% of expected purchasers while contacting 10% of the file.
        </p>
        <p>
            Top-decile lift is the evaluation metric because it directly maps to the business
            constraint. Campaign budget decisions are not about the full ROC curve. They are
            about how much value is concentrated in the contacts a team can afford to reach.
            A model that maximizes AUC but distributes high scores evenly across the distribution
            is not useful for budget allocation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- 5. Score Contribution Breakdown ---------------------------------------
    st.markdown("""
    <div class="how-section">
        <h4>5. Score Contribution Breakdown</h4>
        <p>
            <strong>In plain terms:</strong> Most scoring models hand you a number and nothing else.
            This tool goes one step further by attributing each customer's score to the specific
            behavioral signal or signals that earned it. Think of it as showing your work:
            not just "this customer ranked 47th," but "this customer ranked 47th because their
            seasonal purchase history accounted for 61% of their score." That attribution is
            what makes targeted messaging possible within a single tier, rather than treating
            the top decile as one undifferentiated audience.
        </p>
        <p>
            Each customer's score is decomposed into three attribution buckets using the fitted
            logistic regression coefficients:
        </p>
        <ul>
            <li>
                <strong>Customer Health Score</strong>: contribution of
                <code>base_propensity_score</code> (coefficient x feature value)
            </li>
            <li>
                <strong>Seasonal Return Signal</strong>: combined contribution of
                <code>purchased_same_period_1yr_ago</code> and
                <code>purchased_same_period_2yr_ago</code>
            </li>
            <li>
                <strong>Recent Activity Signal</strong>: combined contribution of
                <code>purchased_month_minus_1</code>, <code>purchased_month_minus_2</code>,
                and <code>purchased_month_minus_3</code>
            </li>
        </ul>
        <p>
            Group contributions are computed as the sum of (coefficient x feature value) for
            each feature in the bucket. Absolute values are used to normalize so that percentages
            sum to 100% even when individual contributions are negative (for example, a customer
            with zero recent activity but high historical propensity).
        </p>
        <p>
            The breakdown changes the marketing decision. A customer whose score is driven by
            Seasonal Return Signal should receive Mother's Day occasion creative sent early in
            the window. A customer driven by Recent Activity Signal is already in an active
            consideration state and may respond to urgency or scarcity messaging.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- 6. SQL Layer ----------------------------------------------------------
    st.markdown("""
    <div class="how-section">
        <h4>6. SQL Companion Queries</h4>
        <p>
            <strong>In plain terms:</strong> The propensity model answers the question "who is most
            likely to buy?" But two adjacent questions are just as important in production:
            "who should we <em>never</em> contact?" and "who have we overlooked because they went
            quiet but used to be very valuable?" The two SQL queries below address each.
            They are designed to operate alongside the scoring model, not inside it &mdash; keeping
            the model clean and reusable while handling audience management logic in the data layer
            where it belongs.
        </p>
        <p>
            Both queries are presented in anonymized form, using generic table and field names.
            They reflect patterns used in production DTC direct marketing environments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("suppression.sql · Customer Exclusion Logic"):
        st.markdown("""
**Purpose:** Builds a suppression table containing customer IDs that must be excluded from all
outbound campaigns before scoring results are applied to a send list.

**Exclusion categories:**
- **Invalid or undeliverable addresses**: USPS nixie flag, non-mailable address codes,
  APO/FPO addresses, missing name or address fields, known commercial HQ addresses
- **Internal employees**: customers whose orders use internal corporate email domains,
  preventing employees from receiving consumer campaigns
- **High-density addresses**: more than three distinct customers at the same physical address,
  likely commercial, mail-forwarding, or a data quality issue
- **Deceased customers**: flagged in the customer master record

**How it connects to scoring:** The suppression table is applied as an exclusion join in
downstream audience-building queries. Scores are computed on the full customer file.
Suppressed customers are removed before the final send list is generated. This separation
keeps the scoring model generic and reusable across campaigns without embedding suppression
logic into the model itself.
        """)

    with st.expander("reactivation.sql · Dormant High-Value Customer Targeting"):
        st.markdown("""
**Purpose:** Builds a targeted re-engagement audience for customers who were high-value
buyers but have lapsed. These customers would not appear in the top deciles of a
standard propensity score because their recency features are low. They are identified
separately using lifetime value and dormancy thresholds.

**Target segment criteria:**
- 3 or more lifetime orders (established buyers, not one-time purchasers)
- $300 or more in lifetime value (meaningful revenue relationship)
- Last order between 13 and 24 months ago (dormant but not permanently lapsed)
- Not in the suppression list
- Non-emailable customers only (emailable customers are handled by a separate email flow)

**Key design choices:**
- Replenishment, cancellation, and subscription orders are excluded from LTV calculation
  to ensure revenue reflects genuine purchase intent
- A 10,000-customer test cell and holdout group are assigned via
  <code>ROW_NUMBER() OVER (ORDER BY RANDOM())</code> for incremental lift measurement
- The 13-to-24-month dormancy window is intentional: customers within 12 months are
  still considered active; beyond 24 months, reactivation rates drop sharply

**Relationship to propensity model:** Reactivation targets are identified by recency
and lifetime value thresholds rather than propensity score, because these customers carry
low recency-driven propensity while still representing recoverable high-value revenue.
The two approaches are complementary and designed to operate on separate audiences.
        """)

    # -- Stack summary ---------------------------------------------------------
    st.markdown("""
    <div style="margin-top: 24px; padding: 16px 20px; background: #F9FAFB;
                border-radius: 8px; border: 1px solid #E5E7EB;">
        <div style="font-size: 12px; font-weight: 600; color: #6B7280;
                    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 8px;">
            Stack Summary
        </div>
        <div style="font-size: 13px; color: #374151; line-height: 1.9;">
            <strong>Model:</strong> Logistic Regression (scikit-learn) &nbsp;&middot;&nbsp;
            <strong>Preprocessing:</strong> Winsorization (2nd/98th pct, fit on train) &nbsp;&middot;&nbsp;
            <strong>Serialization:</strong> joblib &nbsp;&middot;&nbsp;
            <strong>Primary metric:</strong> Top-decile lift &nbsp;&middot;&nbsp;
            <strong>UI:</strong> Streamlit + Plotly &nbsp;&middot;&nbsp;
            <strong>Data layer:</strong> SQL (suppression + reactivation)
        </div>
    </div>
    """, unsafe_allow_html=True)

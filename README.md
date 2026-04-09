# Propensity Scoring Tool for Digital Marketing Teams

A scoring pipeline that identifies which customers to target for an upcoming email or digital campaign — ranked by likelihood to convert, with a breakdown of what's driving each score.

**Live demo:** `streamlit run app/streamlit_app.py`

---

## The Problem It Solves

For a retailer running dozens of email and digital campaigns per year, the core question is always the same: *who do you target, and when?*

Broadcast too broadly, and you waste budget on customers unlikely to respond — and risk deliverability. Target too narrowly, and you miss revenue. The answer isn't just a ranked list; it's understanding *why* a customer scored high, because that changes how you treat them.

A customer in the top decile because they bought in this exact campaign window last year is a different conversation than one who's been accelerating in the past 3 months and hasn't bought yet.

---

## What It Does

1. **Scores** every customer in your database on their likelihood to convert in an upcoming campaign window
2. **Ranks** them into 10 deciles (decile 1 = highest propensity) and sub-deciles for finer-grained targeting
3. **Explains** each customer's score by decomposing it into three contribution buckets:
   - **Historical propensity** — long-run base score from purchase history
   - **Same-period history** — did they buy in this campaign window in prior years?
   - **Recent momentum** — have they been active in the last 1–3 months?
4. **Validates** model quality by measuring top-decile lift: not overall accuracy, but specifically whether high-scoring customers convert at meaningfully higher rates than the population

### Key insight on model evaluation

Overall AUC can be misleading for campaign use cases. A model can have high AUC but spread buyers across multiple deciles — and for a campaign budget, you need buyers concentrated in decile 1. This tool reports **top-decile concentration** as the primary success metric, alongside standard accuracy measures.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (10,000 customers, no real data required)
python src/generate_data.py

# Train the model
python src/model.py

# Score the full dataset
python src/score.py

# Run lift validation analysis + chart
python src/validate.py

# Launch the interactive app
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
propensity-scoring-tool/
  src/
    generate_data.py     Generate synthetic customer dataset
    model.py             Train logistic regression; evaluate top-decile lift
    score.py             Score a customer CSV; output deciles + contribution %
    validate.py          Post-campaign lift analysis and chart
  app/
    streamlit_app.py     Interactive Streamlit app (upload, score, explore)
  sql/
    suppression.sql      Audience exclusion logic (invalid addresses, employees, etc.)
    reactivation.sql     Dormant high-value customer re-engagement targeting
  data/
    synthetic/           Generated dataset (created by generate_data.py)
```

---

## Model Features

The model uses six behavioral signals derived from purchase history:

| Feature | Description |
|---|---|
| `base_propensity_score` | Continuous score from an upstream model reflecting long-run purchase frequency |
| `purchased_same_period_1yr_ago` | Binary: did this customer buy in this campaign window last year? |
| `purchased_same_period_2yr_ago` | Binary: did they buy in this window two years ago? |
| `purchased_month_minus_1/2/3` | Binary: activity in the 1, 2, or 3 months preceding the campaign |

The model outputs a probability score that is then ranked into deciles and sub-deciles. The decile-1 segment historically shows 2–3x lift over the population average conversion rate.

---

## Score Contribution Breakdown

Each scored customer includes three contribution columns that explain what's driving their score:

- **`base_score_contribution_pct`** — customers driven primarily by historical propensity are reliable repeat buyers; they've earned their spot through consistent long-term behavior
- **`recent_behavior_contribution_pct`** — customers driven by same-period history are seasonal buyers who reliably show up for this type of campaign
- **`momentum_contribution_pct`** — customers driven by recent momentum are showing accelerating activity; they may be newly engaged or re-engaging after a lapse

Filtering decile 1 by dominant contribution type lets marketing teams craft different messages for each group rather than treating the top decile as monolithic.

---

## SQL Companion Patterns

The `sql/` directory contains two production-grade SQL patterns:

- **`suppression.sql`** — builds a customer exclusion list covering invalid addresses, deceased customers, corporate addresses, and internal employees; applied as an exclusion join in all audience queries
- **`reactivation.sql`** — targets dormant high-value customers (3+ orders, $300+ LTV, 13–24 months inactive) for a re-engagement campaign with test/holdout split for measuring incremental lift

These are written for a Snowflake-compatible SQL dialect and use generic table names. Adapt schema/table names to your environment.

---

## Background

This tool is modeled after a propensity scoring pipeline built for **Briarwood Goods Co.**, a small DTC ecommerce retailer selling outdoor lifestyle, home, and seasonal gift products. Founded in 2018 and headquartered in Asheville, NC, Briarwood operates a direct marketing program with roughly 55,000 active customers across email and direct mail channels. The company runs 15–20 campaigns per year — seasonal catalog drops, holiday windows, Mother's Day, and summer clearance — where the core planning question is always the same: who do you target, and with what message?

The pipeline scored the full customer file ahead of each campaign window, with model validation running after each send to track decile lift over time.

Briarwood Goods Co. is a fictional company. The portfolio version runs entirely on synthetic data — no real customer records are included or required.

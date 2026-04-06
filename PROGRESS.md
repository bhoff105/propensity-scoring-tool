# Propensity Scoring Tool — Progress & Roadmap

## Project Status: Near Complete

**Last updated:** 2026-04-05

---

## What Is Built

| Area | Status | Notes |
|---|---|---|
| Synthetic dataset (10k customers) | Done | Logistic regression-ready with realistic conversion rate |
| `src/preprocess.py` | Done | Winsorization at 2nd/98th pct, fit on train only |
| `src/model.py` | Done | Logistic regression, `class_weight="balanced"`, top-decile lift as primary metric |
| `src/score.py` | Done | Decile + subdecile ranking, three-bucket contribution breakdown |
| `app/streamlit_app.py` | Done | Four-tab Streamlit dashboard |
| `.streamlit/config.toml` | Done | Explicit light theme, amber primary color |
| `requirements.txt` | Done | Plotly, no matplotlib |

---

## Recent Changes (this session)

- **Subdecile display fix**: Removed `@st.cache_data` so the app always scores fresh. Verified `1_1 = highest-scoring customers` in underlying data.
- **Tab navigation restored**: Replaced `st.sidebar.radio()` page routing with `st.tabs()`. Sidebar now shows campaign context and glossary only.

---

## Open Item

**Class imbalance decision:** `class_weight="balanced"` is in use. This inflates predicted probabilities above the true 7.3% population rate, which is why the app uses actual `purchased_in_campaign` values for conversion rate display rather than raw predicted scores. If the model is ever recalibrated (e.g., using Platt scaling or isotonic regression), the charts would show calibrated probabilities and could drop the actuals merge.

---

## Suggestions for Next Improvements

These are framed from the perspective of a marketing or business stakeholder reviewing the tool — not a technical audience.

---

### 1. Add a "What Should I Do?" Summary at the Top

**The gap:** A senior stakeholder landing on the tool currently has to read through multiple sections before seeing a clear recommendation. The model is strong, but the so-what is buried.

**The improvement:** Add a one-paragraph executive brief at the very top of the Engagement Model page — something like: "For the Mother's Day window, send to 6,000 customers in Priority and High Reach tiers. These segments represent 73% of expected conversions at 60% of list cost. Suppress the bottom 4 deciles." This gives a time-pressed reader everything they need without scrolling.

---

### 2. Add a Budget Scenario Calculator

**The gap:** The current tool tells you what the deciles contain, but a marketer's real question is: "If I have budget for X sends, which customers should I pick?"

**The improvement:** A simple slider in the Campaign Scoring tab where the user enters a send volume (e.g., 3,000, 5,000, 8,000). The tool dynamically shows which tiers that budget covers, estimated conversion count at that volume, and expected revenue if average order value is provided. This makes the model actionable for a specific budget conversation — the kind of question that comes up in every campaign planning meeting.

---

### 3. Compare Two Strategies Side by Side

**The gap:** Right now there is no way for a marketer to see the difference between the old way (e.g., sending to everyone who purchased in the past year) and the model-driven approach.

**The improvement:** Add a comparison view in the Engagement Model tab showing two bars: "Traditional recency-only list" vs. "Propensity model list." Show the estimated conversion lift, list size difference, and estimated cost savings. This is the single clearest way to demonstrate model value to someone who has to justify the change to their team.

---

### 4. Make the Score Driver Mix More Actionable

**The gap:** The Score Driver Mix chart is visually clear, but a non-technical reader may not know what to do with the information.

**The improvement:** Below the chart, add a one-sentence recommendation for each tier. Example: "Priority Tier is driven by the Seasonal Return Signal — send Mother's Day creative by April 14. High Reach is driven by the Customer Health Score — brand value messaging will outperform urgency." This bridges the analytical output to a specific creative or timing decision without requiring the reader to interpret the chart themselves.

---

### 5. Add a Glossary Inline with the Data (Not Just in the Sidebar)

**The gap:** The sidebar glossary is useful, but a first-time viewer looking at the Campaign Scoring table may not know to open it.

**The improvement:** The column headers in the tier table already have hover tooltips for desktop users. Consider adding a collapsed "Column Guide" just above the table — one sentence per column, visible on first load — so the context is in-line with the data. Especially important for the Lift column, which is the most powerful metric but least intuitive for non-analysts.

---

### 6. Show What the Model Does Not Know

**The gap:** The model is strong at ranking customers, but a business user might over-rely on a single score without understanding its limits.

**The improvement:** Add a short "Model Limitations" section to the How It Works tab. Examples: the model does not account for gift-buying behavior that starts from a non-purchaser (new customer acquisition), it does not adjust for customers who have unsubscribed but not suppressed, and scores decay in accuracy after six months without retraining. This builds trust by being honest about scope, which is exactly what a client or senior stakeholder wants to see before signing off on a model.

---

### 7. Add a Tableau or Power BI Dashboard (Cross-Portfolio Priority)

**Why this matters:** Tableau and Power BI appear in the requirements for nearly every Solutions Engineer, Customer Success Engineer, and Technical Account Manager role targeting analytics or data-adjacent buyers. The Streamlit app demonstrates Python and model-building competence, but a BI dashboard signals the ability to deliver insights in the tool a client's marketing or finance team already uses — without requiring them to run code.

**For this project (Propensity Scoring Tool):** A Tableau or Power BI dashboard built on top of `sable_spring_2026_scored.csv` would show the same tier breakdown, lift chart, and driver mix but in a format that a VP of Marketing or non-technical stakeholder could open, filter, and share independently. The scored CSV is already the right shape for a BI connection — no additional data prep required.

**For the A/B test analyzer (if applicable):** A similar dashboard showing test results, confidence intervals, and revenue impact would complement the Python tool with a stakeholder-ready deliverable format.

**Suggested approach:** Build one dashboard per project. Publish to Tableau Public (free) for portfolio visibility. Include a link in the README for each project. This rounds out the portfolio by showing the full analytics workflow: data prep and modeling in Python, insight delivery in BI.

---

### 8. Show Campaign Results After Send (Outcome Tracking)

**The gap:** The current tool is a pre-campaign planning tool. It does not show what happened after the campaign ran.

**The improvement:** Add a fifth tab, "Campaign Results," that shows a simple before/after view: predicted conversion rate per tier vs. actual conversion rate. Include a lift validation table and a one-line verdict ("The model's decile 1 prediction of 21.5% actual conversion rate was within 2 points of the observed 19.8%"). This closes the loop and demonstrates model accountability — which is the single most valuable thing to show a client who is deciding whether to use a model for the next campaign.

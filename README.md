# 🏥 KSMC ICU Registry — Interactive Dashboard

A fully interactive, blue-themed **Streamlit** dashboard built on top of the
EDA in `EDA_Saleh.ipynb`. It turns the 79,000-row ICU registry into a
filterable clinical analytics tool.

## Run it

```bash
pip install streamlit plotly pandas numpy
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).
Make sure `ksmc_icu_registry_dataset.csv` sits next to `app.py`.

## What's inside

**Sidebar filters** (live, affect every chart): hospital, ICU type,
diagnosis group, gender, age range, admission year — plus a running
patient count for the current selection.

**KPI row:** patients · mortality % · 30-day readmission % · ventilated % ·
average length of stay · total cost.

**6 tabs:**
| Tab | Contents |
|-----|----------|
| 📊 Overview | Demographics & case-mix: ICU type, hospital, age, diagnosis donut, gender × nationality |
| 🩺 Clinical Outcomes | Discharge status, mortality by diagnosis, LOS by outcome, severity, outcomes by age group |
| 💰 Cost | Cost KPIs, cost by diagnosis & ICU type, LOS-vs-cost scatter |
| 📈 Trends | Monthly admissions, monthly calculated cost, yearly admissions (mirrors the notebook's temporal section) |
| 🔬 Explorer | Correlation heatmap + a build-your-own scatter (pick any X / Y / color) |
| 🗂️ Data | Filtered table preview + CSV download + numeric summary |

## Design
A single **"blue degrees"** palette (light sky → deep navy) drives every
chart, the gradient hero banner, KPI cards and sidebar, for one cohesive
look. Insight callouts under each section explain what the data shows.

## Key findings surfaced
- Overall mortality ≈ **10%**, flat (~9–11%) across diagnosis groups.
- Correlations between features are **near zero** — the registry is largely
  synthetic/random, so it's great for EDA practice but weak for prediction.
- Cost is driven mainly by **per-day rate**, not length of stay.
- No strong **seasonal** pattern in admissions or spend.

"""
====================================================================
  KSMC ICU Registry — Interactive Analytics Dashboard
====================================================================
A fully interactive Streamlit dashboard built on top of the EDA work
in `EDA_Saleh.ipynb`. It reproduces and extends the notebook's analysis
(data cleaning, distributions, correlation, temporal trends, cost
engineering) into a filterable, blue-themed clinical analytics tool.

Run with:   streamlit run app.py
====================================================================
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------
# 1. PAGE CONFIG  — wide layout, custom title, ICU icon
# --------------------------------------------------------------------
st.set_page_config(
    page_title="KSMC ICU Registry Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------
# 2. BLUE THEME  — a curated palette of blue degrees used everywhere
# --------------------------------------------------------------------
# Sequential blue scale (light -> deep navy) drives every chart so the
# whole app reads as a single cohesive "blue degrees" design language.
BLUE_SCALE = ["#e3f2fd", "#90caf9", "#42a5f5", "#1e88e5", "#1565c0", "#0d47a1"]
BLUE_DEEP = "#0d47a1"
BLUE_MID = "#1e88e5"
BLUE_LIGHT = "#42a5f5"
BLUE_SOFT = "#90caf9"
# A diverging blue<->amber scale for the correlation heatmap (keeps the
# blue identity while still distinguishing positive vs negative).
DIVERGING = ["#0d47a1", "#1976d2", "#90caf9", "#f5f5f5", "#ffcc80", "#ef6c00"]

# Inject custom CSS for the blue gradient header, metric cards and fonts.
st.markdown(
    """
    <style>
        /* App background: very subtle blue wash */
        .stApp { background-color: #f4f8fd; }

        /* Gradient hero banner */
        .hero {
            background: linear-gradient(120deg, #0d47a1 0%, #1565c0 45%, #1e88e5 100%);
            padding: 28px 34px; border-radius: 16px; color: white;
            box-shadow: 0 6px 22px rgba(13,71,161,0.28); margin-bottom: 8px;
        }
        .hero h1 { margin: 0; font-size: 34px; font-weight: 800; letter-spacing:.3px;}
        .hero p  { margin: 6px 0 0 0; font-size: 16px; opacity:.92; }

        /* KPI metric cards */
        div[data-testid="stMetric"] {
            background: white; border: 1px solid #d6e4f5;
            border-left: 6px solid #1e88e5; border-radius: 12px;
            padding: 16px 18px; box-shadow: 0 2px 10px rgba(21,101,192,0.06);
        }
        div[data-testid="stMetricValue"] { color: #0d47a1; font-weight: 800; }
        div[data-testid="stMetricLabel"] { color: #1565c0; font-weight: 600; }

        /* Section headers */
        h2, h3 { color: #0d47a1 !important; }

        /* Sidebar tint */
        section[data-testid="stSidebar"] { background-color: #e8f1fc; }

        /* Insight callout box */
        .insight {
            background: #e3f2fd; border-left: 5px solid #1e88e5;
            border-radius: 8px; padding: 12px 16px; margin: 6px 0 18px 0;
            color: #0d3c75; font-size: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# 3. DATA LOADING + CLEANING
#    Mirrors the notebook: parse dates, engineer month/year, total cost.
#    Cached so filtering stays instant on the 79k-row registry.
# --------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "ksmc_icu_registry_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- Parse dates (notebook: pd.to_datetime on Admission_Date) ---
    df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], errors="coerce")
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], errors="coerce")

    # --- Feature engineering (notebook section "Features engineering") ---
    df["Admission_Month"] = df["Admission_Date"].dt.month
    df["Admission_Year"] = df["Admission_Date"].dt.year
    df["Month_Name"] = df["Admission_Date"].dt.strftime("%b")
    # Calculated cost = LOS * cost-per-day, exactly as in the notebook.
    df["Calculated_Total_Cost"] = df["Length_of_Stay_Days"] * df["Cost_Per_Day_SAR"]

    # --- Helpful binary flags for rate calculations ---
    df["Is_Deceased"] = (df["Discharge_Status"] == "Deceased").astype(int)
    df["Is_Readmitted"] = (df["Readmission_30days"] == "Yes").astype(int)
    df["Is_Ventilated"] = (df["Ventilator_Used"] == "Yes").astype(int)

    # --- Age bands for demographic breakdowns ---
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[-1, 17, 39, 64, 200],
        labels=["Pediatric (0-17)", "Adult (18-39)", "Middle (40-64)", "Senior (65+)"],
    )
    return df


df = load_data()

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --------------------------------------------------------------------
# 4. SIDEBAR FILTERS  — everything below reacts live to these controls.
# --------------------------------------------------------------------
st.sidebar.markdown("## 🔎 Filters")
st.sidebar.caption("Slice the registry — every chart updates instantly.")

hospitals = st.sidebar.multiselect(
    "🏥 Hospital", sorted(df["Hospital_Name"].unique()),
    default=sorted(df["Hospital_Name"].unique()),
)
icu_types = st.sidebar.multiselect(
    "🛏️ ICU Type", sorted(df["ICU_Type"].unique()),
    default=sorted(df["ICU_Type"].unique()),
)
diagnoses = st.sidebar.multiselect(
    "🩺 Diagnosis Group", sorted(df["Primary_Diagnosis_Group"].unique()),
    default=sorted(df["Primary_Diagnosis_Group"].unique()),
)
genders = st.sidebar.multiselect(
    "⚧ Gender", sorted(df["Gender"].unique()),
    default=sorted(df["Gender"].unique()),
)
age_range = st.sidebar.slider(
    "🎂 Age range", int(df["Age"].min()), int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max())),
)
years = st.sidebar.multiselect(
    "📅 Admission Year", sorted(df["Admission_Year"].dropna().unique()),
    default=sorted(df["Admission_Year"].dropna().unique()),
)

# Apply all filters into a single working frame `f`.
f = df[
    df["Hospital_Name"].isin(hospitals)
    & df["ICU_Type"].isin(icu_types)
    & df["Primary_Diagnosis_Group"].isin(diagnoses)
    & df["Gender"].isin(genders)
    & df["Age"].between(age_range[0], age_range[1])
    & df["Admission_Year"].isin(years)
]

st.sidebar.markdown("---")
st.sidebar.metric("Patients in selection", f"{len(f):,}")
st.sidebar.caption("KSMC ICU Registry • 79,000 admissions • 2018–2024")

# --------------------------------------------------------------------
# 5. HERO HEADER
# --------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏥 KSMC ICU Registry — Analytics Dashboard</h1>
        <p>Interactive exploration of 79,000 ICU admissions across four major Saudi medical cities ·
        outcomes, utilization, cost &amp; temporal trends.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard against an empty selection so the rest of the app never crashes.
if f.empty:
    st.warning("No patients match the current filters. Try widening your selection.")
    st.stop()

# --------------------------------------------------------------------
# 6. KPI ROW  — headline metrics for the current selection.
# --------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("👥 Patients", f"{len(f):,}")
c2.metric("⚰️ Mortality", f"{f['Is_Deceased'].mean()*100:.1f}%")
c3.metric("🔁 30-day Readmit", f"{f['Is_Readmitted'].mean()*100:.1f}%")
c4.metric("🫁 Ventilated", f"{f['Is_Ventilated'].mean()*100:.1f}%")
c5.metric("📆 Avg Stay", f"{f['Length_of_Stay_Days'].mean():.1f} d")
c6.metric("💰 Total Cost", f"{f['Total_Cost_SAR'].sum()/1e6:,.1f}M SAR")

st.markdown("")  # small spacer

# --------------------------------------------------------------------
# 7. TABBED LAYOUT  — keeps the dashboard organised & uncluttered.
# --------------------------------------------------------------------
tab_overview, tab_outcomes, tab_cost, tab_trends, tab_explore, tab_data = st.tabs(
    ["📊 Overview", "🩺 Clinical Outcomes", "💰 Cost", "📈 Trends",
     "🔬 Explorer", "🗂️ Data"]
)

# ====================================================================
#  TAB 1 — OVERVIEW : demographics & case-mix
# ====================================================================
with tab_overview:
    st.subheader("Patient Demographics & Case-Mix")

    left, right = st.columns(2)

    # --- Admissions by ICU type (bar) ---
    with left:
        icu_counts = f["ICU_Type"].value_counts().reset_index()
        icu_counts.columns = ["ICU_Type", "Patients"]
        fig = px.bar(
            icu_counts, x="ICU_Type", y="Patients", color="Patients",
            color_continuous_scale=BLUE_SCALE, text="Patients",
            title="Admissions by ICU Type",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          xaxis_title="", yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    # --- Patients per hospital (horizontal bar) ---
    with right:
        hosp = f["Hospital_Name"].value_counts().reset_index()
        hosp.columns = ["Hospital", "Patients"]
        fig = px.bar(
            hosp, x="Patients", y="Hospital", orientation="h", color="Patients",
            color_continuous_scale=BLUE_SCALE, text="Patients",
            title="Patients by Hospital",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          xaxis_title="Patients", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    # --- Age distribution (histogram) ---
    with left2:
        fig = px.histogram(
            f, x="Age", nbins=30, color_discrete_sequence=[BLUE_MID],
            title="Age Distribution",
        )
        fig.update_layout(plot_bgcolor="white", bargap=0.05,
                          yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    # --- Diagnosis mix (donut) ---
    with right2:
        dx = f["Primary_Diagnosis_Group"].value_counts().reset_index()
        dx.columns = ["Diagnosis", "Patients"]
        fig = px.pie(
            dx, names="Diagnosis", values="Patients", hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues[::-1],
            title="Primary Diagnosis Mix",
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig, use_container_width=True)

    # --- Gender x Nationality split (grouped bar) ---
    gn = f.groupby(["Gender", "Nationality"]).size().reset_index(name="Patients")
    fig = px.bar(
        gn, x="Gender", y="Patients", color="Nationality", barmode="group",
        color_discrete_sequence=[BLUE_DEEP, BLUE_LIGHT],
        title="Gender × Nationality",
    )
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> The registry is broadly '
        'balanced across the four ICU types and hospitals, with a near-uniform '
        'adult age spread. Use the sidebar to isolate a single hospital or '
        'diagnosis and watch the case-mix shift in real time.</div>',
        unsafe_allow_html=True,
    )

# ====================================================================
#  TAB 2 — CLINICAL OUTCOMES : mortality, readmission, severity
# ====================================================================
with tab_outcomes:
    st.subheader("Clinical Outcomes & Risk")

    left, right = st.columns(2)

    # --- Discharge status breakdown ---
    with left:
        ds = f["Discharge_Status"].value_counts().reset_index()
        ds.columns = ["Status", "Patients"]
        fig = px.bar(
            ds, x="Status", y="Patients", color="Status", text="Patients",
            color_discrete_sequence=[BLUE_DEEP, BLUE_LIGHT, BLUE_SOFT],
            title="Discharge Status",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(plot_bgcolor="white", showlegend=False,
                          xaxis_title="", yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    # --- Mortality rate by diagnosis ---
    with right:
        mort = (f.groupby("Primary_Diagnosis_Group")["Is_Deceased"]
                .mean().mul(100).round(2).reset_index()
                .sort_values("Is_Deceased", ascending=True))
        fig = px.bar(
            mort, x="Is_Deceased", y="Primary_Diagnosis_Group",
            orientation="h", color="Is_Deceased",
            color_continuous_scale=BLUE_SCALE,
            title="Mortality Rate by Diagnosis (%)",
        )
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          xaxis_title="Mortality %", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # --- Length of stay by discharge status (box) ---
    fig = px.box(
        f, x="Discharge_Status", y="Length_of_Stay_Days",
        color="Discharge_Status",
        color_discrete_sequence=[BLUE_DEEP, BLUE_MID, BLUE_SOFT],
        title="Length of Stay by Discharge Status",
    )
    fig.update_layout(plot_bgcolor="white", showlegend=False,
                      xaxis_title="", yaxis_title="Length of Stay (days)")
    st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    # --- Severity score distribution ---
    with left2:
        fig = px.histogram(
            f, x="Severity_Score", nbins=24,
            color_discrete_sequence=[BLUE_MID], title="Severity Score Distribution",
        )
        fig.update_layout(plot_bgcolor="white", bargap=0.05, yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

    # --- Mortality & readmission by age group ---
    with right2:
        ag = (f.groupby("Age_Group", observed=True)[["Is_Deceased", "Is_Readmitted"]]
              .mean().mul(100).round(2).reset_index())
        fig = go.Figure()
        fig.add_bar(x=ag["Age_Group"], y=ag["Is_Deceased"],
                    name="Mortality %", marker_color=BLUE_DEEP)
        fig.add_bar(x=ag["Age_Group"], y=ag["Is_Readmitted"],
                    name="Readmission %", marker_color=BLUE_LIGHT)
        fig.update_layout(barmode="group", plot_bgcolor="white",
                          title="Mortality & Readmission by Age Group",
                          yaxis_title="Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> Overall mortality sits near '
        '<b>10%</b> and is remarkably flat across diagnosis groups '
        '(≈9–11%) — a hallmark of this synthetic registry. Outcomes show only '
        'weak association with severity or age, so predictive modelling on this '
        'data should expect limited signal (see the Explorer tab).</div>',
        unsafe_allow_html=True,
    )

# ====================================================================
#  TAB 3 — COST : spend drivers & efficiency
# ====================================================================
with tab_cost:
    st.subheader("Cost Analysis")

    k1, k2, k3 = st.columns(3)
    k1.metric("💵 Total Spend", f"{f['Total_Cost_SAR'].sum()/1e6:,.1f}M SAR")
    k2.metric("🧾 Avg Cost / Patient", f"{f['Total_Cost_SAR'].mean():,.0f} SAR")
    k3.metric("📅 Avg Cost / Day", f"{f['Cost_Per_Day_SAR'].mean():,.0f} SAR")

    left, right = st.columns(2)

    # --- Avg cost per patient by diagnosis ---
    with left:
        cost_dx = (f.groupby("Primary_Diagnosis_Group")["Total_Cost_SAR"]
                   .mean().round(0).reset_index()
                   .sort_values("Total_Cost_SAR"))
        fig = px.bar(
            cost_dx, x="Total_Cost_SAR", y="Primary_Diagnosis_Group",
            orientation="h", color="Total_Cost_SAR",
            color_continuous_scale=BLUE_SCALE,
            title="Avg Cost per Patient by Diagnosis",
        )
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          xaxis_title="Avg Cost (SAR)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # --- Avg cost by ICU type ---
    with right:
        cost_icu = (f.groupby("ICU_Type")["Total_Cost_SAR"]
                    .mean().round(0).reset_index()
                    .sort_values("Total_Cost_SAR"))
        fig = px.bar(
            cost_icu, x="ICU_Type", y="Total_Cost_SAR", color="Total_Cost_SAR",
            color_continuous_scale=BLUE_SCALE,
            title="Avg Cost per Patient by ICU Type",
        )
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          xaxis_title="", yaxis_title="Avg Cost (SAR)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Length of stay vs total cost (scatter) ---
    sample = f.sample(min(4000, len(f)), random_state=1)  # sample for speed
    fig = px.scatter(
        sample, x="Length_of_Stay_Days", y="Total_Cost_SAR",
        color="Severity_Score", color_continuous_scale=BLUE_SCALE,
        opacity=0.55, title="Length of Stay vs Total Cost (colored by severity)",
    )
    fig.update_layout(plot_bgcolor="white",
                      xaxis_title="Length of Stay (days)",
                      yaxis_title="Total Cost (SAR)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> Average cost per patient is '
        'fairly uniform across diagnosis and ICU type (~50K SAR), echoing the '
        'flat outcome pattern. Total cost is driven mostly by the per-day rate '
        'rather than length of stay — a useful cue for budgeting that '
        'efficiency gains come from daily-rate management.</div>',
        unsafe_allow_html=True,
    )

# ====================================================================
#  TAB 4 — TRENDS : temporal patterns (notebook's monthly analysis)
# ====================================================================
with tab_trends:
    st.subheader("Temporal Trends")

    # --- Admissions by month (notebook: barplot of monthly counts) ---
    by_month = (f.groupby("Month_Name").size()
                .reindex(MONTH_ORDER).reset_index())
    by_month.columns = ["Month", "Patients"]
    fig = px.bar(
        by_month, x="Month", y="Patients", color="Patients",
        color_continuous_scale=BLUE_SCALE, text="Patients",
        title="ICU Admissions by Month",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                      yaxis_title="Patients")
    st.plotly_chart(fig, use_container_width=True)

    # --- Total calculated cost by month (notebook lineplot) ---
    cost_month = (f.groupby("Month_Name")["Calculated_Total_Cost"].sum()
                  .reindex(MONTH_ORDER).reset_index())
    cost_month.columns = ["Month", "Cost"]
    fig = px.line(
        cost_month, x="Month", y="Cost", markers=True,
        title="Total Calculated Cost Across Months",
    )
    fig.update_traces(line_color=BLUE_DEEP, marker_color=BLUE_MID, line_width=3)
    fig.update_layout(plot_bgcolor="white", yaxis_title="Total Cost (SAR)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Yearly admissions trend ---
    by_year = f.groupby("Admission_Year").size().reset_index(name="Patients")
    fig = px.area(
        by_year, x="Admission_Year", y="Patients",
        title="Admissions by Year",
    )
    fig.update_traces(line_color=BLUE_DEEP,
                      fillcolor="rgba(30,136,229,0.25)")
    fig.update_layout(plot_bgcolor="white", xaxis_title="Year",
                      yaxis_title="Patients")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> Admissions and monthly spend '
        'are evenly distributed across the calendar — there is no strong '
        'seasonal surge in this registry. The data spans 2018 through early '
        '2024 (note the partial final year when reading the yearly area chart).</div>',
        unsafe_allow_html=True,
    )

# ====================================================================
#  TAB 5 — EXPLORER : correlation + build-your-own chart
# ====================================================================
with tab_explore:
    st.subheader("Correlation & Custom Explorer")

    # --- Correlation heatmap (notebook: sns.heatmap of numeric corr) ---
    num_cols = ["Age", "Length_of_Stay_Days", "Severity_Score",
                "Comorbidities_Count", "Number_of_Lab_Tests",
                "Number_of_Radiology_Procedures", "Total_Cost_SAR",
                "Cost_Per_Day_SAR", "Is_Deceased", "Is_Readmitted",
                "Is_Ventilated"]
    corr = f[num_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale=DIVERGING, zmin=-1, zmax=1,
        title="Correlation Matrix (numeric features)",
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> Correlations are uniformly '
        'near zero — the strongest relationships barely exceed |0.01|. This '
        'confirms the dataset is largely randomly generated, so any ML model '
        'will struggle to find genuine predictive structure. It is excellent '
        'for practicing the EDA / dashboarding workflow, less so for inference.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 🛠️ Build your own chart")
    cc1, cc2, cc3 = st.columns(3)
    x_axis = cc1.selectbox("X axis", num_cols, index=1)
    y_axis = cc2.selectbox("Y axis", num_cols, index=6)
    color_by = cc3.selectbox(
        "Color by",
        ["None", "ICU_Type", "Primary_Diagnosis_Group", "Discharge_Status",
         "Gender", "Severity_Score"],
        index=3,
    )

    plot_df = f.sample(min(4000, len(f)), random_state=2)
    fig = px.scatter(
        plot_df, x=x_axis, y=y_axis,
        color=None if color_by == "None" else color_by,
        color_continuous_scale=BLUE_SCALE,
        color_discrete_sequence=px.colors.sequential.Blues[::-1],
        opacity=0.6, title=f"{y_axis} vs {x_axis}",
    )
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# ====================================================================
#  TAB 6 — DATA : preview + download of the filtered slice
# ====================================================================
with tab_data:
    st.subheader("Filtered Data")
    st.caption(f"Showing the current selection — {len(f):,} rows.")

    show_cols = ["Patient_ID", "Age", "Gender", "Nationality", "Hospital_Name",
                 "ICU_Type", "Primary_Diagnosis_Group", "Severity_Score",
                 "Length_of_Stay_Days", "Total_Cost_SAR", "Discharge_Status",
                 "Readmission_30days", "Admission_Date"]
    st.dataframe(f[show_cols].head(500), use_container_width=True, height=430)

    # Download button for the filtered slice.
    csv = f[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data (CSV)", csv,
        file_name="ksmc_icu_filtered.csv", mime="text/csv",
    )

    with st.expander("📋 Numeric summary (describe)"):
        st.dataframe(f.describe().round(2), use_container_width=True)

# --------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit & Plotly · Blue-degrees theme · "
    "Based on EDA_Saleh.ipynb · KSMC ICU Registry (synthetic, 79k admissions)."
)

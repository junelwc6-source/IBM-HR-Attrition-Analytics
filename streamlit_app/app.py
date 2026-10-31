
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==================================================
# 1. PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="HR Attrition & Predictive Risk",
    page_icon="📊",
    layout="wide"
)

NAVY = "#20366B"
TEAL = "#189A9E"
GREY = "#B8B8B8"
RED = "#E56B58"
GOLD = "#F1B83D"

st.markdown("""
<style>
.stApp {
    background-color: #F7F9FC;
}
h1, h2, h3 {
    color: #20366B;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E7EF;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 2. LOAD EXISTING DATASETS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    historical = pd.read_csv(
        BASE_DIR / "HR_Attrition_Historical.csv"
    )
    risk = pd.read_csv(
        BASE_DIR / "HR_Attrition_Predictive_Risk.csv"
    )
    return historical, risk

try:
    historical, risk = load_data()
except Exception as error:
    st.error(f"Unable to load CSV files: {error}")
    st.stop()

historical["Attrition_Flag"] = (
    historical["Attrition"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

risk["Risk_Level"] = (
    risk["Risk_Level"]
    .astype(str)
    .str.strip()
    .str.title()
)

risk["Predicted_Probability"] = pd.to_numeric(
    risk["Predicted_Probability"],
    errors="coerce"
)

# ==================================================
# 3. SHARED FUNCTIONS
# ==================================================

def attrition_chart(
    data,
    column,
    title,
    horizontal=False,
    benchmark=None
):
    summary = (
        data.groupby(column, observed=True)["Attrition_Flag"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary["Rate"] = summary["mean"] * 100

    if horizontal:
        summary = summary.sort_values("Rate")

        fig = px.bar(
            summary,
            x="Rate",
            y=column,
            orientation="h",
            text="Rate",
            color_discrete_sequence=[GREY]
        )

        fig.update_traces(
            texttemplate="%{x:.1f}%",
            textposition="outside",
            cliponaxis=False
        )

        fig.update_xaxes(
            title="Attrition Rate (%)",
            range=[
                0,
                max(45, summary["Rate"].max() * 1.25)
            ]
        )
        fig.update_yaxes(title=None)

        if benchmark is not None:
            fig.add_vline(
                x=benchmark,
                line_color=RED,
                line_width=2
            )

    else:
        fig = px.bar(
            summary,
            x=column,
            y="Rate",
            text="Rate",
            color_discrete_sequence=[GREY]
        )

        fig.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(
            title="Attrition Rate (%)",
            range=[
                0,
                max(35, summary["Rate"].max() * 1.25)
            ]
        )
        fig.update_xaxes(title=None)

        if benchmark is not None:
            fig.add_hline(
                y=benchmark,
                line_color=RED,
                line_width=2
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=320 if not horizontal else 400,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=30, t=55, b=35)
    )

    st.plotly_chart(fig, use_container_width=True)


def department_donut(data, department):
    subset = data[data["Department"] == department]

    stayed = int(
        (subset["Attrition_Flag"] == 0).sum()
    )
    left = int(
        (subset["Attrition_Flag"] == 1).sum()
    )

    fig = go.Figure(
        go.Pie(
            labels=["Stayed", "Left"],
            values=[stayed, left],
            hole=0.65,
            sort=False,
            marker_colors=[TEAL, GREY],
            textinfo="percent"
        )
    )

    fig.update_layout(
        title=dict(
            text=department,
            x=0.5,
            font=dict(size=14)
        ),
        height=225,
        showlegend=False,
        paper_bgcolor="white",
        margin=dict(l=5, r=5, t=45, b=5),
        annotations=[
            dict(
                text=f"{len(subset):,}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=19, color=NAVY)
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)


def risk_donut(data):
    counts = (
        data["Risk_Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.65,
            sort=False,
            marker_colors=[TEAL, GOLD, RED],
            textinfo="label+percent"
        )
    )

    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        margin=dict(l=5, r=5, t=15, b=5),
        showlegend=False,
        annotations=[
            dict(
                text=f"{len(data):,}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=22, color=NAVY)
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)


def employee_table(data, show_all=False):
    if show_all:
        table = data.copy()
    else:
        columns = [
            "EmployeeNumber",
            "Department",
            "Risk_Level",
            "Predicted_Probability"
        ]
        table = data[columns].copy()

    table["Predicted_Probability"] = (
        table["Predicted_Probability"]
        .map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    )

    table = table.rename(columns={
        "EmployeeNumber": "Employee Number",
        "Risk_Level": "Risk Level",
        "Predicted_Probability": "Predicted Probability"
    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 4. NAVIGATION
# ==================================================

st.sidebar.title("📊 HR Analytics Portal")
st.sidebar.caption("IBM HR Employee Attrition Capstone")

page = st.sidebar.radio(
    "Navigation",
    [
        "Integrated Dashboard",
        "Employee Risk Explorer",
        "Model Insights"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Synthetic IBM HR dataset. Existing predictions "
    "are used without retraining the model."
)

# ==================================================
# PAGE 1: INTEGRATED DASHBOARD
# ==================================================

if page == "Integrated Dashboard":

    st.title(
        "EMPLOYEE ATTRITION & PREDICTIVE RISK DASHBOARD"
    )

    st.caption(
        "Historical attrition analysis and predictive "
        "risk insights"
    )

    # ----------------------------------------------
    # HISTORICAL FILTERS
    # ----------------------------------------------

    f1, f2, f3 = st.columns(3)

    with f1:
        departments = st.multiselect(
            "Department",
            sorted(historical["Department"].dropna().unique()),
            default=sorted(
                historical["Department"].dropna().unique()
            )
        )

    with f2:
        genders = st.multiselect(
            "Gender",
            sorted(historical["Gender"].dropna().unique()),
            default=sorted(
                historical["Gender"].dropna().unique()
            )
        )

    with f3:
        age_groups = st.multiselect(
            "Age Group",
            list(historical["AgeGroup"].dropna().unique()),
            default=list(
                historical["AgeGroup"].dropna().unique()
            )
        )

    filtered = historical[
        historical["Department"].isin(departments)
        & historical["Gender"].isin(genders)
        & historical["AgeGroup"].isin(age_groups)
    ].copy()

    if filtered.empty:
        st.warning(
            "No historical employees match the selected filters."
        )
        st.stop()

    # ----------------------------------------------
    # FIVE KPI CARDS
    # ----------------------------------------------

    total = len(filtered)
    left = int(filtered["Attrition_Flag"].sum())
    attrition_rate = left / total * 100
    average_income = filtered["MonthlyIncome"].mean()

    # Match the historical selection to employee IDs
    # for the high-risk KPI.
    selected_risk = risk[
        risk["EmployeeNumber"].isin(
            filtered["EmployeeNumber"]
        )
    ]

    high_risk_count = int(
        selected_risk["Risk_Level"].eq("High").sum()
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total Employees", f"{total:,}")
    k2.metric("Employees Who Left", f"{left:,}")
    k3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    k4.metric(
        "Average Monthly Income",
        f"${average_income:,.0f}"
    )
    k5.metric(
        "High-Risk Employees",
        f"{high_risk_count:,}"
    )

    st.divider()

    # ----------------------------------------------
    # THREE DASHBOARD SECTIONS
    # ----------------------------------------------

    left_panel, middle_panel, right_panel = st.columns(
        [1.1, 3.0, 1.8],
        gap="medium"
    )

    # DEPARTMENT OVERVIEW
    with left_panel:

        st.subheader("DEPARTMENT OVERVIEW")

        for dept in [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]:
            if dept in departments:
                department_donut(filtered, dept)

        st.caption("Teal = Stayed | Grey = Left")

    # HISTORICAL ATTRITION ANALYSIS
    with middle_panel:

        st.subheader("HISTORICAL ATTRITION ANALYSIS")

        attrition_chart(
            filtered,
            "JobRole",
            "Attrition Rate by Job Role",
            horizontal=True,
            benchmark=attrition_rate
        )

        c1, c2 = st.columns(2)

        with c1:
            attrition_chart(
                filtered,
                "OverTime",
                "Attrition Rate by Overtime"
            )

        with c2:
            working = filtered.copy()

            working["WorkingYearsBand"] = pd.cut(
                working["TotalWorkingYears"],
                bins=[-1, 5, 10, 20, float("inf")],
                labels=[
                    "0–5",
                    "6–10",
                    "11–20",
                    "21+"
                ]
            )

            attrition_chart(
                working,
                "WorkingYearsBand",
                "Attrition Rate by Total Working Years",
                benchmark=attrition_rate
            )

        c3, c4, c5 = st.columns(3)

        with c3:
            attrition_chart(
                filtered,
                "PromotionGroup",
                "Attrition Rate by Promotion Band",
                benchmark=attrition_rate
            )

        with c4:
            attrition_chart(
                filtered,
                "IncomeGroup",
                "Attrition Rate by Income Group",
                benchmark=attrition_rate
            )

        with c5:
            attrition_chart(
                filtered,
                "SatisfactionGroup",
                "Attrition Rate by Job Satisfaction",
                benchmark=attrition_rate
            )

        st.caption(
            f"Reference line: {attrition_rate:.1f}% "
            "for the selected historical population."
        )

    # PREDICTIVE RISK ANALYSIS
    with right_panel:

        st.subheader("PREDICTIVE RISK ANALYSIS")

        st.caption(
            "Existing predictions for employees who stayed"
        )

        risk_departments = sorted(
            risk["Department"].dropna().unique()
        )

        selected_risk_depts = st.multiselect(
            "Risk Department",
            risk_departments,
            default=risk_departments
        )

        selected_levels = st.multiselect(
            "Risk Level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

        employee_search = st.text_input(
            "Employee Number"
        )

        risk_filtered = risk[
            risk["Department"].isin(selected_risk_depts)
            & risk["Risk_Level"].isin(selected_levels)
        ].copy()

        if employee_search.strip():
            risk_filtered = risk_filtered[
                risk_filtered["EmployeeNumber"]
                .astype(str)
                .str.contains(
                    employee_search.strip(),
                    regex=False
                )
            ]

        risk_donut(risk_filtered)

        st.subheader("Employee Risk Details")

        employee_table(risk_filtered)

        st.caption(
            "Predictive filters are separate from "
            "historical filters."
        )

# ==================================================
# PAGE 2: EMPLOYEE RISK EXPLORER
# ==================================================

elif page == "Employee Risk Explorer":

    st.title("EMPLOYEE RISK EXPLORER")

    st.write(
        "Explore the existing model predictions "
        "for employees who stayed."
    )

    counts = risk["Risk_Level"].value_counts()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Employees Assessed", f"{len(risk):,}")
    k2.metric("Low Risk", f"{counts.get('Low', 0):,}")
    k3.metric("Medium Risk", f"{counts.get('Medium', 0):,}")
    k4.metric("High Risk", f"{counts.get('High', 0):,}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        selected_levels = st.multiselect(
            "Risk Category",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

    with c2:
        departments = sorted(
            risk["Department"].dropna().unique()
        )

        selected_departments = st.multiselect(
            "Department",
            departments,
            default=departments
        )

    employee_search = st.text_input(
        "Search Employee Number"
    )

    filtered_risk = risk[
        risk["Risk_Level"].isin(selected_levels)
        & risk["Department"].isin(selected_departments)
    ].copy()

    if employee_search.strip():
        filtered_risk = filtered_risk[
            filtered_risk["EmployeeNumber"]
            .astype(str)
            .str.contains(
                employee_search.strip(),
                regex=False
            )
        ]

    st.subheader("Risk Distribution")
    risk_donut(filtered_risk)

    st.subheader("Employee Risk Details")
    employee_table(filtered_risk)

    with st.expander(
        "View all employee attributes and available predictors"
    ):
        employee_table(filtered_risk, show_all=True)

    st.download_button(
        "Download Filtered Results",
        data=filtered_risk.to_csv(index=False),
        file_name="Filtered_Employee_Risk.csv",
        mime="text/csv"
    )

    st.caption(
        "Risk classifications are estimates, not confirmed "
        "future departures. They must not be used for "
        "automatic employment decisions."
    )

# ==================================================
# PAGE 3: MODEL INSIGHTS
# ==================================================

elif page == "Model Insights":

    st.title("PREDICTIVE MODEL INSIGHTS")

    st.subheader(
        "Selected Model: Calibrated Logistic Regression"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test ROC AUC", "0.795")
    c2.metric("Test Recall", "72.3%")
    c3.metric("Test Precision", "37.0%")
    c4.metric("Calibrated CV Brier Score", "0.098")

    st.divider()

    st.subheader("Cross-Validation Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "ROC AUC": [0.831, 0.808],
        "Recall": [0.753, 0.463],
        "F1 Score": [0.495, 0.496]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Why Logistic Regression?")

    st.write(
        "Logistic Regression achieved stronger ROC AUC "
        "and recall during cross-validation. It was "
        "selected to support earlier identification "
        "of employees who may require retention attention."
    )

    st.subheader("Held-Out Model Evaluation")

    st.write(
        "The held-out test dataset contained 294 employees, "
        "including 47 who left. The model correctly identified "
        "34 departures, with 58 false alerts."
    )

    st.subheader("Model Predictors")

    st.write(
        "The final model uses 22 predictors. "
        "Important predictors identified in the analysis "
        "include OverTime, StockOptionLevel, "
        "YearsSinceLastPromotion, NumCompaniesWorked "
        "and TotalWorkingYears."
    )

    st.caption(
        "Predictor importance reflects model behaviour, "
        "not proof of causation. The original notebook "
        "remains the reference for the exact importance scores."
    )

    st.subheader("HR Recommendations")

    st.markdown("""
1. Prioritise retention reviews in higher-attrition groups.
2. Investigate workload and overtime patterns.
3. Strengthen onboarding and career development.
4. Combine model estimates with feedback and HR judgement.
5. Pilot targeted interventions and monitor outcomes.
""")

    st.warning(
        "The dataset is synthetic. Further validation, "
        "fairness assessment and real-world testing "
        "would be required before operational deployment."
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "IBM HR Employee Attrition Analytics | "
    "Capstone Project | Python · Machine Learning · "
    "Power BI · Streamlit"
)

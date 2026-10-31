
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="HR Attrition & Predictive Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
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

.block-container {
    max-width: 100%;
    padding: 0.6rem 1rem 1rem;
}

h1 {
    font-size: 1.45rem !important;
    padding: 0 !important;
}

h2, h3 {
    font-size: 1rem !important;
    padding: 0.1rem 0 !important;
}

[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E7EF;
    border-radius: 8px;
    padding: 0.5rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.3rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD EXISTING DATA
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
# REUSABLE CHART FUNCTIONS
# ==================================================

def attrition_chart(
    data,
    column,
    title,
    order=None,
    horizontal=False,
    benchmark=None,
    height=210
):
    summary = (
        data.groupby(column, observed=True)["Attrition_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="Rate")
    )

    if order is not None:
        summary[column] = pd.Categorical(
            summary[column],
            categories=order,
            ordered=True
        )
        summary = summary.sort_values(column)

    elif horizontal:
        summary = summary.sort_values("Rate")

    fig = go.Figure()

    if horizontal:
        fig.add_trace(go.Bar(
            x=summary["Rate"],
            y=summary[column].astype(str),
            orientation="h",
            marker_color=GREY,
            text=[
                f"{value:.1f}%"
                for value in summary["Rate"]
            ],
            textposition="outside",
            cliponaxis=False
        ))

        fig.update_xaxes(
            range=[
                0,
                max(
                    45,
                    summary["Rate"].max() * 1.25
                )
            ]
        )

        if benchmark is not None:
            fig.add_vline(
                x=benchmark,
                line_color=RED,
                line_width=1.5
            )

    else:
        fig.add_trace(go.Bar(
            x=summary[column].astype(str),
            y=summary["Rate"],
            marker_color=GREY,
            text=[
                f"{value:.1f}%"
                for value in summary["Rate"]
            ],
            textposition="outside",
            cliponaxis=False
        ))

        fig.update_yaxes(
            range=[
                0,
                max(
                    35,
                    summary["Rate"].max() * 1.2
                )
            ]
        )

        if benchmark is not None:
            fig.add_hline(
                y=benchmark,
                line_color=RED,
                line_width=1.5
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=12),
            x=0.02
        ),
        height=height,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(
            l=10,
            r=20,
            t=35,
            b=15
        ),
        font=dict(size=10),
        bargap=0.25
    )

    fig.update_xaxes(
        title=None,
        tickfont=dict(size=9),
        automargin=True
    )

    fig.update_yaxes(
        title=None,
        tickfont=dict(size=9),
        automargin=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )


def donut_chart(
    values,
    labels,
    colors,
    centre_value,
    title="",
    height=175
):
    fig = go.Figure(go.Pie(
        values=values,
        labels=labels,
        hole=0.65,
        sort=False,
        marker_colors=colors,
        textinfo="percent",
        textfont=dict(size=10)
    ))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=11)
        ),
        height=height,
        showlegend=False,
        paper_bgcolor="white",
        margin=dict(
            l=0,
            r=0,
            t=28,
            b=0
        ),
        annotations=[
            dict(
                text=f"{centre_value:,}",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    size=16,
                    color=NAVY
                )
            )
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )


def employee_table(data, show_all=False):

    if show_all:
        table = data.copy()
    else:
        table = data[
            [
                "EmployeeNumber",
                "Department",
                "Risk_Level",
                "Predicted_Probability"
            ]
        ].copy()

    table["Predicted_Probability"] = (
        table["Predicted_Probability"]
        .map(
            lambda x: f"{x:.1%}"
            if pd.notna(x) else ""
        )
    )

    table = table.rename(columns={
        "EmployeeNumber": "Employee Number",
        "Risk_Level": "Risk Level",
        "Predicted_Probability": "Predicted Probability"
    })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=260 if not show_all else 380
    )


# ==================================================
# NAVIGATION
# ==================================================

st.sidebar.title("📊 HR Analytics Portal")

page = st.sidebar.radio(
    "Navigation",
    [
        "Integrated Dashboard",
        "Employee Risk Explorer",
        "Model Insights"
    ]
)

st.sidebar.caption(
    "IBM HR Employee Attrition Capstone"
)

# ==================================================
# PAGE 1: INTEGRATED DASHBOARD
# ==================================================

if page == "Integrated Dashboard":

    st.title(
        "EMPLOYEE ATTRITION & PREDICTIVE RISK DASHBOARD"
    )

    # HISTORICAL FILTERS

    f1, f2, f3 = st.columns(3)

    with f1:
        departments = sorted(
            historical["Department"]
            .dropna()
            .unique()
        )

        selected_departments = st.multiselect(
            "Department",
            departments,
            default=departments
        )

    with f2:
        genders = sorted(
            historical["Gender"]
            .dropna()
            .unique()
        )

        selected_genders = st.multiselect(
            "Gender",
            genders,
            default=genders
        )

    with f3:
        age_order = [
            "Under 30",
            "30-39",
            "40-49",
            "50 and above"
        ]

        selected_ages = st.multiselect(
            "Age Group",
            age_order,
            default=age_order
        )

    filtered = historical[
        historical["Department"].isin(
            selected_departments
        )
        & historical["Gender"].isin(
            selected_genders
        )
        & historical["AgeGroup"].isin(
            selected_ages
        )
    ].copy()

    if filtered.empty:
        st.warning(
            "No employees match the selected filters."
        )
        st.stop()

    # FIVE KPI CARDS

    total = len(filtered)
    left = int(filtered["Attrition_Flag"].sum())
    rate = left / total * 100

    average_income = (
        filtered["MonthlyIncome"].mean()
    )

    selected_risk = risk[
        risk["EmployeeNumber"].isin(
            filtered["EmployeeNumber"]
        )
    ]

    high_risk = int(
        selected_risk["Risk_Level"]
        .eq("High")
        .sum()
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Total Employees",
        f"{total:,}"
    )

    k2.metric(
        "Employees Who Left",
        f"{left:,}"
    )

    k3.metric(
        "Attrition Rate",
        f"{rate:.1f}%"
    )

    k4.metric(
        "Average Monthly Income",
        f"${average_income:,.0f}"
    )

    k5.metric(
        "High-Risk Employees",
        f"{high_risk:,}"
    )

    st.divider()

    # THREE MAIN SECTIONS

    department_panel, history_panel, risk_panel = (
        st.columns(
            [1.05, 3.25, 1.9],
            gap="small"
        )
    )

    # ----------------------------------------------
    # DEPARTMENT ATTRITION OVERVIEW
    # ----------------------------------------------

    with department_panel:

        st.subheader("DEPARTMENT OVERVIEW")

        for department in [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]:

            if department not in selected_departments:
                continue

            subset = filtered[
                filtered["Department"] == department
            ]

            stayed = int(
                subset["Attrition_Flag"]
                .eq(0)
                .sum()
            )

            departed = int(
                subset["Attrition_Flag"]
                .eq(1)
                .sum()
            )

            donut_chart(
                [stayed, departed],
                ["Stayed", "Left"],
                [TEAL, GREY],
                len(subset),
                department,
                height=175
            )

        st.caption(
            "Teal = Stayed | Grey = Left"
        )

    # ----------------------------------------------
    # HISTORICAL ATTRITION ANALYSIS
    # ----------------------------------------------

    with history_panel:

        st.subheader(
            "HISTORICAL ATTRITION ANALYSIS"
        )

        # JOB ROLE

        attrition_chart(
            filtered,
            "JobRole",
            "Attrition Rate by Job Role",
            horizontal=True,
            benchmark=rate,
            height=250
        )

        row1_left, row1_right = st.columns(2)

        # OVERTIME

        with row1_left:

            attrition_chart(
                filtered,
                "OverTime",
                "Attrition Rate by Overtime",
                order=["Yes", "No"],
                height=205
            )

        # TOTAL WORKING YEARS

        with row1_right:

            working = filtered.copy()

            working["WorkingYearsBand"] = pd.cut(
                working["TotalWorkingYears"],
                bins=[
                    -1,
                    5,
                    10,
                    20,
                    float("inf")
                ],
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
                order=[
                    "0–5",
                    "6–10",
                    "11–20",
                    "21+"
                ],
                benchmark=rate,
                height=205
            )

        row2_left, row2_middle, row2_right = (
            st.columns(3)
        )

        # PROMOTION BAND
        # Based on the Power BI screenshot

        with row2_left:

            promotion = filtered.copy()

            promotion["PromotionBand_PBI"] = pd.cut(
                promotion["YearsSinceLastPromotion"],
                bins=[
                    -1,
                    1,
                    3,
                    7,
                    float("inf")
                ],
                labels=[
                    "0–1 years",
                    "2–3 years",
                    "4–7 years",
                    "8+ years"
                ]
            )

            attrition_chart(
                promotion,
                "PromotionBand_PBI",
                "Attrition Rate by Promotion Band",
                order=[
                    "0–1 years",
                    "2–3 years",
                    "4–7 years",
                    "8+ years"
                ],
                benchmark=rate,
                height=210
            )

        # INCOME GROUP

        with row2_middle:

            attrition_chart(
                filtered,
                "IncomeGroup",
                "Attrition Rate by Income Group",
                order=[
                    "Low",
                    "Lower-Middle",
                    "Upper-Middle",
                    "High"
                ],
                benchmark=rate,
                height=210
            )

        # JOB SATISFACTION

        with row2_right:

            attrition_chart(
                filtered,
                "SatisfactionGroup",
                "Attrition Rate by Job Satisfaction",
                order=[
                    "Low",
                    "High",
                    "Medium",
                    "Very High"
                ],
                benchmark=rate,
                height=210
            )

        st.caption(
            f"Red reference line: {rate:.1f}% "
            "historical attrition."
        )

    # ----------------------------------------------
    # PREDICTIVE RISK ANALYSIS
    # ----------------------------------------------

    with risk_panel:

        st.subheader(
            "PREDICTIVE RISK ANALYSIS"
        )

        risk_departments = sorted(
            risk["Department"]
            .dropna()
            .unique()
        )

        selected_risk_departments = st.multiselect(
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
            "Employee Number",
            placeholder="Search employee ID"
        )

        filtered_risk = risk[
            risk["Department"].isin(
                selected_risk_departments
            )
            & risk["Risk_Level"].isin(
                selected_levels
            )
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

        counts = (
            filtered_risk["Risk_Level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High"],
                fill_value=0
            )
        )

        donut_chart(
            counts.values,
            counts.index,
            [TEAL, GOLD, RED],
            len(filtered_risk),
            height=215
        )

        st.subheader(
            "Employee Risk Details"
        )

        employee_table(filtered_risk)

        st.caption(
            "Predictive filters operate separately "
            "from historical filters."
        )

# ==================================================
# PAGE 2: EMPLOYEE RISK EXPLORER
# ==================================================

elif page == "Employee Risk Explorer":

    st.title("EMPLOYEE RISK EXPLORER")

    st.write(
        "Explore the existing predictions for "
        "employees who stayed."
    )

    counts = risk["Risk_Level"].value_counts()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Employees Assessed",
        f"{len(risk):,}"
    )

    k2.metric(
        "Low Risk",
        f"{counts.get('Low', 0):,}"
    )

    k3.metric(
        "Medium Risk",
        f"{counts.get('Medium', 0):,}"
    )

    k4.metric(
        "High Risk",
        f"{counts.get('High', 0):,}"
    )

    f1, f2 = st.columns(2)

    with f1:
        selected_levels = st.multiselect(
            "Risk Category",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

    with f2:
        departments = sorted(
            risk["Department"]
            .dropna()
            .unique()
        )

        selected_departments = st.multiselect(
            "Department",
            departments,
            default=departments
        )

    search = st.text_input(
        "Search Employee Number"
    )

    filtered_risk = risk[
        risk["Risk_Level"].isin(
            selected_levels
        )
        & risk["Department"].isin(
            selected_departments
        )
    ].copy()

    if search.strip():

        filtered_risk = filtered_risk[
            filtered_risk["EmployeeNumber"]
            .astype(str)
            .str.contains(
                search.strip(),
                regex=False
            )
        ]

    counts = (
        filtered_risk["Risk_Level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0
        )
    )

    st.subheader("Risk Distribution")

    donut_chart(
        counts.values,
        counts.index,
        [TEAL, GOLD, RED],
        len(filtered_risk),
        height=260
    )

    st.subheader("Employee Risk Details")

    employee_table(filtered_risk)

    with st.expander(
        "View all employee attributes and predictors"
    ):
        employee_table(
            filtered_risk,
            show_all=True
        )

    st.download_button(
        "Download Filtered Results",
        data=filtered_risk.to_csv(index=False),
        file_name="Filtered_Employee_Risk.csv",
        mime="text/csv"
    )

    st.caption(
        "Risk classifications are estimates, "
        "not confirmed future departures."
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
    c4.metric("Calibrated CV Brier", "0.098")

    st.subheader(
        "Cross-Validation Model Comparison"
    )

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
        "Logistic Regression achieved stronger "
        "cross-validation ROC AUC and recall, "
        "supporting earlier identification of "
        "employees who may require retention attention."
    )

    st.subheader("Held-Out Model Evaluation")

    st.write(
        "Of 294 test employees, 47 had left. "
        "The model correctly identified 34 departures, "
        "with 58 false alerts."
    )

    st.subheader("Model Predictors")

    st.write(
        "The existing model uses 22 predictors. "
        "Important predictors include OverTime, "
        "StockOptionLevel, YearsSinceLastPromotion, "
        "NumCompaniesWorked and TotalWorkingYears."
    )

    st.caption(
        "Predictor importance describes model behaviour, "
        "not causation. Refer to the original notebook "
        "for exact importance values."
    )

    st.subheader("HR Recommendations")

    st.markdown("""
1. Prioritise retention reviews in higher-attrition groups.
2. Investigate overtime and workload patterns.
3. Strengthen onboarding and career development.
4. Combine model estimates with HR feedback and judgement.
5. Pilot interventions and monitor retention outcomes.
""")

    st.warning(
        "The dataset is synthetic. Real-world deployment "
        "requires validation, fairness assessment and "
        "appropriate governance."
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

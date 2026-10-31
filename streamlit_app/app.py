
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Analytics Portal",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F7F9FC;
    }
    h1, h2, h3 {
        color: #17365D;
    }
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E3E8EF;
        border-radius: 12px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
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
    st.error(f"Unable to load project datasets: {error}")
    st.stop()

# Prepare historical attrition indicator
historical["Attrition_Flag"] = (
    historical["Attrition"].astype(str).str.strip().str.lower()
    .map({"yes": 1, "no": 0})
)

if historical["Attrition_Flag"].isna().any():
    st.error("Unexpected values in the Attrition column.")
    st.stop()

risk["Predicted_Probability"] = pd.to_numeric(
    risk["Predicted_Probability"], errors="coerce"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📊 HR Analytics Portal")
st.sidebar.caption("IBM HR Employee Attrition Capstone")

page = st.sidebar.radio(
    "Navigation",
    [
        "Attrition Overview",
        "Employee Risk Explorer",
        "Model Insights"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Data source: IBM HR synthetic employee dataset. "
    "Predictions are imported from the completed capstone model."
)

# --------------------------------------------------
# PAGE 1: ATTRITION OVERVIEW
# --------------------------------------------------
if page == "Attrition Overview":

    st.title("HR Employee Attrition Overview")
    st.write(
        "Explore historical attrition patterns and identify "
        "employee groups requiring further investigation."
    )

    departments = sorted(
        historical["Department"].dropna().unique()
    )

    selected_departments = st.multiselect(
        "Filter by department",
        departments,
        default=departments
    )

    filtered = historical[
        historical["Department"].isin(selected_departments)
    ].copy()

    total = len(filtered)
    left = int(filtered["Attrition_Flag"].sum())
    stayed = total - left
    attrition_rate = left / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Employees", f"{total:,}")
    c2.metric("Employees Left", f"{left:,}")
    c3.metric("Employees Stayed", f"{stayed:,}")
    c4.metric("Attrition Rate", f"{attrition_rate:.1f}%")

    st.divider()

    if filtered.empty:
        st.warning("Please select at least one department.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attrition by Department")

            dept = (
                filtered.groupby("Department")["Attrition_Flag"]
                .mean()
                .mul(100)
                .reset_index(name="Attrition Rate")
            )

            fig = px.bar(
                dept,
                x="Department",
                y="Attrition Rate",
                color="Department",
                text_auto=".1f",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Attrition by Overtime")

            overtime = (
                filtered.groupby("OverTime")["Attrition_Flag"]
                .mean()
                .mul(100)
                .reset_index(name="Attrition Rate")
            )

            fig = px.bar(
                overtime,
                x="OverTime",
                y="Attrition Rate",
                color="OverTime",
                text_auto=".1f"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Attrition by Job Role")

            job = (
                filtered.groupby("JobRole")["Attrition_Flag"]
                .agg(["mean", "size"])
                .reset_index()
            )
            job["Attrition Rate"] = job["mean"] * 100
            job = job.sort_values("Attrition Rate")

            fig = px.bar(
                job,
                x="Attrition Rate",
                y="JobRole",
                orientation="h",
                hover_data=["size"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Attrition by Age Group")

            age = (
                filtered.groupby(
                    "AgeGroup", observed=True
                )["Attrition_Flag"]
                .mean()
                .mul(100)
                .reset_index(name="Attrition Rate")
            )

            fig = px.bar(
                age,
                x="AgeGroup",
                y="Attrition Rate",
                text_auto=".1f"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Historical relationships are descriptive and "
            "should not be interpreted as causal."
        )

# --------------------------------------------------
# PAGE 2: EMPLOYEE RISK EXPLORER
# --------------------------------------------------
elif page == "Employee Risk Explorer":

    st.title("Employee Predictive Risk Explorer")
    st.write(
        "Explore the existing calibrated model predictions "
        "for employees who stayed in the historical dataset."
    )

    st.info(
        "These predictions were generated by the original "
        "capstone model. Streamlit does not retrain the model."
    )

    risk_order = ["Low", "Medium", "High"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Employees Assessed", f"{len(risk):,}")

    for column, level in zip(
        [c2, c3, c4], risk_order
    ):
        count = (
            risk["Risk_Level"].astype(str).str.strip()
            .str.lower().eq(level.lower()).sum()
        )
        column.metric(f"{level} Risk", f"{count:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        selected_risks = st.multiselect(
            "Risk category",
            risk_order,
            default=risk_order
        )

    with col2:
        departments = sorted(
            risk["Department"].dropna().unique()
        )
        selected_depts = st.multiselect(
            "Department",
            departments,
            default=departments
        )

    filtered_risk = risk[
        risk["Risk_Level"].astype(str).str.strip()
        .str.lower().isin(
            [r.lower() for r in selected_risks]
        )
        & risk["Department"].isin(selected_depts)
    ].copy()

    st.subheader("Risk Distribution")

    if filtered_risk.empty:
        st.warning("No employees match the selected filters.")
    else:
        counts = (
            filtered_risk["Risk_Level"]
            .value_counts()
            .rename_axis("Risk Level")
            .reset_index(name="Employees")
        )

        fig = px.pie(
            counts,
            names="Risk Level",
            values="Employees",
            hole=0.5,
            color="Risk Level",
            color_discrete_map={
                "Low": "#2A9D8F",
                "Medium": "#E9C46A",
                "High": "#E76F51"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Employee Risk Details")

        search_id = st.text_input(
            "Search Employee Number",
            placeholder="Enter an employee number"
        )

        if search_id:
            filtered_risk = filtered_risk[
                filtered_risk["EmployeeNumber"]
                .astype(str)
                .str.contains(search_id, regex=False)
            ]

        preferred_columns = [
            "EmployeeNumber",
            "Department",
            "JobRole",
            "Age",
            "Gender",
            "OverTime",
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsSinceLastPromotion",
            "Predicted_Probability",
            "Risk_Level"
        ]

        visible_columns = [
            col for col in preferred_columns
            if col in filtered_risk.columns
        ]

        st.dataframe(
            filtered_risk[visible_columns],
            use_container_width=True,
            hide_index=True
        )

        with st.expander(
            "View all available employee attributes"
        ):
            st.dataframe(
                filtered_risk,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download Filtered Results",
            data=filtered_risk.to_csv(index=False),
            file_name="Filtered_Employee_Risk.csv",
            mime="text/csv"
        )

    st.caption(
        "Risk levels are model estimates, not confirmed future "
        "departures. Predictions must not be used as automatic "
        "employment decisions."
    )

# --------------------------------------------------
# PAGE 3: MODEL INSIGHTS
# --------------------------------------------------
elif page == "Model Insights":

    st.title("Predictive Model Insights")
    st.write(
        "Understanding model selection, performance "
        "and implications for HR decision-making."
    )

    st.subheader("Selected Model: Logistic Regression")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test ROC AUC", "0.795")
    c2.metric("Test Recall", "72.3%")
    c3.metric("Test Precision", "37.0%")
    c4.metric("CV Brier Score", "0.098")

    st.subheader("Cross-Validation Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
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
        "and recall during cross-validation. It was selected "
        "to support earlier identification of employees "
        "who may require retention attention."
    )

    st.subheader("Model Evaluation")

    st.write(
        "On the held-out test dataset of 294 employees, "
        "the model correctly identified 34 of 47 employees "
        "who left, with 58 false alerts."
    )

    st.subheader("Important Predictors")

    st.write(
        "The final model uses 22 predictors. "
        "Important predictors identified in the analysis "
        "include overtime, stock option level, years since "
        "last promotion, number of companies worked "
        "and total working years."
    )

    st.caption(
        "Predictor importance reflects model behaviour, "
        "not proof of a causal relationship."
    )

    st.subheader("HR Recommendations")

    st.markdown("""
    1. Prioritise retention reviews in higher-attrition groups.
    2. Investigate workload and overtime patterns.
    3. Strengthen onboarding, mentoring and career development.
    4. Combine predicted risk with employee feedback and HR judgement.
    5. Pilot targeted interventions and monitor retention outcomes.
    """)

    st.warning(
        "The dataset is synthetic. Further validation, fairness "
        "assessment and real-world testing would be required "
        "before operational deployment."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "IBM HR Employee Attrition Analytics | "
    "Capstone Project | Python · Machine Learning · Streamlit"
)

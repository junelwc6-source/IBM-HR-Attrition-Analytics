
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
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
.stApp {background-color: #F7F9FC;}
.block-container {
    max-width: 100%;
    padding: 0.6rem 1rem 1rem;
}
h1 {font-size: 1.45rem !important; padding: 0 !important;}
h2, h3 {font-size: 1rem !important; padding: 0.1rem 0 !important;}
[data-testid="stVerticalBlock"] {gap: 0.35rem;}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E7EF;
    border-radius: 8px;
    padding: 0.5rem;
}
[data-testid="stMetricValue"] {font-size: 1.3rem;}
[data-testid="stMetricLabel"] {font-size: 0.75rem;}
</style>
""", unsafe_allow_html=True)

# ==================================================
# FILE PATHS AND DATA LOADING
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    historical = pd.read_csv(
        BASE_DIR / "HR_Attrition_Historical.csv"
    )
    risk = pd.read_csv(
        BASE_DIR / "HR_Attrition_Predictive_Risk.csv"
    )
    return historical, risk

@st.cache_resource
def load_prediction_model():
    model = joblib.load(
        MODEL_DIR / "calibrated_attrition_model.joblib"
    )
    features = joblib.load(
        MODEL_DIR / "model_features.joblib"
    )
    thresholds = joblib.load(
        MODEL_DIR / "risk_thresholds.joblib"
    )
    return model, features, thresholds

try:
    historical, risk = load_data()
except Exception as error:
    st.error(f"Unable to load CSV files: {error}")
    st.stop()

historical["Attrition_Flag"] = (
    historical["Attrition"]
    .astype(str).str.strip().str.lower()
    .map({"yes": 1, "no": 0})
)

risk["Risk_Level"] = (
    risk["Risk_Level"].astype(str).str.strip().str.title()
)

risk["Predicted_Probability"] = pd.to_numeric(
    risk["Predicted_Probability"], errors="coerce"
)

for data in (historical, risk):
    data["EmployeeNumber"] = pd.to_numeric(
        data["EmployeeNumber"], errors="coerce"
    )

# ==================================================
# PREDICTOR DEFINITIONS
# ==================================================

PREDICTOR_COLUMNS = [
    "OverTime",
    "StockOptionLevel",
    "YearsSinceLastPromotion",
    "NumCompaniesWorked",
    "TotalWorkingYears",
    "BusinessTravel",
    "JobSatisfaction",
    "JobInvolvement",
    "JobRole",
    "TrainingTimesLastYear",
    "EnvironmentSatisfaction",
    "YearsInCurrentRole",
    "MonthlyIncome",
    "YearsAtCompany",
    "RelationshipSatisfaction",
    "JobLevel",
    "YearsWithCurrManager",
    "DistanceFromHome",
    "PercentSalaryHike",
    "PerformanceRating",
    "WorkLifeBalance"
]

POWERBI_RANKING = [
    ("OverTime", 100),
    ("StockOptionLevel", 23),
    ("YearsSinceLastPromotion", 21),
    ("NumCompaniesWorked", 21),
    ("TotalWorkingYears", 20),
    ("BusinessTravel", 17),
    ("JobSatisfaction", 13),
    ("JobInvolvement", 13),
    ("JobRole", 11),
    ("TrainingTimesLastYear", 10),
    ("EnvironmentSatisfaction", 9),
    ("YearsInCurrentRole", 9),
    ("MonthlyIncome", 7),
    ("YearsAtCompany", 6),
    ("RelationshipSatisfaction", 5),
    ("JobLevel", 4),
    ("YearsWithCurrManager", 4),
    ("DistanceFromHome", 3),
    ("PercentSalaryHike", 2),
    ("PerformanceRating", 0),
    ("WorkLifeBalance", 0)
]

EXCLUDED_DISPLAY_COLUMNS = {
    "Attrition", "Attrition_Flag",
    "Predicted_Probability", "Risk_Level",
    "EmployeeNumber", "Department",
    "AgeGroup", "Age_Group", "IncomeGroup",
    "SatisfactionGroup", "PromotionBand",
    "WorkingYearsBand", "WorkingYearsGroup",
    "Prediction", "Predicted_Attrition"
}

# ==================================================
# REUSABLE CHART FUNCTIONS
# ==================================================

def attrition_chart(
    data, column, title, order=None,
    horizontal=False, benchmark=None, height=210
):
    summary = (
        data.groupby(column, observed=True)["Attrition_Flag"]
        .mean().mul(100).reset_index(name="Rate")
    )

    if order is not None:
        summary[column] = pd.Categorical(
            summary[column], categories=order, ordered=True
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
            text=[f"{v:.1f}%" for v in summary["Rate"]],
            textposition="outside",
            cliponaxis=False
        ))
        maximum = summary["Rate"].max() if not summary.empty else 0
        fig.update_xaxes(range=[0, max(45, maximum * 1.25)])

        if benchmark is not None:
            fig.add_vline(
                x=benchmark, line_color=RED, line_width=1.5
            )
    else:
        fig.add_trace(go.Bar(
            x=summary[column].astype(str),
            y=summary["Rate"],
            marker_color=GREY,
            text=[f"{v:.1f}%" for v in summary["Rate"]],
            textposition="outside",
            cliponaxis=False
        ))
        maximum = summary["Rate"].max() if not summary.empty else 0
        fig.update_yaxes(range=[0, max(35, maximum * 1.2)])

        if benchmark is not None:
            fig.add_hline(
                y=benchmark, line_color=RED, line_width=1.5
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12), x=0.02),
        height=height,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=20, t=35, b=15),
        font=dict(size=10),
        bargap=0.25
    )

    fig.update_xaxes(
        title=None, tickfont=dict(size=9), automargin=True
    )
    fig.update_yaxes(
        title=None, tickfont=dict(size=9), automargin=True
    )

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False}
    )


def donut_chart(
    values, labels, colors, centre_value,
    title="", height=175
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
            text=title, x=0.5, font=dict(size=11)
        ),
        height=height,
        showlegend=False,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=28, b=0),
        annotations=[dict(
            text=f"{centre_value:,}",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=NAVY)
        )]
    )

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False}
    )


def employee_table(data):
    columns = [
        "EmployeeNumber", "Department",
        "Risk_Level", "Predicted_Probability"
    ]

    table = data[columns].copy()

    table["Predicted_Probability"] = (
        table["Predicted_Probability"]
        .map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    )

    table["EmployeeNumber"] = (
        table["EmployeeNumber"].astype("Int64")
    )

    table = table.rename(columns={
        "EmployeeNumber": "Employee Number",
        "Risk_Level": "Risk Level",
        "Predicted_Probability": "Predicted Probability"
    })

    st.dataframe(
        table, use_container_width=True,
        hide_index=True, height=260
    )


def full_employee_risk_table(data):
    primary_columns = [
        "EmployeeNumber", "Department",
        "Risk_Level", "Predicted_Probability"
    ]

    available_predictors = [
        c for c in PREDICTOR_COLUMNS if c in data.columns
    ]

    additional_columns = [
        c for c in data.columns
        if c not in primary_columns
        and c not in available_predictors
        and c not in EXCLUDED_DISPLAY_COLUMNS
    ]

    display_columns = (
        primary_columns
        + available_predictors
        + additional_columns
    )

    table = data[display_columns].copy()

    table["Predicted_Probability"] = (
        table["Predicted_Probability"]
        .map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    )

    table["EmployeeNumber"] = (
        table["EmployeeNumber"].astype("Int64")
    )

    rename_columns = {
        "EmployeeNumber": "Employee Number",
        "Risk_Level": "Risk Level",
        "Predicted_Probability": "Predicted Probability",
        "OverTime": "Overtime",
        "StockOptionLevel": "Stock Option Level",
        "YearsSinceLastPromotion": "Years Since Last Promotion",
        "NumCompaniesWorked": "Num Companies Worked",
        "TotalWorkingYears": "Total Working Years",
        "BusinessTravel": "Business Travel",
        "JobSatisfaction": "Job Satisfaction",
        "JobInvolvement": "Job Involvement",
        "TrainingTimesLastYear": "Training Times Last Year",
        "EnvironmentSatisfaction": "Environment Satisfaction",
        "YearsInCurrentRole": "Years In Current Role",
        "MonthlyIncome": "Monthly Income",
        "YearsAtCompany": "Years At Company",
        "RelationshipSatisfaction": "Relationship Satisfaction",
        "JobLevel": "Job Level",
        "YearsWithCurrManager": "Years With Curr Manager",
        "DistanceFromHome": "Distance From Home",
        "PercentSalaryHike": "Percent Salary Hike",
        "PerformanceRating": "Performance Rating",
        "WorkLifeBalance": "Work Life Balance"
    }

    table = table.rename(columns=rename_columns)

    def colour_risk(value):
        if value == "High":
            return "background-color: #F5B8B8; color: #6A1010;"
        if value == "Medium":
            return "background-color: #F9E59A; color: #654D00;"
        if value == "Low":
            return "background-color: #BDEBC1; color: #175B20;"
        return ""

    styled = table.style.map(
        colour_risk, subset=["Risk Level"]
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=540,
        row_height=30
    )

    st.caption(
        "Scroll horizontally to view additional employee "
        "predictors. High = Red, Medium = Yellow, Low = Green."
    )


def predictor_ranking_chart():
    ranking = pd.DataFrame(
        POWERBI_RANKING,
        columns=["Predictor", "Ranking Score"]
    )

    ranking["Display Name"] = (
        ranking["Predictor"]
        .str.replace(r"(?<!^)(?=[A-Z])", " ", regex=True)
    )

    ranking = ranking.iloc[::-1]

    fig = go.Figure(go.Bar(
        x=ranking["Ranking Score"],
        y=ranking["Display Name"],
        orientation="h",
        marker_color="#EF8B61",
        text=ranking["Ranking Score"],
        textposition="outside",
        cliponaxis=False
    ))

    fig.update_layout(
        title=dict(
            text="Ranking by Predictor",
            font=dict(size=16), x=0
        ),
        height=680,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=15, r=45, t=45, b=35),
        font=dict(size=11),
        bargap=0.35
    )

    fig.update_xaxes(
        range=[0, 110],
        title="Power BI ranking value",
        showgrid=True,
        gridcolor="#E8EBF0"
    )
    fig.update_yaxes(title=None, automargin=True)

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False}
    )


# ==================================================
# LIVE PREDICTION FUNCTION
# ==================================================

def live_prediction_section():
    st.divider()
    st.subheader("LIVE ATTRITION PREDICTION")

    st.caption(
        "Enter employee attributes to generate a new "
        "attrition risk prediction using the trained "
        "calibrated Logistic Regression model."
    )

    try:
        model, features, thresholds = load_prediction_model()
    except Exception as error:
        st.error(f"Unable to load trained model: {error}")
        return

    missing = [c for c in features if c not in risk.columns]

    if missing:
        st.error(
            "Missing predictor columns: " + ", ".join(missing)
        )
        return

    if len(features) != 22:
        st.warning(
            f"Expected 22 predictors; found {len(features)}."
        )

    categorical_features = {
        "Department",
        "JobRole",
        "BusinessTravel",
        "OverTime"
    }

    st.info(
        "The form starts with typical values from the "
        "historical employee dataset. Change any fields "
        "to explore a different employee profile."
    )

    with st.form("live_attrition_prediction"):
        input_values = {}
        cols = st.columns(3)

        for i, feature in enumerate(features):
            with cols[i % 3]:
                source = risk[feature]

                if feature in categorical_features:
                    options = sorted(
                        source.dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if not options:
                        st.error(
                            f"No available values for {feature}"
                        )
                        return

                    input_values[feature] = st.selectbox(
                        feature,
                        options,
                        key=f"predict_{feature}"
                    )

                else:
                    numeric = pd.to_numeric(
                        source, errors="coerce"
                    ).dropna()

                    if numeric.empty:
                        st.error(
                            f"No numeric values for {feature}"
                        )
                        return

                    minimum = float(numeric.min())
                    maximum = float(numeric.max())
                    median = float(numeric.median())

                    # Preserve integer-valued predictor types
                    if (numeric % 1 == 0).all():
                        input_values[feature] = st.number_input(
                            feature,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(round(median)),
                            step=1,
                            key=f"predict_{feature}"
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=median,
                            key=f"predict_{feature}"
                        )

        submitted = st.form_submit_button(
            "Predict Attrition Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        new_employee = pd.DataFrame(
            [input_values], columns=features
        )

        try:
            probability = float(
                model.predict_proba(new_employee)[0, 1]
            )

            low = float(thresholds["low_threshold"])
            high = float(thresholds["high_threshold"])

            if probability <= low:
                level = "Low"
            elif probability <= high:
                level = "Medium"
            else:
                level = "High"

            st.success("Prediction generated successfully!")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted Attrition Probability",
                f"{probability:.1%}"
            )

            c2.metric(
                "Predicted Risk Level",
                level
            )

            c3.metric(
                "Model Predictors Used",
                len(features)
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            st.caption(
                f"Risk thresholds: Low up to {low:.2%}; "
                f"Medium up to {high:.2%}; "
                "High above the upper threshold."
            )

            st.warning(
                "This is a statistical model estimate, "
                "not a confirmed future departure. "
                "It should not be used as the sole basis "
                "for employment decisions."
            )

        except Exception as error:
            st.error(
                f"Unable to generate prediction: {error}"
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

    st.caption(
        "Department, Gender and Age Group filters apply "
        "to both historical and predictive analysis."
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        departments = sorted(
            historical["Department"].dropna().unique()
        )
        selected_departments = st.multiselect(
            "Department",
            departments,
            default=departments,
            key="main_department"
        )

    with f2:
        genders = sorted(
            historical["Gender"].dropna().unique()
        )
        selected_genders = st.multiselect(
            "Gender",
            genders,
            default=genders,
            key="main_gender"
        )

    with f3:
        age_order = [
            "Under 30", "30-39", "40-49", "50 and above"
        ]
        available_ages = [
            a for a in age_order
            if a in historical["AgeGroup"].unique()
        ]
        selected_ages = st.multiselect(
            "Age Group",
            available_ages,
            default=available_ages,
            key="main_age"
        )

    filtered = historical[
        historical["Department"].isin(selected_departments)
        & historical["Gender"].isin(selected_genders)
        & historical["AgeGroup"].isin(selected_ages)
    ].copy()

    if filtered.empty:
        st.warning("No employees match the selected filters.")
        st.stop()

    selected_employee_ids = (
        filtered["EmployeeNumber"].dropna().unique()
    )

    shared_risk = risk[
        risk["EmployeeNumber"].isin(selected_employee_ids)
    ].copy()

    total = len(filtered)
    left = int(filtered["Attrition_Flag"].sum())
    rate = left / total * 100
    average_income = filtered["MonthlyIncome"].mean()
    high_risk = int(
        shared_risk["Risk_Level"].eq("High").sum()
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total Employees", f"{total:,}")
    k2.metric("Employees Who Left", f"{left:,}")
    k3.metric("Attrition Rate", f"{rate:.1f}%")
    k4.metric(
        "Average Monthly Income",
        f"${average_income:,.0f}"
    )
    k5.metric("High-Risk Employees", f"{high_risk:,}")

    st.divider()

    department_panel, history_panel, risk_panel = (
        st.columns([1.05, 3.25, 1.9], gap="small")
    )

    # DEPARTMENT OVERVIEW

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

            if subset.empty:
                continue

            stayed = int(
                subset["Attrition_Flag"].eq(0).sum()
            )
            departed = int(
                subset["Attrition_Flag"].eq(1).sum()
            )

            donut_chart(
                [stayed, departed],
                ["Stayed", "Left"],
                [TEAL, GREY],
                len(subset),
                department,
                height=175
            )

        st.caption("Teal = Stayed | Grey = Left")

    # HISTORICAL ATTRITION ANALYSIS

    with history_panel:
        st.subheader("HISTORICAL ATTRITION ANALYSIS")

        attrition_chart(
            filtered,
            "JobRole",
            "Attrition Rate by Job Role",
            horizontal=True,
            benchmark=rate,
            height=250
        )

        row1_left, row1_right = st.columns(2)

        with row1_left:
            attrition_chart(
                filtered,
                "OverTime",
                "Attrition Rate by Overtime",
                order=["Yes", "No"],
                height=205
            )

        with row1_right:
            working = filtered.copy()
            working["WorkingYearsBand"] = pd.cut(
                working["TotalWorkingYears"],
                bins=[-1, 5, 10, 20, float("inf")],
                labels=["0–5", "6–10", "11–20", "21+"]
            )

            attrition_chart(
                working,
                "WorkingYearsBand",
                "Attrition Rate by Total Working Years",
                order=["0–5", "6–10", "11–20", "21+"],
                benchmark=rate,
                height=205
            )

        row2_left, row2_middle, row2_right = (
            st.columns(3)
        )

        with row2_left:
            promotion = filtered.copy()
            promotion["PromotionBand_PBI"] = pd.cut(
                promotion["YearsSinceLastPromotion"],
                bins=[-1, 1, 3, 7, float("inf")],
                labels=[
                    "0–1 years", "2–3 years",
                    "4–7 years", "8+ years"
                ]
            )

            attrition_chart(
                promotion,
                "PromotionBand_PBI",
                "Attrition Rate by Promotion Band",
                order=[
                    "0–1 years", "2–3 years",
                    "4–7 years", "8+ years"
                ],
                benchmark=rate,
                height=210
            )

        with row2_middle:
            attrition_chart(
                filtered,
                "IncomeGroup",
                "Attrition Rate by Income Group",
                order=[
                    "Low", "Lower-Middle",
                    "Upper-Middle", "High"
                ],
                benchmark=rate,
                height=210
            )

        with row2_right:
            attrition_chart(
                filtered,
                "SatisfactionGroup",
                "Attrition Rate by Job Satisfaction",
                order=[
                    "Low", "High", "Medium", "Very High"
                ],
                benchmark=rate,
                height=210
            )

        st.caption(
            f"Red reference line: {rate:.1f}% "
            "historical attrition."
        )

    # PREDICTIVE RISK ANALYSIS

    with risk_panel:
        st.subheader("PREDICTIVE RISK ANALYSIS")

        st.caption(
            "Uses the Department, Gender and Age Group "
            "filters above."
        )

        selected_levels = st.multiselect(
            "Risk Level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="main_risk_level"
        )

        employee_search = st.text_input(
            "Employee Number",
            placeholder="Search employee ID",
            key="main_employee_search"
        )

        filtered_risk = shared_risk[
            shared_risk["Risk_Level"].isin(selected_levels)
        ].copy()

        if employee_search.strip():
            filtered_risk = filtered_risk[
                filtered_risk["EmployeeNumber"]
                .astype("Int64").astype(str)
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

        if filtered_risk.empty:
            st.info(
                "No predicted employees match "
                "the selected filters."
            )
        else:
            donut_chart(
                counts.values,
                counts.index,
                [TEAL, GOLD, RED],
                len(filtered_risk),
                height=215
            )

        st.subheader("Employee Risk Details")
        employee_table(filtered_risk)

        st.caption(
            "High-Risk Employees KPI reflects the top "
            "three filters. Risk Level and Employee Number "
            "further filter the predictive chart and table."
        )

# ==================================================
# PAGE 2: EMPLOYEE RISK EXPLORER
# ==================================================

elif page == "Employee Risk Explorer":

    st.title("EMPLOYEE RISK DETAILS")

    st.caption(
        "Explore employee-level predicted risk and "
        "the original model predictor fields."
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        departments = sorted(
            risk["Department"].dropna().unique()
        )

        selected_departments = st.multiselect(
            "Department",
            departments,
            default=departments,
            key="explorer_department"
        )

    with f2:
        employee_search = st.text_input(
            "Employee Number",
            placeholder="Search employee number",
            key="explorer_search"
        )

    with f3:
        selected_levels = st.multiselect(
            "Risk Level",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="explorer_risk"
        )

    filtered_risk = risk[
        risk["Department"].isin(selected_departments)
        & risk["Risk_Level"].isin(selected_levels)
    ].copy()

    if employee_search.strip():
        filtered_risk = filtered_risk[
            filtered_risk["EmployeeNumber"]
            .astype("Int64").astype(str)
            .str.contains(
                employee_search.strip(),
                regex=False
            )
        ]

    filtered_risk = filtered_risk.sort_values(
        "EmployeeNumber"
    )

    counts = filtered_risk["Risk_Level"].value_counts()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Employees Displayed",
        f"{len(filtered_risk):,}"
    )
    k2.metric("Low Risk", f"{counts.get('Low', 0):,}")
    k3.metric("Medium Risk", f"{counts.get('Medium', 0):,}")
    k4.metric("High Risk", f"{counts.get('High', 0):,}")

    st.divider()

    st.subheader(
        "Employee Risk Details and Predictors"
    )

    st.caption(
        "The table includes employee risk fields "
        "and available original predictors. Scroll "
        "horizontally to view all columns."
    )

    if filtered_risk.empty:
        st.info(
            "No employees match the selected filters."
        )
    else:
        full_employee_risk_table(filtered_risk)

    st.download_button(
        "Download Filtered Employee Risk Details",
        data=filtered_risk.to_csv(index=False),
        file_name="Filtered_Employee_Risk_Details.csv",
        mime="text/csv"
    )

    st.caption(
        "The historical employee risk classifications "
        "are model estimates, not confirmed future "
        "departures. Saved prediction values have "
        "not been recalculated."
    )

    # NEW: LIVE MODEL DEPLOYMENT

    live_prediction_section()

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
        "supporting earlier identification of employees "
        "who may require retention attention."
    )

    st.subheader("Held-Out Model Evaluation")

    st.write(
        "Of 294 test employees, 47 had left. "
        "The model correctly identified 34 departures, "
        "with 58 false alerts."
    )

    st.divider()

    st.subheader("Predictor Ranking")

    st.write(
        "The chart reproduces the previously documented "
        "Power BI predictor ranking values for comparison. "
        "These display values are not recalculated "
        "from the model."
    )

    predictor_ranking_chart()

    st.caption(
        "The display ranking contains 21 predictors. "
        "The trained model uses 22 predictors; the final "
        "model importance CSV remains the authoritative "
        "source for complete rankings."
    )

    st.subheader("Model Predictors")

    st.write(
        "The calibrated Logistic Regression model uses "
        "22 predictors. Key factors in the model ranking "
        "include OverTime, StockOptionLevel, "
        "YearsSinceLastPromotion, NumCompaniesWorked "
        "and TotalWorkingYears."
    )

    st.caption(
        "Predictor rankings describe model behaviour "
        "and do not establish causation."
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

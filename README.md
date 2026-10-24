<p align="center">
  <img src="IBM%20HR%20Attrition%20Analytics%20Emblem.png" alt="IBM HR Attrition Analytics Logo" width="450">
</p>

# IBM HR Analytics & Attrition Dashboard

## 1. Project Overview

Employee attrition can affect workforce stability, operational continuity and recruitment planning. Understanding employee attrition patterns is important for developing effective retention strategies.

This project uses the IBM HR Employee Attrition dataset to analyse employee attrition patterns, develop and evaluate predictive machine learning models, and propose practical HR retention recommendations.

The main objective is to **support HR in reducing employee attrition**, while enabling more proactive workforce planning through data-driven insights.

## 2. Business Problem

How can employee data be analysed to identify key factors associated with employee attrition, predict employees' likelihood of leaving, and support targeted retention strategies and proactive workforce planning to reduce attrition and minimise staffing gaps?

## 3. Project Objectives

1. Analyse overall employee attrition patterns, departmental differences and the relationship between overtime and attrition.
2. Develop and compare machine learning models to predict employee attrition.
3. Evaluate the selected predictive model, assess probability calibration and identify important attrition predictors.
4. Assess predicted employee attrition risk and identify higher-risk employee groups across departments.
5. Propose practical HR retention strategies and an implementation plan based on the analytical findings.

## 4. Dataset Overview

**Dataset:** IBM HR Employee Attrition dataset

The project uses five related CSV files:

- Employees.csv
- Compensation.csv
- Performance.csv
- Experience.csv
- Employment Status.csv

The datasets were merged using `EmployeeNumber` as the common identifier.

### Prepared Datasets for Power BI

Two prepared datasets are available in this GitHub repository:

- **IBM_HR_Attrition_Cleaned.csv:** The cleaned historical dataset containing all 1,470 employee records, used for descriptive analytics.
- **HR_Attrition_Predictive_Risk.csv:** The predictive risk dataset containing 1,233 employees who stayed, including estimated attrition probabilities and risk categories.

These datasets are imported directly into Power BI to support the interactive dashboard.

### Dataset Summary

| Indicator | Result |
|---|---:|
| Total Employees | 1,470 |
| Employees Who Left | 237 |
| Employees Who Stayed | 1,233 |
| Overall Attrition Rate | 16.12% |

**Note:** The IBM HR dataset is synthetic. Findings are intended for educational and analytical demonstration rather than direct application to a real organisation.

## 5. Tools and Technologies

| Tool | Purpose |
|---|---|
| Python | Data cleaning, analysis and predictive modelling |
| Pandas and NumPy | Data preparation and transformation |
| Matplotlib and Seaborn | Data visualisation |
| Scikit-learn | Machine learning, calibration and model evaluation |
| Google Colab | Python development environment |
| GitHub | Project documentation and version control |
| Power BI | Interactive HR analytics dashboard |

## 6. Data Preparation

The five datasets were integrated into a single employee-level dataset.

Key preparation activities included:

- Merging datasets using `EmployeeNumber`.
- Checking duplicate records, data types and missing values.
- Preparing numerical and categorical variables for analysis.
- Creating employee grouping variables for descriptive analytics.
- Selecting 22 predictors for machine learning.
- Separating training and testing datasets using a stratified split.
- Applying missing-value imputation, scaling and categorical encoding within machine learning pipelines to reduce data leakage.

The dataset was divided into:

- **Training dataset:** 1,176 employees (80%).
- **Testing dataset:** 294 employees (20%).

A stratified split was used to preserve the proportion of employees who left and stayed.

## 7. Descriptive Analytics

The descriptive analysis examined overall employee attrition patterns, departmental differences and factors associated with employee attrition.

The analysis covered:

- **Overall attrition:** Total employees, employees who left and overall attrition rate.
- **Departmental attrition:** Comparison of attrition rates across Sales, Human Resources and Research & Development.
- **Overtime and attrition:** Examination of the relationship between overtime and employee attrition across departments.
- **Employee characteristics:** Analysis of attrition patterns by age, job role, monthly income, total working years, years since last promotion, performance rating, job satisfaction and work-life balance.

### Departmental Attrition

| Department | Total Employees | Employees Who Left | Attrition Rate |
|---|---:|---:|---:|
| Sales | 446 | 92 | 20.63% |
| Human Resources | 63 | 12 | 19.05% |
| Research & Development | 961 | 133 | 13.84% |

### Key Findings

- Sales recorded the highest observed attrition rate at 20.63%.
- Employees working overtime experienced higher observed attrition rates across departments.
- Lower-income and less-experienced employee groups generally showed higher attrition.
- Employee satisfaction, job roles and other characteristics were examined to identify potential retention concerns.

These findings represent associations and do not establish the causes of employee attrition.

## 8. Predictive Analytics

Two machine learning models were developed using 22 selected predictors:

- Logistic Regression
- Random Forest

Both models were evaluated using five-fold stratified cross-validation on the training dataset.

### Cross-Validation Performance

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| ROC AUC | 0.831 | 0.808 |
| Recall | 0.753 | 0.463 |
| F1-Score | 0.495 | 0.496 |

**Logistic Regression was selected** because it achieved higher ROC AUC (0.831 vs 0.808) and substantially higher recall (0.753 vs 0.463). Although Random Forest achieved a marginally higher F1-score, Logistic Regression was considered more suitable for this project because identifying employees at potential risk of leaving is a key priority for proactive HR retention planning.

### Held-Out Test Performance

The selected Logistic Regression model was evaluated on 294 previously unseen test employees.

| Evaluation Metric | Logistic Regression |
|---|---:|
| Accuracy | 0.759 |
| Precision | 0.370 |
| Recall | 0.723 |
| F1-Score | 0.489 |
| ROC AUC | 0.795 |

The model correctly identified 34 of the 47 employees who left in the test dataset. However, it also classified 58 employees who stayed as likely departures at the default classification threshold.

This demonstrates the importance of considering both missed departures and false alerts when interpreting model predictions.

### Probability Calibration

Because the original Logistic Regression model used balanced class weights, its predicted probabilities required further assessment.

Sigmoid probability calibration was applied using cross-validation on the training dataset.

| Training Cross-Validation Measure | Result |
|---|---:|
| Original Model Brier Score | 0.160 |
| Calibrated Model Brier Score | 0.098 |
| Calibrated Model ROC AUC | 0.827 |
| Mean Calibrated Predicted Probability | 16.2% |
| Actual Training Attrition Rate | 16.2% |

The calibrated model achieved a lower cross-validated Brier Score (0.098 compared with 0.160), indicating improved probability prediction quality. The mean calibrated predicted probability (16.2%) was also aligned with the actual training attrition rate (16.2%).

Calibration was performed using the training dataset, while the held-out test dataset was kept separate for subsequent evaluation and risk analysis.

### Important Predictive Factors

Cross-validated permutation importance analysis identified the leading predictive factors:

1. Total Working Years
2. Overtime
3. Job Role
4. Business Travel
5. Stock Option Level
6. Monthly Income
7. Job Satisfaction
8. Years With Current Manager
9. Years Since Last Promotion
10. Job Involvement

These factors contributed to model prediction performance. Their importance does not demonstrate that they cause employee attrition.

## 9. Employee Attrition Risk Study

The calibrated Logistic Regression model was used to estimate employee attrition probabilities for the held-out test dataset.

Employees were classified into Low, Medium and High Risk categories using thresholds derived from cross-validated training predictions.

The approximate thresholds were:

- **Low Risk:** Predicted probability up to 9.87%.
- **Medium Risk:** Above 9.87% and up to 26.74%.
- **High Risk:** Above 26.74%.

### Risk Segmentation Results

| Risk Category | Employees | Employees Who Left | Observed Attrition Rate |
|---|---:|---:|---:|
| Low Risk | 163 | 12 | 7.36% |
| Medium Risk | 70 | 5 | 7.14% |
| High Risk | 61 | 30 | 49.18% |

The High Risk category contained 61 employees, representing approximately 20.7% of the test dataset.

Of the 47 employees who actually left, 30 were classified as High Risk. This means the High Risk group captured **63.8% of observed departures**.

However, Low and Medium Risk groups had similar observed attrition rates. The model differentiated High Risk employees more clearly than it differentiated Low and Medium Risk employees.

Risk categories represent relative predicted risk rather than certainty that an employee will leave.

### Departmental Risk Findings

| Department | High Risk Employees | High Risk Proportion |
|---|---:|---:|
| Sales | 31 | 37.80% |
| Human Resources | 3 | 20.00% |
| Research & Development | 27 | 13.71% |

Sales had the highest proportion of employees classified as High Risk in the test dataset.

The Human Resources test sample was small, so its results should be interpreted cautiously.

### High Risk Employee Profiles

The analysis compared employee characteristics across predicted risk levels within each department.

High Risk employees in Sales and Research & Development generally showed:

- Greater overtime exposure.
- Fewer total working years.
- Lower median monthly income.

For example, in Research & Development, 70.37% of High Risk employees worked overtime, compared with 11.02% of Low Risk employees.

Job satisfaction was also examined, although differences across risk groups were not consistent in every department.

These patterns partly reflect characteristics used by the prediction model and should not be interpreted as independent evidence of causation.

The held-out test dataset was used to evaluate predicted risk categories against actual attrition outcomes. Separately, the calibrated model was applied to the 1,233 employees who stayed to demonstrate how predictive risk assessment could support HR retention planning. These analyses serve different purposes and represent different employee populations.

## 10. HR Retention Recommendations

Based on the descriptive and predictive findings, five retention strategies are proposed for further investigation and pilot implementation.

| Priority Area | Proposed HR Recommendation |
|---|---|
| Working Experience and Career Development | Strengthen onboarding, mentoring, employee check-ins and development support for less-experienced employees. |
| Workload and Overtime | Review overtime patterns, staffing levels and workload distribution, particularly in teams with greater overtime exposure. |
| Job Role and Compensation | Review attrition patterns by job role, compensation competitiveness, internal pay equity and career progression opportunities. |
| Employee Engagement | Conduct pulse surveys, structured feedback sessions and department-specific improvement initiatives. |
| Sales Retention | Conduct a focused Sales retention review and pilot suitable interventions before wider implementation. |

### Departmental Priority

**Sales is proposed as the initial focus** because it recorded both the highest overall observed attrition rate (20.63%) and the highest proportion of High Risk employees in the test dataset (37.80%).

Research & Development should also be included in targeted investigation, particularly regarding overtime exposure and less-experienced employees.

Interventions should be informed by employee feedback and operational investigation rather than predictive scores alone.

## 11. Proposed Implementation Roadmap

A phased approach is recommended to investigate retention concerns, pilot interventions and evaluate their effectiveness.

| Phase | Timeline | Main Activities |
|---|---|---|
| Phase 1: Investigate | Months 1–2 | Review attrition patterns, overtime, employee feedback and working experience; identify priority groups. |
| Phase 2: Pilot | Months 3–5 | Introduce selected workload, mentoring, engagement and career development initiatives. |
| Phase 3: Evaluate | Months 6–8 | Monitor satisfaction, overtime, participation and retention indicators against baseline performance. |
| Phase 4: Refine and Scale | Months 9–12 | Improve initiatives based on evaluation and consider wider implementation. |

The timelines are proposed for planning purposes and do not represent actual implementation commitments.

### Suggested HR Monitoring Indicators

- Overall and departmental attrition rates.
- Attrition rates by experience group, job role and income group.
- Overtime hours and percentage of employees working overtime.
- Employee satisfaction scores.
- Six-month and twelve-month retention.
- Participation in retention initiatives.
- Changes in retention outcomes following pilot implementation.

Predictive risk scores should support workforce-level analysis and voluntary employee support, not serve as the sole basis for individual employment decisions.

## 12. Interactive Power BI Dashboard

**Status: In progress**

### Power BI Report

[Download the Power BI Dashboard](IBM_HR_Attrition_Dashboard.pbix)

The Power BI report contains four interactive dashboard pages covering attrition overview, attrition drivers, predictive risk and employee risk details.

The report can be opened using Power BI Desktop.

### Page 1: Attrition Overview

- Overall employee attrition rate and employee counts.
- Departmental attrition comparisons.
- Attrition patterns across employee groups.

### Page 2: Attrition Drivers

- Analysis of employee characteristics associated with attrition.
- Comparisons across overtime, income, job roles and other relevant factors.
- Interactive filters to explore attrition patterns.

### Page 3: Predictive Risk

- Distribution of predicted attrition risk among employees who stayed.
- Low, Medium and High Risk segmentation.
- Departmental comparisons to identify employee groups requiring further investigation.
- Model inferencing results generated using the calibrated Logistic Regression model in Python.

### Page 4: Employee Risk Details

- Employee-level predictive risk information.
- Predicted attrition probabilities and risk categories.
- Interactive filtering to support further analysis.

### Model Inferencing

The trained and calibrated Logistic Regression model was applied in Python to the 1,233 employees who stayed to generate predicted attrition probabilities.

Employees were classified into Low, Medium and High Risk categories based on the model's predicted probabilities and the risk thresholds established during model development.

The inference results were exported to `HR_Attrition_Predictive_Risk.csv` and imported into Power BI for interactive visualisation.

This demonstrates how machine learning predictions can be used to support HR in identifying higher-risk employee groups and planning targeted retention initiatives.

### Data Integration

The Power BI dashboard imports two datasets directly from this GitHub repository:

| Power BI Dataset | GitHub Source |
|---|---|
| HR_Attrition_Historical | IBM_HR_Attrition_Cleaned.csv |
| HR_Attrition_Predictive_Risk | HR_Attrition_Predictive_Risk.csv |

The historical dataset contains 1,470 employee records. The predictive risk dataset contains 1,233 employees who stayed, with model-generated attrition risk estimates.

The dashboard complements the Python analysis by enabling interactive exploration of employee attrition patterns and predicted risk.

Predictive risk scores are intended to support HR investigation and retention planning, not to determine individual employment decisions.

## 13. Project Limitations

- The IBM HR dataset is synthetic and may not reflect actual organisational behaviour.
- Predictive results are based on a limited held-out test dataset.
- The Human Resources department has a small sample, limiting the reliability of departmental comparisons.
- Low and Medium Risk groups showed similar observed attrition rates in the test dataset.
- The dataset does not contain resignation dates, recruitment lead times or onboarding durations needed to estimate recruitment start dates.
- Statistical associations and model predictions do not establish causation.
- Predictive models require further validation, calibration assessment and bias review before real-world use.
- Employee risk predictions should not be used as the sole basis for individual employment decisions.

## 14. Conclusion

This project demonstrates how descriptive and predictive analytics can support HR decision-making.

The descriptive analysis identified departmental attrition patterns, while predictive modelling estimated employee attrition risk using a calibrated Logistic Regression model.

The risk analysis highlighted Sales as a priority for further investigation and identified employee characteristics that may inform targeted retention initiatives.

The findings provide a foundation for HR retention strategies, ongoing performance monitoring and more proactive workforce planning.

The ultimate business objective is **not simply to predict employee departures, but to help HR identify retention concerns early and evaluate actions that may improve employee retention**.

## 15. Python Notebook

[View the Google Colab Notebook on GitHub](IBM_HR_Attrition_Capstone.ipynb)

The notebook contains the full data preparation, exploratory analysis, predictive modelling, risk study and HR retention recommendations.

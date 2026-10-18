
# Employee Attrition Analytics & Prediction

### A Data-Driven Approach to Employee Retention and Workforce Planning

**Capstone Project | Associate Data Analyst Programme**

## 1. Project Overview

Employee attrition can affect workforce stability, operational continuity and recruitment planning. Understanding why employees leave is important for developing effective retention strategies.

This project uses the IBM HR Employee Attrition dataset to analyse employee attrition patterns, develop a predictive machine learning model and propose practical HR retention recommendations.

The main objective is to **support HR in reducing employee attrition**, while enabling more proactive workforce planning through data-driven insights.

## 2. Business Problem

Employee attrition can create manpower gaps, increase recruitment costs and affect organisational performance.

When HR relies mainly on historical attrition figures, retention and recruitment decisions may become reactive.

This project explores how descriptive and predictive analytics can help HR:

- Understand employee attrition patterns.
- Identify factors associated with employee departures.
- Identify employee groups with higher predicted attrition risk.
- Prioritise potential retention initiatives.
- Support proactive workforce planning.

## 3. Project Objectives

1. Analyse overall and departmental employee attrition patterns.
2. Identify individual and combined factors associated with employee attrition.
3. Develop and evaluate machine learning models to predict employee attrition.
4. Classify employees into relative attrition risk categories.
5. Recommend practical HR strategies to investigate and address retention concerns.

## 4. Dataset Overview

**Dataset:** IBM HR Employee Attrition dataset

The project uses five related CSV files:

- Employees.csv
- Compensation.csv
- Performance.csv
- Experience.csv
- Employment Status.csv

The datasets were merged using `EmployeeNumber` as the common identifier.

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
| Scikit-learn | Machine learning and model evaluation |
| Google Colab | Python development environment |
| GitHub | Project documentation and version control |
| Power BI | Interactive HR analytics dashboard (planned) |

## 6. Data Preparation

The five datasets were integrated into a single employee-level dataset.

Key preparation activities included:

- Merging datasets using EmployeeNumber.
- Checking data types, missing values and consistency.
- Preparing numerical and categorical variables for analysis.
- Separating training and testing datasets.
- Applying preprocessing within the machine learning pipeline to reduce data leakage.

The dataset was divided into:

- **Training dataset:** 1,176 employees (80%)
- **Testing dataset:** 294 employees (20%)

A stratified split was used to preserve the proportion of employees who left and stayed.

## 7. Descriptive Analytics

The descriptive analysis examined overall attrition and departmental patterns across nine factors:

1. Age
2. Job Role
3. Monthly Income
4. Total Working Years
5. Years Since Last Promotion
6. Performance Rating
7. Job Satisfaction
8. Work-Life Balance
9. Overtime

### Departmental Attrition

| Department | Total Employees | Employees Who Left | Attrition Rate |
|---|---:|---:|---:|
| Sales | 446 | 92 | 20.63% |
| Human Resources | 63 | 12 | 19.05% |
| Research & Development | 961 | 133 | 13.84% |

### Key Findings

- Sales recorded the highest observed attrition rate at 20.63%.
- Employees working overtime experienced higher observed attrition rates across all three departments.
- Lower-income and less-experienced employee groups generally showed higher attrition.
- Job satisfaction and other employee characteristics were examined to identify potential retention concerns.

These findings represent associations and do not establish the causes of employee attrition.

## 8. Predictive Analytics

Two machine learning models were developed:

- Logistic Regression
- Random Forest

The models were evaluated using accuracy, precision, recall, F1-score and ROC AUC.

### Model Performance

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.704 | 0.823 |
| Precision | 0.304 | 0.451 |
| Recall | 0.660 | 0.489 |
| F1-Score | 0.416 | 0.469 |
| ROC AUC | 0.733 | 0.753 |

Random Forest achieved higher accuracy, precision, F1-score and ROC AUC, while Logistic Regression achieved higher recall.

The Random Forest model was subsequently calibrated to improve the reliability of predicted attrition probabilities.

### Probability Calibration

| Metric | Original Random Forest | Calibrated Random Forest |
|---|---:|---:|
| ROC AUC | 0.753 | 0.761 |
| Brier Score | 0.151 | 0.118 |

The calibrated model achieved a lower Brier Score on the test dataset, indicating improved probability prediction quality.

### Important Predictive Factors

Permutation importance analysis identified the following leading predictive factors:

1. Overtime
2. Job Satisfaction
3. Monthly Income
4. Age
5. Work-Life Balance

Predictive importance does not demonstrate that a factor causes attrition.

## 9. Employee Attrition Risk Study

The calibrated Random Forest model was used to estimate employee attrition probabilities.

Employees in the held-out test dataset were classified into three relative risk categories using thresholds derived from out-of-fold training predictions.

| Risk Category | Employees | Observed Attrition Rate |
|---|---:|---:|
| Low Risk | 159 | 7.55% |
| Medium Risk | 77 | 15.58% |
| High Risk | 58 | 39.66% |

The High Risk category showed a higher observed attrition rate than the Low and Medium Risk categories.

These categories represent relative predicted risk rather than certainty that an employee will leave.

### Departmental Risk Findings

In the test dataset, the proportion of employees classified as High Risk was:

| Department | High Risk Employees | High Risk Proportion |
|---|---:|---:|
| Human Resources | 2 | 13.33% |
| Research & Development | 31 | 15.74% |
| Sales | 25 | 30.49% |

Sales had the highest proportion of High Risk employees in the test sample.

The Human Resources sample was small, so departmental comparisons should be interpreted cautiously.

### High Risk Employee Profiles

Across departments, employees classified as High Risk generally showed:

- Higher overtime participation.
- Lower median monthly income.
- Younger average age.
- Lower job satisfaction in several comparisons.

These profiles partly reflect the characteristics used by the prediction model. They should not be interpreted as independent evidence of causation.

## 10. HR Retention Recommendations

The findings suggest four areas for further HR investigation and potential retention initiatives.

| Priority Area | Proposed HR Recommendation |
|---|---|
| Workload and Overtime | Review workload distribution, staffing levels and sustained overtime patterns. |
| Employee Engagement | Conduct pulse surveys, manager check-ins and targeted employee feedback sessions. |
| Compensation and Career Development | Review pay equity, compensation competitiveness and development opportunities. |
| Early-Career Retention | Strengthen onboarding, mentoring and structured career support. |

### Departmental Priority

Sales is proposed as the initial focus for retention investigation because it recorded both the highest overall observed attrition rate and the highest proportion of High Risk employees in the test dataset.

Interventions should be validated through employee feedback, operational investigation and pilot evaluation before wider implementation.

## 11. Proposed Implementation Roadmap

| Phase | Timeline | Main Activities |
|---|---|---|
| Phase 1: Investigate and Plan | Months 1–2 | Review retention concerns and establish baseline KPIs. |
| Phase 2: Pilot Initiatives | Months 3–6 | Implement selected workload, engagement and development initiatives. |
| Phase 3: Evaluate and Improve | Months 7–12 | Review outcomes, refine interventions and consider wider rollout. |

### Suggested HR Monitoring Indicators

- Overall and departmental attrition rates.
- Overtime hours per employee.
- Employee satisfaction scores.
- First-year employee retention.
- Six-month and twelve-month retention.
- Participation in retention initiatives.

## 12. Interactive Power BI Dashboard

**Status: Planned**

An interactive Power BI dashboard will be developed to present descriptive and predictive analytics findings.

The dashboard is intended to support:

- Overall HR attrition monitoring.
- Departmental and employee-group comparisons.
- Exploration of attrition-related factors.
- Review of predicted employee risk categories.
- HR management discussions and workforce planning.

Dashboard screenshots and details will be added after development.

## 13. Project Limitations

- The IBM HR dataset is synthetic and may not reflect actual organisational behaviour.
- Predictive results are based on a limited held-out test dataset.
- The dataset does not contain resignation dates, recruitment lead times or onboarding durations needed to estimate recruitment start dates.
- Statistical associations and model predictions do not establish causation.
- Predictive models require further validation before real-world use.
- Employee risk predictions should not be used as the sole basis for individual employment decisions.

## 14. Conclusion

This project demonstrates how descriptive and predictive analytics can support HR decision-making.

The analysis identified employee groups with higher observed attrition, developed a predictive model and highlighted areas for further retention investigation.

The findings provide a foundation for targeted HR initiatives, regular performance monitoring and more proactive workforce planning.

The ultimate business objective is not simply to predict employee departures, but to help HR identify retention concerns early and evaluate actions that may improve employee retention.

## 15. Python Notebook

[View the Google Colab Notebook on GitHub](IBM_HR_Attrition_Capstone.ipynb)

The notebook contains the full data preparation, exploratory analysis, predictive modelling, risk study and HR recommendations.

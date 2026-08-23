# Architecture and Design: Attrition Prediction API

This document explains the full journey behind this project: why it was built, the difficulties encountered, why specific decisions were made, and what the results actually mean.

## 1. Why This Was Built

The original attrition analysis existed as a course project (Kent State, Sem 1, Comp Stat), done in R, producing static statistical output. That was useful for understanding the data, but not usable by anyone outside that one analysis session. This project rebuilds that analysis in Python and turns it into a live, servable system: a trained model that any external application can query for a real-time prediction, plus a plain-English dashboard so a non-technical HR user can use it directly, with zero coding knowledge required.

## 2. Why Python, Not the Original R Code

The original project files (.RData, .Rhistory, final-project.R) were R-based. Rather than porting R code to Python line by line, the model was retrained from scratch in Python on the same dataset (the IBM HR Analytics Employee Attrition dataset, 1,470 employees). This was a deliberate choice: R and Python handle categorical variable encoding differently under the hood, and the entire deployment path (FastAPI, joblib, scikit-learn) is Python-native. Retraining cleanly in Python avoided subtle encoding mismatches and produced an artifact directly compatible with the serving layer.

## 3. Data Exploration Findings

The dataset had zero missing values across all 35 columns, clean data, no imputation needed. The target variable, however, revealed a significant class imbalance: only 237 of 1,470 employees (16.1 percent) had actually left the company, versus 1,233 (83.9 percent) who stayed. This imbalance shaped every subsequent modeling decision, since a naive model could achieve high accuracy by simply predicting stayed for everyone, while being useless at the one thing that matters: catching real attrition risk.

## 4. Why Logistic Regression Was Chosen First

The general principle applied here: start with the simplest model that fits the problem shape, and only add complexity once a concrete limitation is found. For a binary classification problem on structured, tabular data, logistic regression is the standard starting point, for several reasons:

- It is fast to train and requires no hyperparameter search to get a reasonable first result.
- Its coefficients are directly interpretable: each feature gets one number, and the sign and magnitude of that number can be explained to a non-technical stakeholder in plain language.
- It establishes a baseline. Any more complex model must be shown to meaningfully outperform this baseline to justify its added complexity, slower training, and reduced interpretability.

No concrete evidence was found during this project that a more complex model was necessary. Logistic regression's results were reasonable and, critically, fully explainable, which matters more for a client-facing HR tool than a marginal accuracy gain would.

## 5. Difficulty Encountered: Convergence Warnings

The first several training runs produced a ConvergenceWarning from scikit-learn's optimizer, even after increasing max_iter from 1000 to 2000. The warning's own message hinted at the actual cause: unscaled features. Columns like MonthlyIncome (ranging into the thousands) and JobSatisfaction (ranging 1 to 4) sat on wildly different numeric scales. This forces the optimizer to take simultaneously tiny and huge steps across different dimensions, slowing and destabilizing convergence.

The fix: StandardScaler was applied to every feature before training, transforming each into a z-score (mean 0, standard deviation 1). This is a general statistical technique, not project-specific. It puts every feature on comparable footing so the optimizer can take efficient, similarly sized steps everywhere. After scaling, the convergence warning disappeared entirely, and, notably, model performance also measurably improved, not just the warning. This confirmed scaling was fixing a real modeling issue, not just silencing a cosmetic warning.

A related, easy to miss requirement: the scaler's mean and standard deviation were computed only from the training set (fit_transform), then applied unchanged to the test set (transform only). Letting the scaler see the test set while computing these statistics would leak information the model should not have access to during evaluation, inflating reported performance beyond what would actually be seen on genuinely new data.

## 6. Difficulty Encountered: Class Imbalance and the Accuracy Trap

An early experiment without any class weighting produced 88 percent overall accuracy, but a closer look at the classification report showed only 32 percent recall on the Left class, meaning the model was missing roughly two thirds of employees who actually left. High accuracy was hiding a model that was, for the actual business problem, not very useful: it was mostly just predicting the majority class by default.

The fix: class_weight was used to make the model penalize mistakes on the minority (Left) class more heavily during training. Three configurations were compared directly:

| Configuration | Precision (Left) | Recall (Left) | F1 (Left) |
|---|---|---|---|
| No weighting | 0.79 | 0.32 | 0.45 |
| class_weight=balanced | 0.34 | 0.64 | 0.44 |
| Manual (0:1, 1:2), final | 0.49 | 0.45 | 0.47 |

This is a genuine trade-off, not a bug to be eliminated: weighting toward the minority class increases recall (catches more real leavers) at the cost of precision (more false alarms). The final configuration was chosen as a deliberate middle ground, reflecting a judgment that for attrition specifically, missing a real at-risk employee is typically costlier to a business than an unnecessary check-in with someone who was fine. This ratio is a tunable business decision, not a fixed constant. A client with different priorities could reasonably request a different weighting.

## 7. What the Model Actually Learned

After scaling, the model's coefficients (each representing a feature's learned influence, in comparable units) were ranked by magnitude. The strongest predictors of attrition, in order:

1. OverTime = Yes (strongest single predictor): working overtime substantially increases attrition risk.
2. JobRole = Laboratory Technician and BusinessTravel = Travel_Frequently: specific roles and frequent travel are strong risk factors.
3. TotalWorkingYears (negative): more career experience reduces risk.
4. EnvironmentSatisfaction and JobSatisfaction (negative): higher satisfaction reduces risk, as expected.

These findings are directly actionable for an HR client: overtime policy and travel expectations are levers a business can actually pull, unlike, say, an employee's age or education field, which showed far less influence.

## 8. Known Limitation: Interaction Effects

Logistic regression assumes each feature contributes independently. It cannot natively detect cases where two features only matter in combination (for example, overtime might only be a strong risk factor when job satisfaction is also low, not on its own). Tree-based models (random forest, gradient boosting) can capture these interactions structurally, since they make sequential, conditional decisions rather than summing independent contributions. This was identified as a known limitation of the current model, and a candidate direction for future iteration, not pursued in this version, since no concrete evidence yet showed interaction effects were meaningfully hurting performance for this dataset.

## 9. Why FastAPI, and Why a Separate Dashboard

The trained model, on its own, was only usable by running a Python script, unusable by any external system, and definitely unusable by a non-technical HR employee. Two separate layers were added to solve two separate problems:

FastAPI turns the model into a network-accessible service: any software, in any language, can send it employee data via a standard HTTP request and get a prediction back, without needing Python, scikit-learn, or knowledge of how the model works internally. Pydantic's built-in validation also rejects malformed requests automatically, before they reach the model.

Streamlit dashboard solves a different problem: even a working API is not something a non-technical HR person can use directly, since they do not write HTTP requests. The dashboard provides a plain form (dropdowns, sliders, number fields) that collects the same information a technical API call would need, sends it to the API behind the scenes, and displays the result as a plain language, color coded message rather than raw JSON.

This mirrors the layered structure of a typical real-world ML deployment: model, then API, then user interface, each solving a distinct problem, each independently useful.

## 10. A Deliberate UX Trade-off in the Dashboard

The dashboard form collects only the roughly 12 fields shown to matter most from the coefficient analysis (overtime, satisfaction scores, tenure, role, and so on), rather than all 30 original input fields. The remaining, less influential fields (for example DailyRate, HourlyRate) are set to reasonable fixed defaults behind the scenes. This was a deliberate design choice: a 30 field form for a quick risk check would be a poor tool in practice, even though it would technically expose more configurability. Reducing the interface to the fields that actually move the prediction meaningfully was judged more valuable than full parameter exposure.

## 11. Model Persistence and Consistency

Three artifacts are saved from training and loaded by the API at startup: the trained model (attrition_model.joblib), the fitted scaler (scaler.joblib), and the exact list and order of columns the model expects (model_columns.joblib). All three are required together. A single new employee record, once one-hot encoded, will not naturally contain all the same columns the full training set had, since a single row cannot produce a Department_Sales column if that employee happens to be in a different department. The saved column list is used to force any new input into exactly the same shape the model was trained on, filling any absent categories with zero, before scaling and predicting.

## 12. Possible Extensions

- Compare against a random forest or gradient boosting model directly, to quantify whether interaction effects meaningfully improve recall or F1 over the current baseline.
- Add SHAP-based per-prediction explanations, so a specific employee's risk score can be broken down into why: which features pushed the prediction up or down, and by how much, rather than only having aggregate, model-wide coefficients.
- Deploy both the API and dashboard publicly, not just localhost, so the tool is usable outside the local development machine.

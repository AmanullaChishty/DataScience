Created a template structure and added the intitial libraries to get started. 
Cmd to enter virtual env after cd to applied-ml-churn repo -  source .venv/bin/activate

Problem Statement: Customer churn is costly for telecom businesses because it reduces recurring revenue and increases customer acquisition costs. Using telco_churn.csv, which contains customer demographics, tenure, subscribed services, contract type, and billing/payment information, this project aims to predict whether a customer will churn (Churn = Yes/No). The goal is to identify churn risk early and understand the key factors driving churn to support retention actions

Target: Churn (binary classification: Yes = churned, No = retained)


Deliverables:

- Data cleaning & preprocessing workflow (handle categorical encoding, missing/invalid values, type fixes like TotalCharges)
- Exploratory Data Analysis (EDA) with churn rate breakdowns (e.g., by Contract, tenure, InternetService, PaymentMethod)
- Baseline churn model (e.g., Logistic Regression) + evaluation (Accuracy, Precision/Recall, F1, ROC-AUC, confusion matrix)
- Improved model (e.g., Random Forest / XGBoost / Gradient Boosting) with comparison to baseline
- Model interpretation (feature importance + business insights: what factors most contribute to churn)
- Final report section: “Key findings + actionable retention recommendations”
- Reproducible run steps (requirements.txt, and commands to run notebook/script)


Evaluation Approach

To evaluate churn prediction performance fairly and avoid data leakage, the dataset will be split into train / validation / test sets.
- Train set: used to fit model parameters.
- Validation set: used to tune hyperparameters, choose thresholds, and compare candidate models.
- Test set: held out until the end and used exactly once for the final unbiased evaluation.

Split plan: train/val/test = 70% / 15% / 15% (stratified by Churn to keep the churn rate similar across splits).

Metrics
We will report ROC-AUC and PR-AUC on validation and test sets:
- ROC-AUC (Area Under the ROC Curve): measures how well the model ranks churners above non-churners across all thresholds.
- PR-AUC (Area Under the Precision–Recall Curve): measures the tradeoff between precision and recall across thresholds, focusing on the positive class (Churn = Yes).

Why PR-AUC is important
Churn prediction is often imbalanced (fewer churners than non-churners). In imbalanced settings, ROC-AUC can look strong even when the model produces too many false positives. PR-AUC is more sensitive to performance on churners, and aligns better with outreach use-cases where:
Precision matters (contacting the wrong customers wastes effort/cost),
while still tracking recall (catching as many true churners as possible).
In addition to AUC metrics, we will select an operating threshold (e.g., top-N highest risk customers) and report precision/recall at that threshold to match real retention outreach constraints.

ROC-AUC (what it means in simple terms)
ROC-AUC answers:
“If I pick one churner and one non-churner at random, how often does the model give a higher churn score to the churner?”
0.5 = random guessing
1.0 = perfect ranking
Key point: ROC-AUC looks at ranking quality across all thresholds.

Precision, Recall, and PR-AUC (why it matters for churn)

When you predict churn, your “positive class” is usually:
Positive = Churn = Yes
Precision = Of the people I predicted will churn, how many actually churned? “Were my outreach targets worth contacting?”
Recall = Of all true churners, how many did I catch? “Did I miss many people who were going to leave?”
The Precision–Recall curve shows the tradeoff between these as you change the score threshold, and PR-AUC summarizes that curve.

Why PR-AUC is better than ROC-AUC for imbalanced churn
If churners are rare, a model can get a nice ROC-AUC while still creating a lot of false alarms.
Example intuition:
Suppose only 10% churn.
A model flags many customers as high risk.
You might end up contacting lots of non-churners → expensive / annoying.
PR-AUC penalizes this more directly because precision will drop when false positives increase.
So, for a retention team, PR-AUC is often closer to the real question:
“When I reach out to the ‘high-risk’ list, what fraction actually churns?”

Practical add-on (recommended): evaluate at “top-N”
Businesses often don’t contact everyone — they contact the top 500 / top 5% highest-risk customers.
So beyond AUC:
Evaluate Precision@K (e.g., Precision among top 5% risk)
Evaluate Recall@K (how many churners are captured in that top list)
This turns model performance into something actionable.


Operational thresholding

Model probabilities must be converted into an “action/no-action” decision for retention outreach. We choose an operational threshold based on business constraints:
Capacity-based (top_pct): If we can contact only a fixed fraction of customers (e.g., 15%), we choose a threshold that flags approximately that fraction as “high risk.”
Recall-based (min_recall): If we require catching a minimum fraction of churners (e.g., recall ≥ 0.70), we choose a threshold that meets the recall constraint and maximizes precision among feasible thresholds.
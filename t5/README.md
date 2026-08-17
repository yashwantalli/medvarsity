# Diabetes Prediction — End-to-End ML Pipeline

A full machine learning pipeline on the Pima Indians Diabetes dataset: non-trivial missing-value imputation, EDA, feature engineering, model selection, and an extensive evaluation suite (including SHAP, threshold analysis, and multiple imbalance-handling strategies).

## Project Brief

Per the stated requirements, this pipeline had to:
- Take **uncleaned** data and handle missing values **without** standard mean/median/mode imputation
- Perform proper EDA
- Do feature engineering
- Justify model choice based on the data
- Evaluate with: accuracy, precision, recall (sensitivity), specificity, F1, ROC-AUC, PR-AUC, confusion matrix, classification report, MCC, Cohen's Kappa, probability distribution plots, threshold analysis, and SHAP values/plots

## Dataset

`diabetes.csv` — the Pima Indians Diabetes dataset: 768 rows, 8 features + binary `Outcome` (1 = diabetes, 0 = no diabetes).

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood based on family history |
| Age | Age in years |
| Outcome | Target: 1 = diabetic, 0 = not |

## 1. Data Cleaning — Regression-Based Imputation (No Mean/Median/Mode)

The dataset encodes missing values as literal `0`s in physiologically impossible fields (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). These were first converted to `NaN`.

Instead of naive imputation, missing values were predicted via **linear regression using correlated features**, column by column:

| Column | Missing (before) | Strongest predictor | Multi-feature R² | Single-feature R² | Missing after regression |
|---|---|---|---|---|---|
| Insulin | 374 | Glucose (r=0.58) | 0.351 | 0.413 | 4 |
| SkinThickness | 227 | BMI (r=0.65) | 0.434 | 0.488 | 9 |
| BMI | 11 | SkinThickness (r=0.65) | 0.531 | 0.500 | 9 |
| BloodPressure | 35 | Age (r=0.33) | 0.194 | 0.111 | 0 |
| Glucose | 5 | Insulin (r=0.58) | 0.518 | 0.495 | 4 |

**Approach:** for each column, fit a multi-feature linear regression on rows where the target and predictors are all present, then predict the missing values; where predictor features are themselves missing, fall back to the single strongest-correlated feature. A Pearson correlation heatmap across the missing-prone columns guided predictor selection.

**Remaining gaps** (a handful of rows per column where even the fallback predictors were missing) were closed with **KNN imputation** (`sklearn.impute.KNNImputer`) across all numeric columns — still not mean/median/mode.

**Outlier handling:** IQR-based capping (clip to `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`) after visualizing distributions with boxplots across all 8 features.

## 2. EDA

- `.describe()`, `.info()`, null/zero-value counts before and after cleaning
- Grouped statistics by `Outcome` for `Glucose`, `BMI`, `Insulin` to check separability
- Correlation heatmap among the missing-prone features to guide imputation
- Scatter plots for imputation-relevant pairs (e.g., Glucose vs. Insulin, BMI vs. SkinThickness)
- Boxplots per feature for outlier inspection

## 3. Feature Engineering

New interaction/derived features tested in later pipeline iterations:
- `Glucose_BMI` — Glucose × BMI
- `Insulin_Glucose_Ratio` — Insulin / (Glucose + 1)
- `Age_Pregnancies` — Age × Pregnancies

## 4. Model Selection

Six classifiers were trained and compared on standardized features (`StandardScaler`, 80/20 stratified split):

- Logistic Regression
- Decision Tree
- Random Forest
- **Gradient Boosting** ← selected
- SVM (probability=True)
- KNN (k=5)

**Why Gradient Boosting:** it ranked 1st across 7 of 8 metrics (Accuracy, Precision, Recall, F1, AUC-ROC, MCC, Cohen's Kappa) and tied for 2nd on Specificity (0.84 vs. 0.85). Its Recall of 0.630 was the highest among all models — the most important property for a diabetes *screening* use case, where missing a true diabetic (false negative) is costlier than a false alarm. AUC-ROC of 0.824 confirmed strong discrimination, and MCC (0.479) / Kappa (0.478) indicated moderate-to-good agreement beyond chance.

**Threshold tuning:** lowering the default 0.5 threshold to **0.3** boosted Recall to 0.87 and F1 to 0.707 — the recommended operating point for a clinical screening scenario prioritizing sensitivity.

## 5. Imbalance-Handling Iterations

Multiple approaches were benchmarked head-to-head in a final comparison cell:

1. **Baseline** — default Gradient Boosting
2. **Feature Engineering + SMOTE (100%) + Tuned GB** — engineered features above, SMOTE oversampling, `RandomizedSearchCV`-tuned Gradient Boosting
3. **XGBoost + `scale_pos_weight`** — class-imbalance correction via positive-class weighting instead of resampling, with manual threshold sweep (0.30–0.50)
4. **XGBoost + `RandomizedSearchCV`** — hyperparameter search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `scale_pos_weight`, `gamma`, `reg_alpha`, `reg_lambda` (5-fold stratified CV)
5. **XGBoost + best RandomizedSearchCV params + CV-tuned threshold** *(in `improved_cells.ipynb`)* — the most rigorous variant:
   - Computes `scale_pos_weight` directly from the training split
   - Maps the best `GradientBoostingClassifier` hyperparameters found via search onto their XGBoost equivalents
   - Finds the optimal decision threshold via **out-of-fold (OOF) predictions from 5-fold CV on training data only** (sweeping 0.20–0.65, maximizing F1) — critically, the test set is never touched during threshold selection, avoiding leakage
   - Applies that CV-chosen threshold once, at the end, to the held-out test set for final evaluation

All approaches are compared side-by-side on Accuracy, Precision, Recall, F1, AUC-ROC, MCC, and Kappa in a grouped bar chart.

## 6. Evaluation Suite

Implemented for the final model(s):

- Classification report (precision, recall, F1, support)
- Confusion matrix (heatmap, labeled No Diabetes / Diabetes)
- **ROC curve** — plotted as a proper step function (`drawstyle='steps-post'`) rather than a smoothed line, with AUC-ROC
- **Precision-Recall curve** with average precision (PR-AUC)
- Probability distribution plot — histogram and KDE of predicted probabilities split by true class, with default (0.5) and CV-tuned threshold lines marked
- **Threshold analysis** — Accuracy/Precision/Recall/F1/Specificity swept across thresholds 0.1–0.9, with the optimal (max-F1) point identified
- **MCC** (Matthews Correlation Coefficient) and **Cohen's Kappa**
- **SHAP** — `shap.Explainer` summary plot (feature impact distribution) and bar plot (mean absolute SHAP importance) for model interpretability

## Repository Contents

| File | Description |
|---|---|
| `t5_ds.ipynb` | Main pipeline notebook: cleaning → EDA → feature engineering → model comparison → full evaluation suite → SHAP |
| `improved_cells.ipynb` | Refined final-stage cells (XGBoost + mapped best params + leak-free CV threshold tuning) — designed to be appended after the `RandomizedSearchCV` cell in the main notebook |
| `diabetes.csv` / `archive.zip` | Source dataset (Pima Indians Diabetes) |
| `requirement` | Original task specification |

## Tech Stack

Python · Pandas · NumPy · scikit-learn (LinearRegression, KNNImputer, StandardScaler, GradientBoostingClassifier, RandomizedSearchCV, StratifiedKFold) · XGBoost · imbalanced-learn (SMOTE) · SHAP · Matplotlib · Seaborn

## Key Takeaways

- Regression-based (rather than mean/median) imputation preserves inter-feature relationships that matter for a clinical dataset where missingness is informative (e.g., Insulin missingness correlates with the value itself).
- Model selection should weight the metric that matches the deployment context — here, Recall/sensitivity for a screening tool — not just raw accuracy.
- Threshold tuning must be done on out-of-fold or validation data, never on the test set, to get an honest final evaluation.
- SHAP values provide per-feature interpretability on top of the aggregate metrics, useful for clinical trust and feature auditing.

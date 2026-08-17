# Car Crash Prediction on Busy Streets Using Logistic Regression and XGBoost

A comparative machine learning study estimating the probability of a car crash from urban traffic, weather, and infrastructure conditions.

## Overview

Urban crashes cluster where traffic density, speed, and pedestrian flow interact with poor weather and road conditions. This project builds a synthetic, privacy-safe dataset with known risk relationships and trains two models — an interpretable linear model (Logistic Regression) and a nonlinear ensemble (XGBoost) — to predict crash occurrence before it happens, then statistically compares them.

## Dataset

- **6,500** synthetic records, **10** input features, near 50/50 class balance (Crash = 49.7%, No Crash = 50.3%)
- **Split:** stratified 80/20 → 5,200 training / 1,300 test records
- **Generation:** a hidden linear risk score is computed from weighted feature contributions, passed through a logistic sigmoid to get crash probability, then crash labels are drawn via Bernoulli sampling (`np.random.binomial`)

**Features**

| Feature | Description |
|---|---|
| `traffic_density` | Vehicles in the area |
| `rainfall` | Rainfall level |
| `road_condition` | Road surface quality (1–5) |
| `signal` | Signal delay |
| `speed` | Vehicle speed |
| `intersection` | Number of intersections |
| `visibility` | Visibility level |
| `pedestrians` | Pedestrian count |
| `time_of_day` | Categorical (early morning, morning, afternoon, evening, night, midnight) |
| `vehicle_type` | Categorical (car, cars+bikes, cars+bus, heavy vehicles, mixed) |

**Target:** `crash_occurs` (1 = Crash, 0 = No Crash)

## Methodology

1. **Encoding** – categorical fields (road condition, time of day, vehicle type) converted to numeric form
2. **Scaling** – features standardized (`StandardScaler`) for Logistic Regression; trees are scale-invariant so XGBoost uses unscaled features
3. **Stratified 80/20 split** preserving crash/no-crash balance
4. **Quality checks** – no missing values, feature correlations reviewed pre-training

Both models are trained and evaluated on identical processed features and the same held-out test set, isolating algorithm choice as the only source of performance difference.

## Models

**Logistic Regression** — `Risk = β₀ + β₁x₁ + … + βₙxₙ`, `P(crash) = 1 / (1 + e⁻Risk)`. L2-regularized, trained on standardized features. Chosen for its transparent, coefficient-level interpretability (odds ratios) in a safety-critical setting.

**XGBoost** — gradient-boosted decision trees added sequentially, each correcting prior errors. Captures nonlinear effects and feature interactions automatically, with built-in regularization.

## Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Accuracy | 0.767 | 0.765 |
| Precision | 0.765 | 0.765 |
| Recall | 0.766 | 0.760 |
| F1-score | 0.766 | 0.762 |
| ROC-AUC | 0.854 | 0.845 |

The two models are near-identical on accuracy, precision, recall, and F1 (differing by at most ~0.5 pp). Logistic Regression holds a consistent edge on ROC-AUC.

**Top risk drivers (Logistic Regression odds ratios)**

| Feature | Odds ratio | Effect |
|---|---|---|
| Rainfall | 3.13× | Strong risk increase |
| Traffic density | 1.80× | Risk increase |
| Pedestrian count | 1.78× | Risk increase |
| Vehicle speed | 1.56× | Risk increase |
| Visibility | 0.82× | Protective |
| Road condition (good) | 0.41× | Strong protective |

XGBoost's gain-based feature importance ranks road condition and rainfall highest, broadly agreeing with the logistic odds ratios.

**Threshold tuning:** the default 0.50 threshold was compared against a sweep from 0.10–0.90. F1 peaks at a threshold of **0.40** (Precision 0.73, Recall 0.86, F1 0.79), trading some precision for higher recall — appropriate when missing a real crash is costlier than a false alarm.

**Significance testing (DeLong's test):** comparing the two ROC-AUCs on the same test set gives Z = 2.21–3.83 and p = 0.027–0.0001 across runs, rejecting the null hypothesis of equal AUC at α = 0.05. Logistic Regression's AUC advantage is statistically significant, not due to chance.

## Conclusion

Logistic Regression is the recommended model: it matches XGBoost on accuracy while holding a statistically significant AUC edge, and it remains fully interpretable — essential for explaining *why* risk is elevated in a safety-critical deployment. This result is expected given the data-generating process: the synthetic risk score is linear, which favors a linear model by construction.

## Limitations

- **Synthetic data** — encodes assumed relationships and may not capture real-world noise, rare events, or complexity
- **Limited features** — omits driver behavior, signal timing detail, and intersection geometry
- **No temporal modeling** — records are treated independently
- **Unverified generalization** — untested on real crash data; the 0.40 threshold should be re-tuned on field data

## Future Work

- Real-world validation on field crash records
- Richer spatial, temporal, and behavioral features from sensors/traffic systems
- Sequence and deep-learning models where added complexity is justified
- Real-time scoring from streaming traffic/weather feeds
- Integration with signal control and driver-warning systems

## Repository Contents

| File | Description |
|---|---|
| `t1.ipynb` | Full notebook: data generation, preprocessing, model training, evaluation, threshold tuning, DeLong's test |
| `logistic_odds_ratios.csv` | Logistic Regression coefficients and odds ratios per feature |
| `report.pdf` / `Car_Crash_Prediction_Using_Machine_Learning.docx` | Full written report |
| `ppt.pdf` | Presentation slides summarizing the study |

## Tech Stack

Python 3.12 · NumPy · Pandas · scikit-learn (Logistic Regression, StandardScaler) · XGBoost · SciPy (DeLong's test) · Matplotlib

## References

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*, 44(3).
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8).
- World Health Organization (2023). Global Status Report on Road Safety.
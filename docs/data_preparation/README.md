# Stage 1 — Data Preparation

Notebook: `src/notebooks/air_quality_regression.ipynb` (Stage 1 sections 1.0–1.12)
Dataset: UCI Air Quality (`data/air+quality/AirQualityUCI.csv`), hourly readings, 10 Mar 2004 – 4 Apr 2005.

## Assignment requirements

| Stage | Requirement |
|---|---|
| Data preparation | Load data, handle missing values, choose a target, select features, 80/20 split |
| Model training | Linear and polynomial regression from scratch (explain the degree choice, cross-check with sklearn); Decision Tree with `max_depth` 3, 5, 10, None |
| Evaluation | MSE, RMSE, R² for all models; metric vs epoch (regression) and vs depth (tree) |
| Visualization | Depth-3 tree plot; feature importances |
| Comparison and discussion | Linear vs Tree; most important features; which model did better and why |
| Deliverables | Notebook (.ipynb) plus a PDF with code, results and discussion |

## Loading and cleaning

- The CSV is semicolon-separated with decimal commas, plus two empty trailing columns and blank rows. These are removed.
- Date and Time are combined into a datetime index. The series is a clean hourly grid with no duplicates or gaps.

## Missing values

- `-200` is the missing-value marker in every column. It is converted to `NaN`.
- `NMHC(GT)` is about 90% missing, so it is excluded.
- 366 rows have all sensor and weather readings missing (instrument outages). They carry no information and are dropped.
- Rows with no `CO(GT)` value are dropped. The target is never imputed.
- Result: **7,344 usable hourly rows** (78.5% of 9,357).

## Target and features

- **Target: `CO(GT)`** (mg/m³), kept in original units.
  - `NMHC(GT)` is too incomplete.
  - `C6H6(GT)` is almost a copy of the `PT08.S2` sensor (|r| ≈ 0.98), so it would be a trivial task.
  - `NOx(GT)` and `NO2(GT)` are about 17% missing, and `NOx` is very skewed.
  - `CO(GT)` is strongly but not perfectly related to the sensors (|r| ≈ 0.92 with `PT08.S2`).
- **Features (8):** `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`. The other ground-truth pollutants are not used as predictors.

## EDA findings

- **Target distribution:** right-skewed (skew ≈ 1.36), mean 2.13, median 1.8, max 11.9.
- **Class balance** (CO bands): Low (<1.5) 38.4%, Moderate (1.5–3) 38.2%, High (3–5) 18.6%, Very high (≥5) 4.7%. This is about 8:1, so the target is imbalanced.
- **Scale:** features differ by up to about 1,000× in standard deviation.
- **Outliers:** kept. About 70% of flagged CO hours sit next to another flagged hour, and they concentrate in the 08–09 h and 17–20 h rush hours. No physically impossible values remain.
- **Relationships:**
  - Strongest with CO: `PT08.S2` (0.92), `PT08.S1` (0.88), `PT08.S5` (0.85), `PT08.S3` (−0.70).
  - Weather has weak linear association (|r| < 0.05).
  - `PT08.S3` is monotonic but non-linear.
- **Multicollinearity:** VIF above 10 for `PT08.S2` (15), `T` (14), `PT08.S4` (10), `AH` (10). The condition number is about 11.
- **Time structure:**
  - Daily peaks about 2.9 mg/m³ at 09 h and 3.7 at 19 h.
  - Weekend dip.
  - Monthly CO highest in Oct–Dec, lowest in Aug.
  - `PT08.S4` drifts down over the year.
  - Lag-1 autocorrelation 0.83.

## Preprocessing decisions

| Aspect | Decision | Why |
|---|---|---|
| Missing marker | `-200 → NaN` | min is −200 in every column |
| Missing rows | drop, no imputation | whole-row outages and multi-day target gaps |
| Feature scaling | z-score, fitted on train only | scale mismatch; needed for gradient descent and polynomial terms; no test leakage |
| Min-max scaling | not used | outliers would squash the bulk of the data |
| Target | original units | interpretable RMSE; log is an optional experiment |
| Outliers | kept | real pollution episodes |
| Split | 80/20, shuffled, stratified on CO bands, seed 42 | keeps train and test representative |
| Regularisation | none now; Ridge held in reserve for polynomial regression | multicollinearity; to be chosen on training data only |
| Polynomial degree | choose by validation/CV on the training set | keeps the test set untouched |

Split sizes: train 5,875, test 1,469. Band shares match within about 0.1 percentage points.

## Why a shuffled split

A time-ordered split trains on Mar 2004 – Jan 2005 (includes summer) and tests on the colder, drier late Jan – Apr 2005. That gives a large shift: `T` −1.0 sd, `AH` −1.1 sd, `PT08.S4` −1.2 sd. The test would mostly measure season extrapolation. The shuffled split is used instead. Because neighbouring hours are highly correlated, test scores will look somewhat optimistic. This is stated in the notebook.

## Hand-off to Stage 2

`X_train_s`, `X_test_s`, `y_train`, `y_test`, `FEATURES`, `TARGET`, `scaler`.

## Verification notes

- The notebook runs top to bottom with no errors and shows 14 figures.
- Observation text was checked against the outputs and corrected where needed (for example, CO peaks in Oct–Dec, not across winter).
- The rendered figures have not been visually reviewed yet.

## Assignment checklist: how each data-preparation item was handled

| Requirement | How it was handled | Notebook section |
|---|---|---|
| Load the dataset into Python | `pd.read_csv` with `sep=";"` and `decimal=","`. Removed blank rows and the 2 empty columns. Built a datetime index. Checked for duplicates and time gaps. | 1.1–1.2 |
| Handle missing values | Treated `-200` as missing. Measured missingness per column, over time and by overlap. Dropped `NMHC(GT)` (about 90% missing), the 366 outage rows, and rows with no `CO(GT)`. No imputation, because the feature gaps are whole-row outages and the target must not be filled in. | 1.3–1.5 |
| Choose one pollutant as the target | `CO(GT)`. Compared all five by completeness, skew and strongest feature correlation. Rejected `NMHC` (too incomplete), `C6H6` (almost a copy of `PT08.S2`), and `NOx`/`NO2` (about 17% missing; `NOx` very skewed). | 1.5 |
| Select relevant features (sensors + weather) | All 8: 5 `PT08.*` sensors plus `T`, `RH`, `AH`. Checked correlations, non-linearity and VIF. Kept all 8, because the brief asks for sensors plus weather and dropping collinear ones would not help trees. The other ground-truth pollutants are excluded, because they would not exist at prediction time and would leak the answer. | 1.5, 1.7–1.9 |

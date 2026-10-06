# Stage 1: Data Preparation

Notebook: `src/notebooks/air_quality_regression.ipynb` (Stage 1, sections 1.0-1.12).
Dataset: UCI Air Quality (`data/air+quality/AirQualityUCI.csv`), hourly readings, 10 Mar 2004 to 4 Apr 2005.

## Assignment requirements (data preparation)

| Requirement | How it was handled | Notebook section |
|---|---|---|
| Load the dataset | `pd.read_csv` with `sep=";"` and `decimal=","`; blank rows and the two empty columns removed; datetime index built; no duplicates or time gaps | 1.1-1.2 |
| Handle missing values | `-200` treated as missing; missingness measured per column, over time and by overlap; `NMHC(GT)` (about 90% missing), 366 fully blank outage rows and rows without the target dropped; nothing imputed | 1.3-1.5 |
| Choose one pollutant as the target | `CO(GT)` | 1.5 |
| Select features (sensors and weather) | All 8: the 5 `PT08.*` sensors plus `T`, `RH`, `AH` | 1.5, 1.7-1.9 |
| Split 80/20 | Shuffled, stratified on CO bands, seed 42; scaler fitted on the training set only | 1.12 |

## Missing values

- `-200` is the missing-value marker in every column and is converted to `NaN`.
- All sensor and weather readings share the same 366 missing hours (instrument outages), so there is nothing to impute for the features; those rows carry no information.
- Rows with no `CO(GT)` are dropped. The target is never imputed.
- Result: 7,344 usable hourly rows (78.5% of 9,357).

## Target choice

`CO(GT)` (mg/m3), kept in original units.
- `NMHC(GT)` is too incomplete.
- `C6H6(GT)` is almost a copy of the `PT08.S2` sensor (|r| about 0.98), which would make the task trivial.
- `NOx(GT)` and `NO2(GT)` are about 17% missing, and `NOx` is very skewed.
- `CO(GT)` is strongly but not perfectly related to the sensors (|r| about 0.92 with `PT08.S2`).

The other ground-truth pollutants are not used as predictors: they would not exist at prediction time and would leak the answer.

## EDA findings

- **Target:** right-skewed (skew 1.36), mean 2.13, median 1.8, maximum 11.9.
- **Class balance** (CO bands): Low (<1.5) 38.4%, Moderate (1.5-3) 38.2%, High (3-5) 18.6%, Very high (>=5) 4.7%; about 8:1.
- **Scale:** feature standard deviations differ by up to about 1,000 times.
- **Outliers:** kept. About 70% of flagged CO hours sit next to another flagged hour, and they concentrate in the 08-09 h and 17-20 h rush hours. No physically impossible values remain.
- **Relationships:** strongest with CO are `PT08.S2` (0.92), `PT08.S1` (0.88), `PT08.S5` (0.85) and `PT08.S3` (-0.70); weather is weakly associated (|r| < 0.05); `PT08.S3` is monotonic but non-linear.
- **Multicollinearity:** VIF above 10 for `PT08.S2` (15), `T` (14), `PT08.S4` (10) and `AH` (10); condition number about 11.
- **Time structure:** daily peaks of about 2.9 mg/m3 at 09 h and 3.7 at 19 h, a weekend dip, highest monthly CO in Oct-Dec and lowest in Aug, `PT08.S4` drifting down over the year, lag-1 autocorrelation 0.83.

## Preprocessing decisions

| Aspect | Decision | Why |
|---|---|---|
| Missing marker | `-200` to `NaN` | minimum is -200 in every column |
| Missing rows | drop, no imputation | whole-row outages; multi-day target gaps |
| Feature scaling | z-score, fitted on train only | scale mismatch; needed for gradient descent and polynomial terms; no test leakage |
| Min-max scaling | not used | outliers would squash the bulk of the data |
| Target | original units | interpretable RMSE; log tested as an ablation in Stage 2 |
| Outliers | kept | real pollution episodes |
| Split | 80/20, shuffled, stratified on CO bands, seed 42 | keeps train and test representative |
| Regularisation | none by default; Ridge studied in Stage 2 | multicollinearity; chosen on training data only |

Split sizes: train 5,875, test 1,469; band shares match within about 0.1 percentage points.

## Why a shuffled split

A time-ordered split trains on Mar 2004 to Jan 2005 (including summer) and tests on the colder, drier late Jan to Apr 2005, a large shift in `T` (-1.0 sd), `AH` (-1.1 sd) and `PT08.S4` (-1.2 sd). The shuffled split is therefore used. Because neighbouring hours are highly correlated, test scores are somewhat optimistic; the notebook states this and checks a chronological split in Stage 2 (it turned out not to be materially harder).

## Hand-off to Stage 2

`X_train_s`, `X_test_s`, `y_train`, `y_test`, `FEATURES`, `TARGET`, `scaler`.

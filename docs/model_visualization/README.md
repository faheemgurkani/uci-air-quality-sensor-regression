# Stage 4 — Model Visualization

Notebook: `src/notebooks/air_quality_regression.ipynb`, Stage 4 (sections 4.1–4.5).

## What was produced

- **Depth-3 tree plot (4.1):** fitted on unscaled features so thresholds are in sensor units. A check confirms its predictions equal those of the tree trained on standardised data. A rule list and a leaf table are included.
- **Feature importance (4.2):** four methods compared: tree impurity, permutation importance on the test set, absolute standardised linear coefficients with bootstrap CIs, and drop-one retraining.
- **Grouped permutation importance (4.3):** correlated sensors are shuffled together.
- **Partial dependence (4.4):** 1-D curves for the top four features, an on-data binned-means plot, and a 2-D plot of `PT08.S2` against `T`.
- **Episode reconstruction (4.5):** eight days around the highest test CO value.

## Findings

- The depth-3 tree uses only `PT08.S2(NMHC)`: all 7 splits, with thresholds 704, 846, 994, 1,121, 1,225, 1,400 and 1,587, and predicted CO rising from 0.72 to 7.11 mg/m³.
- `PT08.S2(NMHC)` ranks first in all 8 importance methods, with a mean share of 77%. Next are `T` (6.8%) and `PT08.S1(CO)` (6.1%). `PT08.S1` matters for the linear model but not for the trees, because it is redundant with `PT08.S2` (r ≈ 0.89).
- Shuffling all sensors together costs the linear model +1.47 mg/m³ RMSE; shuffling all weather variables costs +0.06.
- The polynomial model's permutation importances and partial-dependence curves look implausible (large weights on humidity, negative CO at low `PT08.S2`). The cause is that these methods evaluate the model at feature combinations that never occur. On the actual data it is the most accurate model (binned-mean deviation 0.09 mg/m³ vs 0.13 tuned tree, 0.23 linear, 0.26 depth-3 tree). Treat it as fragile off the data.
- All models follow the timing of pollution episodes but under-predict the highest peak (8.7 mg/m³ actual; polynomial 7.3, tree 7.1, linear 6.4).

# Stage 2 — Model Training

Notebook: `src/notebooks/air_quality_regression.ipynb` (Stage 2, sections 2.0–2.9). Inputs come from Stage 1 (`X_train_s`, `X_test_s`, `y_train`, `y_test`).

## Design

| Stage 1 finding | Consequence |
|---|---|
| Strong hourly autocorrelation | All choices use 5-fold CV on the training set, grouped by calendar week (57 groups), so no validation fold sits next to training hours. The test set is only used to report. |
| Collinear features (VIF > 10) | Ridge studied as an ablation; coefficients get bootstrap CIs. |
| Features ~10³× apart in scale | Gradient descent runs on standardised inputs; an ablation shows why. |
| Skewed target, 8:1 regime imbalance | Residuals and per-regime errors stored; log-target tested. |
| Shuffled split is optimistic | Out-of-time check run for the final models. |

## What was built

- **Linear regression from scratch** (NumPy): normal equation, batch gradient descent and Adam, with per-epoch train/test metrics. Verified by a finite-difference gradient check and recovery of known coefficients.
- **Polynomial regression from scratch**: own feature expansion (matches `sklearn.PolynomialFeatures`), re-standardised inside the model.
- **Metrics from scratch**: MSE, RMSE, R², MAE (match `sklearn.metrics`).
- **Decision trees** (sklearn): required `max_depth` of 3, 5, 10 and None, a depth sweep 1–20, a `max_depth × min_samples_leaf` grid with the 1-SE rule, and cost-complexity pruning.

## How the choices were made

- **Learning rate:** theoretical limit 0.238 (from λ_max); 0.25 diverges; 0.15 is the fastest stable rate. Chosen on training loss only.
- **Polynomial degree:** CV RMSE by degree is 0.488, 0.453, 0.500, 1.23 and 8.6 for degrees 1–5. Degree 2 is both the minimum and the 1-SE choice. Ridge rescues degrees 4–5 but never beats degree 2.
- **Tree:** CV-best depth is 6. The 1-SE choice is depth 5 with `min_samples_leaf=50`.

## Results (test set)

| Model | Train RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Mean baseline | 1.440 | 1.423 | −0.000 |
| Linear (scratch = sklearn) | 0.480 | 0.493 | 0.880 |
| Poly d=2 (scratch = sklearn) | 0.411 | 0.447 | 0.901 |
| Tree depth 3 | 0.552 | 0.585 | 0.831 |
| Tree depth 5 | 0.440 | 0.513 | 0.870 |
| Tree depth 10 | 0.260 | 0.544 | 0.854 |
| Tree depth None | 0.000 | 0.610 | 0.817 |
| Tree tuned (depth 5, leaf 50) | 0.470 | 0.516 | 0.869 |

Scratch and scikit-learn agree to numerical precision for every family and solver.

## Other findings

- `PT08.S2(NMHC)` dominates in every importance view and in the ablations. Weather alone is no better than the mean predictor but adds a small correction to the sensors.
- Ridge does not improve the linear model. Collinearity affects coefficient interpretation, not accuracy.
- Log-target hurts the linear model and is neutral for the others. Without scaling, gradient descent converges much more slowly.
- Every model under-predicts the rare Very-high CO hours (RMSE 2–3× larger there).
- The out-of-time split is not materially harder in RMSE (linear 0.484, polynomial 0.446, tuned tree 0.513), with the same model ranking.

## Artifacts kept in the notebook for later stages

`RESULTS`/`MASTER`, `FITTED`, `PRED`, `RESID`, `HISTORY` (per-epoch metrics for the two gradient-based models), `TREE_SWEEP`, `POLY_CV`, `IMP_TREE`, `PERM`, `COEF_BOOT`, `ABL_FEAT`, `ABL_SCALE`, `ABL_LOG`, `ABL_CRIT`, `RIDGE_PATH`/`RIDGE_CV`, `LC`, `REGIME`, `OOT`.

## Caveats

- Differences below about 0.03 RMSE are within fold-to-fold noise (CV sd ≈ 0.05).
- The depth-sweep test curve is shown for the Stage 3 plots only. Selection used CV.
- A few explanations in the notebook are flagged as "plausible, not tested" (for example the log-target result).

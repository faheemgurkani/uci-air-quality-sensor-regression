# Stage 3 — Model Evaluation

Notebook: `src/notebooks/air_quality_regression.ipynb`, Stage 3 (sections 3.1–3.7).

## What was reported

- **Required metrics (3.1):** MSE, RMSE and R² for all 15 models on train and test, plus MAE, CV scores and the train–test gap.
- **Uncertainty (3.2):** week-clustered bootstrap (1,000 resamples of 57 test weeks) for test RMSE and R², paired differences between models on the same resamples, and a fold-wise CV comparison. Hours inside a week are correlated, so weeks, not hours, are resampled.
- **Metrics vs epoch (3.3):** train and test MSE, RMSE and R² per epoch for the gradient-trained linear model and the polynomial (Adam) model, with the closed-form optimum marked, and a learning-rate convergence plot.
- **Metrics vs tree depth (3.4):** train, CV and test MSE, RMSE and R² for depths 1–20 and None, with the required depths 3, 5, 10 and None marked.
- **Residual diagnostics (3.5) and error breakdown (3.6):** predicted vs actual, residuals vs fitted, distributions, time, CO regime, hour of day and month.
- **Summary (3.7):** accuracy vs complexity vs fit time.

## Key results

| Model | Test RMSE | Test R² | 95% CI (RMSE) |
|---|---|---|---|
| Polynomial d=2 | 0.447 | 0.901 | 0.368 – 0.532 |
| Linear | 0.493 | 0.880 | 0.415 – 0.577 |
| Tree depth 5 | 0.513 | 0.870 | 0.440 – 0.591 |
| Tree depth 3 | 0.585 | 0.831 | 0.510 – 0.669 |
| Tree depth 10 | 0.544 | 0.854 | 0.473 – 0.616 |
| Tree depth None | 0.610 | 0.817 | 0.523 – 0.699 |

- Individual intervals overlap, so the paired differences are the real comparison. Polynomial beats linear (ΔRMSE −0.046, CI excludes 0, 5 of 5 folds). Linear beats the best tree by 0.020 (CI just touches 0) and wins 5 of 5 folds against every fixed-depth tree.
- Linear converges within 1% of its optimum in 135 epochs; the polynomial (Adam) takes about 640. Test error follows train error at every epoch, so there is no over-training.
- Tree test error is U-shaped in depth. CV-best depth is 6; test-best in hindsight is 7.
- The linear model's residuals show a U-shaped curve vs fitted values; the polynomial's are flat. All models are heteroscedastic and heavy-tailed.
- All models under-predict in the evening (18–21 h) and over-predict in the early morning (06–08 h). RMSE is 2–3 times larger in Nov–Jan than in summer.
- Gradient-based fits take about 150 times longer than the closed forms for identical results.

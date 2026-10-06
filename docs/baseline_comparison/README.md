# Stage 5 — Baseline Comparison (Linear vs Decision Tree)

Notebook: `src/notebooks/air_quality_regression.ipynb`, Stage 5 (sections 5.1–5.6).

## What was done

- **Head-to-head (5.1):** linear regression vs every tree configuration, with paired week-bootstrap CIs and per-fold wins.
- **Non-linearity probes (5.2):** nested F-test of linear vs degree 2, the largest polynomial terms, single-feature vs all-feature behaviour, and a hybrid of the linear model plus a tree fitted to its residuals (an extension, not an assignment model).
- **Synthetic demonstration (5.3):** one input, four known truths (linear, quadratic, sine, step), plus extrapolation. Labelled as synthetic.
- **Where each model wins (5.4):** RMSE by decile of `PT08.S2`, and behaviour outside the training range on the chronological split.
- **Ablation ledger (5.5)** and **scorecard (5.6).**

## Findings

- Linear has lower test RMSE than every tree: 0.493 vs 0.513 (depth 5), 0.516 (tuned), 0.585 (depth 3), 0.544 (depth 10), 0.610 (None). The gap to the best tree is small (CI just touches 0); the other gaps exclude 0. Linear wins 5 of 5 CV folds against every fixed-depth tree.
- Non-linearity exists but is smooth and mild. Degree 2 improves test R² from 0.880 to 0.901. A tree on the linear residuals lowers test RMSE from 0.493 to 0.470, still behind the polynomial (0.447).
- With `PT08.S2` alone the tuned tree (CV RMSE 0.534) beats the line (0.575). Adding the other seven features improves linear by 15% and the tree by only 4%. Trees are good at one-variable non-linearity and inefficient at adding variables up.
- Synthetic data: trees win on abrupt shapes (sine, step), the degree-2 polynomial wins on a quadratic, and trees freeze beyond the training range.
- The polynomial has the lowest RMSE in all ten `PT08.S2` deciles. The tuned tree beats linear only in the lowest decile.
- Only 3.3% of the out-of-time test hours fall outside the training range, and those hours are easy, so this data gives no empirical evidence about extrapolation failure.
- Ablation ledger: sensors-only is as good as all features; weather-only is as bad as the mean predictor; degree 2 and tree depth 5 are the right sizes; ridge, tree criterion, pruning and the chronological split are within noise; no scaling hurts gradient descent; a log target hurts the linear model.

# Stage 6 — Discussion and deliverables

Notebook: `src/notebooks/air_quality_regression.ipynb`, Stage 6. PDF report: `docs/report/air_quality_regression_report.pdf` (74 pages, generated from the executed notebook).

## Short answers

**Most important features.** `PT08.S2(NMHC)` is by far the most important: rank 1 in all 8 importance methods (mean share 77%), the only feature the depth-3 tree uses, and removing it costs the CV RMSE +0.12 (linear), +0.07 (polynomial) and +0.15 (tree). Temperature and `PT08.S1(CO)` come next. Weather alone is no better than predicting the mean; sensors alone are almost as good as everything.

**Decision Tree vs Linear Regression.** Linear Regression performed better overall (test RMSE 0.493 vs 0.513 for the best tree), though the margin over the best-sized tree is small. The degree-2 polynomial was best of all (0.447). Reasons: the relation is a smooth, mostly additive dose–response that a line represents with 9 numbers but a tree only as a staircase; trees add several variables up inefficiently; deeper trees overfit. Trees win only in the lowest sensor decile and on abrupt shapes.

## Limitations

- Random 80/20 split on autocorrelated hourly data makes test scores somewhat optimistic. Mitigated by week-grouped CV, week-clustered bootstrap, and a chronological check (a single split).
- One site, about 13 months, one sensor array; `PT08.S4` drifts.
- Differences among the top models are small relative to uncertainty.
- Coefficients and single-feature importances of correlated sensors are not reliable.
- Explanations marked "plausible, not tested" are hypotheses.

## Suggested next steps

Hour-of-day and day-of-week features, ensembles (random forest, gradient boosting), quantile or heteroscedastic regression for extreme episodes, time-aware nested CV, and sensor-drift monitoring.

## Deliverables

The notebook contains a checklist mapping every requirement of the brief to a notebook section (final cell of Stage 6). The PDF is regenerated from the notebook with `nbconvert` to HTML and headless Chrome printing.

# Sensor-Based Air Pollutant Concentration Regression

Predicting the true concentration of an air pollutant from low-cost metal-oxide sensor signals and weather readings, using the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality) (hourly measurements from an Italian city, March 2004 – April 2005). This is Assignment #1 of the Machine Learning course (MS, NUST).

Everything lives in one notebook, [`src/notebooks/air_quality_regression.ipynb`](src/notebooks/air_quality_regression.ipynb), saved with its outputs so results can be read directly on GitHub. A PDF export of the executed notebook (full code and outputs) is in [`docs/report/air_quality_regression_notebook.pdf`](docs/report/air_quality_regression_notebook.pdf). The house-style assignment report (cover page, results, discussion, code excerpts) is built from the notebook by the scripts in `src/report/`.

## Project status

| Stage | Contents | Status | Notes |
|---|---|---|---|
| 1. Data Preparation | Loading, missing values, EDA, target and feature selection, scaling, 80/20 split | Done | [docs](docs/data_preparation/README.md) |
| 2. Model Training | Linear and polynomial regression from scratch (checked against scikit-learn); Decision Tree with a `max_depth` sweep; CV tuning and ablations | Done | [docs](docs/model_training/README.md) |
| 3. Model Evaluation | MSE, RMSE, R²; metric vs epoch and vs tree depth; bootstrap CIs; residual analysis | Done | [docs](docs/model_evaluation/README.md) |
| 4. Model Visualization | Depth-3 tree plot, feature importances, partial dependence | Done | [docs](docs/model_visualization/README.md) |
| 5. Baseline Comparison | Linear regression vs Decision Tree, ablation ledger, scorecard | Done | [docs](docs/baseline_comparison/README.md) |
| 6. Discussion | Short written answers, limitations, deliverables checklist | Done | [docs](docs/discussion/README.md) |

## Approach in brief

- **Data:** `-200` is the dataset's missing-value marker. `NMHC(GT)` (about 90% missing), fully blank outage rows and rows without the target are dropped; nothing is imputed. Target: `CO(GT)`. Features: the 5 `PT08.*` sensors plus `T`, `RH`, `AH`. 7,344 hourly rows remain.
- **Validation:** hourly samples are strongly autocorrelated, so every modelling choice uses 5-fold CV grouped by calendar week on the training set. The test set is used only to report.
- **Models:** linear and polynomial regression written from scratch in NumPy (normal equation, gradient descent, Adam), each verified against scikit-learn; Decision Tree regressors for `max_depth` 3, 5, 10 and None, plus tuned and pruned trees.

## Results (test set)

| Model | RMSE (mg/m³) | R² |
|---|---|---|
| Mean predictor | 1.423 | 0.000 |
| Linear regression | 0.493 | 0.880 |
| Polynomial, degree 2 | **0.447** | **0.901** |
| Decision Tree, depth 3 | 0.585 | 0.831 |
| Decision Tree, depth 5 | 0.513 | 0.870 |
| Decision Tree, depth 10 | 0.544 | 0.854 |
| Decision Tree, depth None | 0.610 | 0.817 |

- `PT08.S2(NMHC)` is by far the most important feature (rank 1 in all 8 importance methods). Weather alone is no better than the mean predictor.
- Linear regression beats every Decision Tree; the margin over the best tree (depth 5) is small. The relation is a smooth, mostly additive dose–response that a line captures with 9 coefficients, while a tree needs a staircase of constant steps. The degree-2 polynomial is best overall.
- All models under-predict the rare very-high CO episodes.

Test scores are somewhat optimistic because of the random split of autocorrelated data; the notebook quantifies this with week-clustered bootstrap intervals and a chronological check.

## Repository layout

```
data/air+quality/        UCI Air Quality dataset (CSV and XLSX)
src/notebooks/           Main notebook: air_quality_regression.ipynb
src/report/              Scripts that build the assignment report from the notebook
docs/                    Notes for each stage (data_preparation, model_training,
                         model_evaluation, model_visualization,
                         baseline_comparison, discussion) and report/ (notebook PDF export)
```

## Running the notebook

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter ipykernel
jupyter notebook src/notebooks/air_quality_regression.ipynb
```

The notebook reads the data through a path relative to its own folder, so run it from `src/notebooks/` (Jupyter does this by default). A full run takes about 30 seconds.

To re-execute it from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace src/notebooks/air_quality_regression.ipynb
```

## Building the assignment report

The report is generated from the executed notebook, so its figures and numbers cannot drift from the analysis:

```bash
.venv/bin/python src/report/export_notebook_data.py        # figures, tables and code excerpts into src/report/build/
.venv/bin/python src/report/make_report.py --answers cover.json
```

`cover.json` holds the cover details (university, degree, student, student ID, course code and name, professor, assignment number and title, date). It contains personal details, so keep it out of git. The builder is a personal Claude Code skill, `geometrika-report-builder`, with its `msc-assignment` profile.

## Dataset

Source: S. De Vito et al., *Air Quality*, UCI Machine Learning Repository. Each row is one hourly average of five sensor signals (`PT08.S1`–`PT08.S5`), reference ("ground truth") analyser concentrations for CO, NMHC, benzene, NOx and NO₂, and temperature and humidity.

## License

MIT. See [`LICENSE`](LICENSE).

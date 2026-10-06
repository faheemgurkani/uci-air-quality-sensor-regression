# Sensor-Based Air Pollutant Concentration Regression

Predicting the true concentration of an air pollutant from low-cost metal-oxide sensor signals and weather readings, using the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality) (hourly measurements from an Italian city, March 2004 – April 2005). This is Assignment #1 of the Machine Learning course (MS, NUST).

The repository is built up in stages. Each stage is committed as it is completed.

## Project status

| Stage | Contents | Status |
|---|---|---|
| 1. Data Preparation | Loading, missing values, EDA, target and feature selection, scaling, 80/20 split | Done |
| 2. Model Training | Linear and polynomial regression from scratch (checked against scikit-learn); Decision Tree with a `max_depth` sweep | Planned |
| 3. Model Evaluation | MSE, RMSE, R²; metric vs epoch and vs tree depth | Planned |
| 4. Model Visualization | Depth-3 tree plot, feature importances | Planned |
| 5. Baseline Comparison | Linear regression vs Decision Tree | Planned |
| 6. Discussion | Short written answers | Planned |

## Stage 1 summary

- **Missing values:** `-200` is the dataset's missing-value marker. It is converted to `NaN`. `NMHC(GT)` (about 90% missing) is excluded, and rows with all readings missing (366) or no target are dropped. Nothing is imputed. 7,344 hourly rows remain.
- **Target:** `CO(GT)`, true hourly CO concentration (mg/m³).
- **Features:** the 5 `PT08.*` sensor signals plus temperature, relative humidity and absolute humidity.
- **Preprocessing:** standardisation fitted on the training set only. The 80/20 split is shuffled and stratified on CO concentration bands (seed 42).

Full details and reasoning are in [`docs/data_preparation/README.md`](docs/data_preparation/README.md).

## Repository layout

```
data/air+quality/        UCI Air Quality dataset (CSV and XLSX)
src/notebooks/           Main notebook: air_quality_regression.ipynb
docs/data_preparation/   Notes on the Stage 1 decisions
```

## Running the notebook

Requires Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter ipykernel
jupyter notebook src/notebooks/air_quality_regression.ipynb
```

The notebook reads the data through a path relative to its own folder, so run it from `src/notebooks/` (Jupyter does this by default). It is saved with its outputs, so results can be viewed directly on GitHub.

To re-execute it from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace src/notebooks/air_quality_regression.ipynb
```

## Dataset

Source: S. De Vito et al., *Air Quality*, UCI Machine Learning Repository. Each row is one hourly average of five sensor signals (`PT08.S1`–`PT08.S5`), reference ("ground truth") analyser concentrations for CO, NMHC, benzene, NOx and NO₂, and temperature and humidity.

## License

MIT. See [`LICENSE`](LICENSE).

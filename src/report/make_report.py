"""Build the assignment report PDF (house style, MSc profile) from the executed notebook.

    .venv/bin/python src/report/export_notebook_data.py          # figures, tables and code excerpts from the notebook
    .venv/bin/python src/report/make_report.py --answers PATH    # writes docs/report/air_quality_regression_report.pdf

--answers is a JSON file with the cover details (university, degree, student, student_id, course_code, course_name,
professor, assignment_no, assignment_title, date). Keep it out of git: it holds personal details.
The builder is the personal skill ~/.claude/skills/geometrika-report-builder (profile msc-assignment).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
ROOT = HERE.parents[1]
BUILDER = Path.home() / ".claude/skills/geometrika-report-builder/scripts/build_report.py"
R = json.loads((BUILD / "results.json").read_text())


def f(x, n=3):
    return f"{x:.{n}f}".replace("-", "−") if isinstance(x, (int, float)) else str(x)


def pct(x, n=1):
    return f"{x * 100:.{n}f}%"


def code(name):
    return (BUILD / "code" / f"{name}.py").read_text().rstrip()


def by(rows, key):
    return {r[key]: r for r in rows}


M = by(R["master"], "model")
S = {str(r["max_depth"]): r for r in R["tree_sweep"]}
H = by(R["h2h"], "Decision Tree")
BOOT = by(R["boot"], "index")
PAIR = by(R["paired"], "comparison (A vs B)")
DEG = R["deg_star"]
LIN, LGD, LSK = "Linear | scratch (normal eq.)", "Linear | scratch (GD)", "Linear | sklearn"
POL = f"Poly(d={DEG}) | scratch (normal eq.)"
T3, T5, T10, TN = "Tree | max_depth=3", "Tree | max_depth=5", "Tree | max_depth=10", "Tree | max_depth=None"
TT, TP, MEAN = "Tree | tuned (CV, 1-SE)", "Tree | pruned (ccp, CV)", "Baseline | mean predictor"
fig = lambda name, cap, w=0.97: {"type": "figure", "path": f"figures/{name}.png", "width": w, "caption": cap}
SC = {r["index"]: r for r in R["score"]}
FEAT_ABL = by(R["abl_feat"], "feature set")
SF = by(R["single_feat"], "feature set")
CO = by(R["coef"], "index")
top = R["top_share"]; top_names = list(top)
s2, s2share = top_names[0], list(top.values())[0]
dS2 = {m: FEAT_ABL["Drop PT08.S2(NMHC)"][m] - FEAT_ABL["All 8 features"][m] for m in ("Linear", f"Poly(d={DEG})", "Tree (tuned)")}
miss = {r["index"]: r for r in R["miss"]}
bal = R["bal"]; sb = R["split_bal"]


def mtab(models, cols):
    return [[m.replace("Tree | ", "Tree, ").replace("Baseline | ", "") ] + [f(M[m][c]) for c in cols] for m in models]


spec = {"profile": "msc-assignment", "sections": [], "references": [], "abstract": ""}
add = lambda title, blocks: spec["sections"].append({"title": title, "blocks": blocks})

spec["abstract"] = (
    f"This report predicts the hourly CO concentration (CO(GT)) from five metal-oxide sensor signals and three weather variables of the UCI Air Quality dataset [1]. "
    f"After removing the {R['n']['raw_rows'] - R['n']['model_rows']:,} unusable rows, {R['n']['model_rows']:,} hours remain, split 80/20 and standardised. "
    f"Linear and polynomial regression were written from scratch (NumPy) and verified against scikit-learn [2]; Decision Trees were trained for max_depth 3, 5, 10 and None. "
    f"All choices were made with week-grouped cross-validation. On the test set the degree-2 polynomial is best (RMSE {f(M[POL]['test_rmse'])} mg/m³, R² {f(M[POL]['test_r2'])}), "
    f"linear regression follows (RMSE {f(M[LIN]['test_rmse'])}, R² {f(M[LIN]['test_r2'])}) and beats every Decision Tree (best: depth 5, RMSE {f(M[T5]['test_rmse'])}). "
    f"The PT08.S2(NMHC) sensor dominates every importance measure."
)

# ---------------------------------------------------------------- 1 summary
rows = [[n, f(M[k]["test_rmse"]), f(M[k]["test_r2"]), f(M[k]["cv_rmse"]) if M[k].get("cv_rmse") is not None else "-"] for n, k in
        [("Mean predictor", MEAN), ("Linear regression", LIN), (f"Polynomial, degree {DEG}", POL), ("Decision Tree, depth 3", T3), ("Decision Tree, depth 5", T5),
         ("Decision Tree, depth 10", T10), ("Decision Tree, depth None", TN)]]
add("Summary of results", [
    {"type": "qa", "question": "What was done, and what was found?",
     "answer": f"CO(GT) was predicted from eight sensor and weather features. The degree-{DEG} polynomial is the most accurate model; linear regression beats every Decision Tree, but only narrowly for the best-sized tree."},
    {"type": "table", "headers": ["Model", "Test RMSE (mg/m³)", "Test R²", "CV RMSE"], "widths": [0.4, 0.22, 0.18, 0.2], "rows": rows, "shade": [2]},
    {"type": "list", "items": [
        f"**Data.** `-200` is the missing-value marker; `NMHC(GT)` is {pct(miss['NMHC(GT)']['missing_%'] / 100, 0)} missing and is excluded; {R['n']['model_rows']:,} hourly rows are used.",
        f"**Most important feature.** `{s2}`, with a mean share of {pct(s2share, 0)} across eight importance measures; the weather variables alone are no better than the mean predictor.",
        f"**Linear vs Decision Tree.** Linear wins: test RMSE {f(M[LIN]['test_rmse'])} against {f(M[T5]['test_rmse'])} for the best tree; the relation is a smooth, mostly additive dose-response that a line represents with 9 numbers.",
        f"**Weak spot.** Every model under-predicts the rare very-high CO episodes.",
    ]},
    {"type": "callout", "title": "How to read the test scores",
     "text": "The 80/20 split is random and hourly samples are strongly autocorrelated, so test scores are somewhat optimistic. Every choice therefore used week-grouped cross-validation, and the uncertainty is shown with week-clustered bootstrap intervals (Section 4)."},
])

# ---------------------------------------------------------------- 2 data
add("Data preparation", [
    {"type": "qa", "question": "How was the dataset cleaned and prepared for modelling?",
     "answer": f"The semicolon-separated file was parsed, `-200` was treated as missing, outage rows and rows without the target were dropped (no imputation), features were standardised on the training set only, and the data were split 80/20 stratified on CO level."},
    {"type": "subsection", "title": "Loading and missing values"},
    {"type": "p", "text": f"The CSV uses `;` separators and decimal commas, with {R['n']['raw_rows'] - R['n']['clean_rows']} blank trailing rows and two empty columns. After removing them, {R['n']['clean_rows']:,} hourly records remain (10 Mar 2004 to 4 Apr 2005, no gaps or duplicates). The value `-200` appears in every column and is the dataset's missing-value marker; left untreated it would distort every statistic and fit."},
    {"type": "table", "headers": ["Column", "Missing", "Missing %", "Valid"], "widths": [0.4, 0.2, 0.2, 0.2], "size": "small",
     "rows": [[r["index"], f"{r['missing']:,}", f"{r['missing_%']:.1f}%", f"{r['valid']:,}"] for r in R["miss"]]},
    fig("missingness", "Missing share per column (left) and missingness over time (right). The 366 blank hours are instrument outages shared by all sensor and weather columns."),
    {"type": "list", "items": [
        f"`NMHC(GT)` is {miss['NMHC(GT)']['missing_%']:.0f}% missing (only {miss['NMHC(GT)']['valid']:,} valid hours) and is excluded.",
        "All sensor and weather columns share the same 366 missing hours; there are no partially missing feature rows, so nothing needs imputing and those rows are dropped.",
        f"`CO(GT)` is missing for {miss['CO(GT)']['missing']:,} hours in total. The target is never imputed: a label filled in by a heuristic would teach the model the heuristic.",
        f"Result: **{R['n']['model_rows']:,} usable hourly rows** ({pct(R['n']['model_rows'] / R['n']['clean_rows'])} of the cleaned records).",
    ]},
    {"type": "subsection", "title": "Target and feature selection"},
    {"type": "p", "text": "`CO(GT)` (mg/m³) is the target, kept in original units. `NMHC(GT)` is too incomplete; `C6H6(GT)` is almost a copy of the `PT08.S2` sensor (|r| ≈ 0.98), which would make the task trivial; `NOx(GT)` and `NO2(GT)` are about 17% missing and `NOx` is very skewed. `CO(GT)` is strongly but not perfectly related to the sensors. The features are the five `PT08.*` sensors plus `T`, `RH` and `AH`; the other ground-truth pollutants are not used because they would not exist at prediction time."},
    {"type": "table", "headers": ["Feature", "Pearson r with CO(GT)", "Spearman rho", "VIF"], "widths": [0.34, 0.24, 0.22, 0.2], "size": "small",
     "rows": [[r["index"], f(r["Pearson r"]), f(r["Spearman rho"]), f(next(v["VIF"] for v in R["vif"] if v["index"] == r["index"]), 1)] for r in R["target_rel"]]},
    {"type": "p", "text": "The sensors are strongly collinear (variance inflation factors above 10 for four features), and the weather variables are only weakly related to CO. Spearman exceeds Pearson most clearly for `PT08.S3(NOx)`, a monotonic but non-linear relation."},
    fig("correlations", "Pearson (left) and Spearman (right) correlations of the sensor, weather and target variables."),
    {"type": "subsection", "title": "Class-wise balance of the target"},
    {"type": "p", "text": "CO(GT) is continuous, so its balance is described with four concentration regimes. The regimes are imbalanced (about 8 : 1), which motivates a stratified split and regime-wise error reporting."},
    {"type": "table", "headers": ["CO regime (mg/m³)", "Hours", "Share", "Train share", "Test share"], "widths": [0.32, 0.14, 0.14, 0.2, 0.2],
     "rows": [[b["CO(GT)"], f"{b['count']:,}", f"{b['share_%']:.1f}%", f"{s['train_%']:.1f}%", f"{s['test_%']:.1f}%"] for b, s in zip(bal, sb)]},
    fig("class_balance", "Class-wise balance of the CO regimes and the empirical distribution of CO(GT).", 0.85),
    {"type": "subsection", "title": "Scaling, split and regularisation decisions"},
    {"type": "table", "headers": ["Aspect", "Decision", "Why"], "widths": [0.2, 0.34, 0.46], "size": "small", "rows": [
        ["Missing marker", "`-200` becomes missing", "the minimum is -200 in every column"],
        ["Outliers", "kept", "they are real rush-hour pollution episodes, not impossible values"],
        ["Feature scaling", "z-score, fitted on the training set only", "scales differ by about 1,000 times; needed for gradient descent and polynomial terms; no test leakage"],
        ["Target", "original units", "RMSE stays interpretable; a log target is tested as an ablation"],
        ["Split", f"80/20 shuffled, stratified on CO regime, seed 42 (train {R['n']['train']:,}, test {R['n']['test']:,})", "keeps the test set representative; regime shares match within about 0.1 percentage points"],
        ["Regularisation", "none by default; Ridge studied in Section 3", "collinearity; strength is chosen on training data only"]]},
    {"type": "callout", "title": "Why a shuffled split",
     "text": "A chronological split would train on March 2004 to January 2005 and test on the colder, drier late January to April 2005, a large shift in `T`, `AH` and the `PT08.S4` sensor. The shuffled split is used, with the limitation stated above; a chronological check in Section 4 shows no material loss in RMSE."},
])

# ---------------------------------------------------------------- 3 training
lr = R["lr_tbl"]
add("Model training", [
    {"type": "qa", "question": "How were the models trained, and how were their settings chosen?",
     "answer": f"Linear and polynomial regression were written in NumPy and checked against scikit-learn; the polynomial degree ({DEG}), the learning rate and the tree settings were chosen by week-grouped cross-validation on the training set, never on the test set."},
    {"type": "subsection", "title": "Validation protocol"},
    {"type": "p", "text": f"Neighbouring hours are near-duplicates, so a plain shuffled K-fold would leak. The training set is split into five folds made of whole calendar weeks ({R['groups']} weekly groups, `GroupKFold`); no week appears in both a training and a validation fold. All selection (learning rate, degree, ridge strength, tree depth and leaf size, ablations) uses these folds."},
    {"type": "subsection", "title": "Linear regression from scratch"},
    {"type": "p", "text": "One NumPy estimator implements ordinary least squares by the normal equation (`lstsq`), batch gradient descent and Adam; the objective is the mean squared error with an optional L2 term that never penalises the intercept. Only NumPy arithmetic is used inside the model: the scikit-learn interface serves purely as a wrapper for the shared cross-validation tools. The code is quoted in Appendix A. The analytic gradient agrees with finite differences to 5e-9, and both solvers recover known coefficients on synthetic data."},
    {"type": "p", "text": f"Plain gradient descent is stable only for a learning rate below 2/L, where L is twice the largest eigenvalue of X'X/n; for the standardised features this limit is {f(R['lr_stable'])}. The sweep agrees: a rate of 0.25 diverges and {R['lr_lin']} is the fastest stable rate (chosen on training loss only)."},
    {"type": "table", "headers": ["Learning rate", "Diverged", "Final train RMSE", "Epochs to 1% of optimum", "Distance to OLS coefficients"], "widths": [0.18, 0.14, 0.22, 0.22, 0.24], "size": "small",
     "rows": [[str(r["lr"]), "yes" if r["diverged"] else "no", "-" if r["diverged"] else f(r["final_train_rmse"]), "-" if r["diverged"] else f"{int(r['epochs_to_1%'])}", "-" if r["diverged"] else f"{r['||theta - theta_OLS||']:.3f}"] for r in lr]},
    fig("lr_sweep", "Learning-rate sweep for batch gradient descent (left) against the theoretical stability limit (right).", 0.95),
    {"type": "table", "headers": ["Linear model", "Train RMSE", "Test RMSE", "Test R²"], "widths": [0.4, 0.2, 0.2, 0.2], "rows": [
        [n, f(M[k]["train_rmse"], 4), f(M[k]["test_rmse"], 4), f(M[k]["test_r2"], 4)] for n, k in [("Scratch, normal equation", LIN), ("Scratch, gradient descent", LGD), ("scikit-learn LinearRegression", LSK)]]},
    {"type": "p", "text": "The three implementations give the same coefficients (to 1e-8 for the closed form) and identical metrics. The train-test RMSE gap is small (0.013), so the linear model is limited by bias, not variance."},
    {"type": "subsection", "title": "Polynomial regression and the choice of degree"},
    {"type": "p", "text": f"A from-scratch feature expansion builds every monomial of total degree 1 to d (it matches `sklearn.preprocessing.PolynomialFeatures` exactly); the expanded matrix is re-standardised with training-fold statistics before the least-squares fit. The degree is chosen by cross-validation, using the one-standard-error rule [3], [4]: the smallest degree whose CV RMSE is within one standard error of the best."},
    {"type": "table", "headers": ["Degree", "Terms", "Train RMSE", "CV RMSE", "CV RMSE sd", "CV R²", ""], "widths": [0.1, 0.12, 0.17, 0.17, 0.17, 0.12, 0.15], "size": "small", "shade": [DEG - 1],
     "rows": [[str(i + 1), f"{int(r['n_terms'])}", f(r["train_rmse"]), f(r["cv_rmse"]), f(r["cv_rmse_sd"]), f(r["cv_r2"]), "chosen" if (i + 1) == DEG else ""] for i, r in enumerate(R["poly_unreg"])]},
    fig("poly_degree", f"Degree selection. Left: training RMSE falls monotonically while CV RMSE is U-shaped. Centre and right: crossing the degree with a ridge penalty rescues degrees 4 and 5 but never beats degree {DEG}."),
    {"type": "p", "text": f"Degree {DEG} is both the CV minimum and the one-standard-error choice. Degrees 4 and 5 (494 and 1,286 collinear terms) extrapolate wildly on unseen weeks. The best combination with ridge is degree {R['poly_best'][0]}, alpha {R['poly_best'][1]:g}, with a CV RMSE only marginally lower, so no regularisation is needed. The scratch polynomial, its Adam-trained version ({R['gd_epochs']['poly']} epochs to within 1% of the optimum) and the scikit-learn pipeline agree on all metrics (train RMSE {f(M[POL]['train_rmse'], 4)}, test RMSE {f(M[POL]['test_rmse'], 4)})."},
    {"type": "subsection", "title": "Decision Tree regressors"},
    {"type": "p", "text": f"`DecisionTreeRegressor` was trained for the required depths 3, 5, 10 and None, followed by a depth sweep from 1 to 20, a joint grid over `max_depth` and `min_samples_leaf`, and cost-complexity pruning. The grid with the one-standard-error rule selects depth {R['tree_best']['max_depth']} with `min_samples_leaf` = {R['tree_best']['min_samples_leaf']} ({int(M[TT]['complexity'])} leaves). Trees are scale-invariant; this was verified, and they were fed the same standardised matrix for a like-for-like comparison."},
    {"type": "table", "headers": ["max_depth", "Actual depth", "Leaves", "Train RMSE", "Test RMSE", "Test R²", "CV RMSE"], "widths": [0.14, 0.14, 0.12, 0.15, 0.15, 0.14, 0.16], "size": "small",
     "rows": [[str(r["max_depth"]) if r["max_depth"] is not None else "None", f"{int(r['depth_actual'])}", f"{int(r['n_leaves']):,}", f(r["train_rmse"]), f(r["test_rmse"]), f(r["test_r2"]), f(r["cv_rmse"])] for r in R["req_depth"]]},
    {"type": "p", "text": "Depth 3 under-fits, depth 5 is the best of the four required values, and depth 10 and None over-fit: the unrestricted tree grows 4,245 leaves, memorises the training set (R² = 1.000) and loses to depth 5 on the test set by 19%."},
])

# ---------------------------------------------------------------- 4 evaluation
req_models = [LIN, LGD, LSK, POL, f"Poly(d={DEG}) | scratch (Adam GD)", f"Poly(d={DEG}) | sklearn", T3, T5, T10, TN, TT, TP, MEAN]
req_rows = [[m.replace("Tree | ", "Tree, ").replace("Baseline | ", "")] + [f(M[m][c]) for c in ("train_mse", "train_rmse", "train_r2", "test_mse", "test_rmse", "test_r2")] for m in M if m in M]
pair_rows = [[p["comparison (A vs B)"], f(p["ΔRMSE point"]), f"[{f(p['ΔRMSE 95% CI low'])}, {f(p['ΔRMSE 95% CI high'])}]", pct(p["P(A better)"], 0)] for p in R["paired"]]
add("Model evaluation", [
    {"type": "qa", "question": "How well does each model perform, and how certain are the numbers?",
     "answer": f"The degree-{DEG} polynomial is best (test RMSE {f(M[POL]['test_rmse'])}), followed by linear regression ({f(M[LIN]['test_rmse'])}) and the Decision Trees ({f(M[T5]['test_rmse'])} to {f(M[TN]['test_rmse'])}); the differences between the top models are small compared with the sampling uncertainty."},
    {"type": "subsection", "title": "MSE, RMSE and R² for all models"},
    {"type": "table", "headers": ["Model", "Train MSE", "Train RMSE", "Train R²", "Test MSE", "Test RMSE", "Test R²"], "widths": [0.35, 0.1, 0.11, 0.1, 0.1, 0.12, 0.12], "size": "small",
     "rows": req_rows, "shade": [list(M).index(POL)]},
    {"type": "p", "text": "MSE is in (mg/m³)², RMSE in mg/m³, R² is unitless. The scikit-learn and from-scratch versions of the same model are identical to four decimals, so the table holds nine distinct models. Overfitting appears only in the trees: the train-test RMSE gap grows from 0.034 (depth 3) to 0.073 (depth 5), 0.284 (depth 10) and 0.610 (depth None), against 0.013 for linear."},
    {"type": "subsection", "title": "Uncertainty: week-clustered bootstrap"},
    {"type": "p", "text": f"The test set is resampled by calendar week ({R['weeks_test']} weeks, 1,000 resamples), because hours inside a week are correlated; all models use the same resamples, so differences between models are paired."},
    fig("forest", "Test RMSE and R² with 95% week-bootstrap intervals (mean predictor omitted: its RMSE is about three times larger)."),
    {"type": "table", "headers": ["Comparison (A vs B)", "ΔRMSE (A − B)", "95% interval", "P(A better)"], "widths": [0.4, 0.18, 0.26, 0.16], "size": "small", "rows": pair_rows},
    {"type": "p", "text": f"Individual intervals overlap heavily (about ±0.08 RMSE), so the paired differences are the real comparison. Polynomial beats linear (ΔRMSE {f(PAIR['Poly d=2 vs Linear']['ΔRMSE point'])}, interval excludes 0, better in 5 of 5 CV folds). Depth 5 is clearly better than depth 3 and than None but not significantly better than depth 10. Linear against the trees is analysed in Section 6."},
    {"type": "subsection", "title": "Performance metrics versus epoch"},
    fig("epochs", "Train and test MSE, RMSE and R² per epoch for batch gradient descent (linear) and Adam (polynomial); dotted lines are the closed-form optimum."),
    {"type": "p", "text": f"Linear gradient descent reaches within 1% of its final training loss after {R['gd_epochs']['linear']} epochs and then follows the closed-form optimum exactly; the polynomial (Adam) needs about {R['gd_epochs']['poly']} epochs. Train and test curves move together at every epoch, so there is no over-training and early stopping would not help. The gradient-based fits are about 150 times slower than the closed forms for identical results."},
    {"type": "subsection", "title": "Performance metrics versus tree depth"},
    fig("depth", "Train, CV and test MSE, RMSE and R² against tree depth; dotted lines mark the required depths 3, 5, 10 and None (depth 28)."),
    {"type": "p", "text": "Training error falls to zero while test and CV error are U-shaped: they drop until depth 5 to 6, then rise as the tree memorises noise. The CV-optimal depth is 6; the test-optimal depth in hindsight is 7, but depths 5 to 8 differ negligibly on the test set."},
    {"type": "subsection", "title": "Residual diagnostics"},
    {"type": "table", "headers": ["Model", "Mean residual", "sd", "Skew", "Excess kurtosis", "rho(|res|, fitted)"], "widths": [0.22, 0.16, 0.12, 0.12, 0.2, 0.18], "size": "small",
     "rows": [[r["model"], f(r["mean residual"]), f(r["sd"]), f(r["skew"]), f(r["excess kurtosis"]), f(r["ρ(|res|, fitted)"])] for r in R["resid_stats"]]},
    fig("residuals", "Test-set diagnostics for linear, polynomial and depth-5 tree: predicted vs actual, residuals vs fitted, residual distribution and residuals over time."),
    {"type": "list", "items": [
        "The linear model's residuals show a U-shaped bias against the fitted value (under-prediction at low and high fitted values), direct evidence of non-linearity; the polynomial's are flat.",
        "The tree predicts a staircase of only 32 distinct values and never exceeds about 6.8 mg/m³.",
        "All models are heteroscedastic (residual spread roughly doubles from low to high fitted values) and heavy-tailed (excess kurtosis 4.7 to 7.1), so Gaussian error assumptions would be unreliable.",
    ]},
    {"type": "subsection", "title": "Where the errors occur"},
    fig("regime_bars", "Test RMSE and mean error (prediction − truth) by CO regime for the final models."),
    fig("hour_month", "Mean residual and RMSE by hour of day, and RMSE by month (test set)."),
    {"type": "p", "text": "Every model under-predicts the rare very-high hours (mean error −0.93 for linear, about −0.56 for the polynomial and the tuned tree). All models also under-predict in the evening (18 to 21 h) and over-predict in the early morning (06 to 08 h): hour of day is not a feature, so this is structure the sensors and weather do not carry. RMSE is two to three times larger in November to January than in summer."},
    {"type": "subsection", "title": "Robustness on a chronological split"},
    {"type": "table", "headers": ["Model", "Out-of-time test RMSE", "Random-split test RMSE", "Difference"], "widths": [0.34, 0.24, 0.24, 0.18], "size": "small",
     "rows": [[r["model"], f(r["oot_test_rmse"]), f(r["random-split test RMSE"]), f(r["degradation (oot - random)"])] for r in R["oot"]]},
    {"type": "p", "text": "With hyper-parameters fixed and the first 80% of the timeline used for training, RMSE is essentially unchanged (|difference| ≤ 0.015 for the main models) and the ranking is the same. Despite the covariate shift in `T`, `AH` and `PT08.S4`, the CO-sensor relationship is stable across seasons. This is a single split, so it supports rather than proves the conclusion."},
])

# ---------------------------------------------------------------- 5 visualization
imp = {r["index"]: r for r in R["imp_n"]}
shares = [[k, pct(v), ] for k, v in top.items()]
add("Model visualization", [
    {"type": "qa", "question": "What does the depth-3 tree look like, and which variable contributes most?",
     "answer": f"The depth-3 tree splits only on `{s2}`; that sensor ranks first in all eight importance measures (mean share {pct(s2share, 0)}), far ahead of temperature and `PT08.S1(CO)`."},
    {"type": "subsection", "title": "Decision tree with max_depth = 3"},
    fig("tree3", "Decision tree for max_depth = 3 fitted on unscaled features, so thresholds are in sensor units; value is the mean CO(GT) of the node."),
    {"type": "table", "headers": ["Leaf rule (PT08.S2 range)", "Train hours", "Mean CO", "sd", "Test hours", "Test RMSE in leaf"], "widths": [0.36, 0.13, 0.12, 0.1, 0.13, 0.16], "size": "small",
     "rows": [[r["rule"].replace("PT08.S2(NMHC) ", "").replace(" & ", " and "), f"{r['n_train']:,}", f(r["mean CO (train)"], 2), f(r["sd CO (train)"], 2), f"{r['n_test']:,}", f(r["test RMSE in leaf"])] for r in R["leaves"]]},
    {"type": "p", "text": "All seven splits use the same sensor, at thresholds from 704 to 1,587; predicted CO rises monotonically from 0.72 to 7.11 mg/m³. The highest leaf holds only 89 training hours with a large spread, so a single constant represents the extreme episodes badly."},
    {"type": "subsection", "title": "Feature importance from four methods"},
    {"type": "p", "text": "Tree impurity, permutation importance on the test set, absolute standardised linear coefficients (with bootstrap intervals) and drop-one retraining are compared, because each has blind spots; sensors are collinear, so redundant sensors share credit."},
    fig("importance", "Importance share by method (left), normalised importance of three models (centre) and linear coefficients with 95% bootstrap intervals (right)."),
    {"type": "table", "headers": ["Feature", "Mean share", "Linear coefficient", "95% interval"], "widths": [0.34, 0.18, 0.22, 0.26], "size": "small",
     "rows": [[k, pct(v), f(CO[k]["coef"]), f"[{f(CO[k]['ci_low'])}, {f(CO[k]['ci_high'])}]"] for k, v in top.items()]},
    {"type": "list", "items": [
        f"`{s2}` is rank 1 in all eight methods; temperature (`T`) and `PT08.S1(CO)` follow with mean shares of {pct(top['T'])} and {pct(top['PT08.S1(CO)'])}.",
        "`PT08.S1(CO)` matters for the linear model but not for the trees: it is redundant with `PT08.S2` (r ≈ 0.89), a collinearity effect rather than irrelevance.",
        "Only `RH` has a coefficient whose 95% interval includes zero.",
    ]},
    {"type": "subsection", "title": "Grouped permutation importance"},
    {"type": "p", "text": "Correlated columns are shuffled together on the test set, so the model cannot lean on a twin."},
    fig("grouped_perm", "Increase in test RMSE when a group of features is permuted jointly."),
    {"type": "p", "text": f"Shuffling all five sensors costs the linear model {f(next(g['Linear'] for g in R['gperm'] if g['index'] == 'All 5 sensors'), 2)} mg/m³ of RMSE, against {f(next(g['Linear'] for g in R['gperm'] if g['index'] == 'All weather (T, RH, AH)'), 2)} for all weather variables. The polynomial model shows large values for non-S2 groups: shuffling creates feature combinations that never occur, and the degree-2 model with many cross-terms reacts violently, a sign of fragility off the data manifold rather than real reliance."},
    {"type": "subsection", "title": "Partial dependence and on-data behaviour"},
    fig("pdp", "Partial dependence of the four leading features for each model."),
    fig("mplot", "Binned means of actual and predicted CO along each feature (test set): realistic inputs only."),
    {"type": "p", "text": f"The PDP of the linear model is a line and the trees follow the same line as a staircase for `PT08.S2`, so its average effect is essentially linear. The polynomial's PDP is implausible (negative CO at low S2) because a PDP evaluates the model at unrealistic combinations. On the actual data the polynomial is the most faithful (largest binned deviation {f(R['mplot_dev']['Poly d=2'], 2)} mg/m³ against {f(R['mplot_dev']['Tree tuned'], 2)} tuned tree, {f(R['mplot_dev']['Linear'], 2)} linear, {f(R['mplot_dev']['Tree depth 3'], 2)} depth-3 tree); the linear model under-estimates at both ends of the sensor ranges."},
    fig("episode", "Eight days around the highest test CO value; dots are hours not used for training."),
])

# ---------------------------------------------------------------- 6 baseline comparison
tree_keys = ["max_depth=3", "max_depth=5", "max_depth=10", "max_depth=None", "tuned (CV, 1-SE)", "pruned (ccp, CV)"]
sy = lambda t, m: R["syn"][f"{t}|{m}"]
ex = lambda t, m: R["ext"][f"{t}|{m}"]
lin_h = M[LIN]
add("Baseline comparison: linear regression vs Decision Tree", [
    {"type": "qa", "question": "Does a Decision Tree capture structure that linear regression misses?",
     "answer": f"No: linear regression is more accurate than every tree configuration (test RMSE {f(lin_h['test_rmse'])} against {f(M[T5]['test_rmse'])} for the best tree). The non-linearity in the data is real but smooth and mild, and is captured best by a degree-2 polynomial."},
    {"type": "subsection", "title": "Head-to-head"},
    {"type": "table", "headers": ["Decision Tree", "Leaves", "Test RMSE", "Test R²", "ΔRMSE vs linear", "95% interval", "Linear wins (folds of 5)"], "widths": [0.2, 0.09, 0.12, 0.1, 0.14, 0.2, 0.15], "size": "small",
     "rows": [[k, f"{int(H[k]['leaves']):,}", f(H[k]["test RMSE"]), f(H[k]["test R²"]), f(H[k]["Δ test RMSE vs Linear"]), f"[{f(H[k]['95% CI low'])}, {f(H[k]['95% CI high'])}]",
              "-" if H[k]["folds Linear wins (of 5)"] is None else str(int(H[k]["folds Linear wins (of 5)"]))] for k in tree_keys]},
    {"type": "p", "text": f"Linear regression (9 coefficients) has test RMSE {f(lin_h['test_rmse'])} and R² {f(lin_h['test_r2'])}. The gap to the best tree (depth 5) is {f(H['max_depth=5']['Δ test RMSE vs Linear'])} RMSE, with an interval that just touches zero; every other tree is clearly worse. Linear wins in 5 of 5 CV folds against every fixed-depth tree and in 4 of 5 against the tuned tree. The trees also generalise worse (train-test gap up to 0.610 against 0.013)."},
    {"type": "subsection", "title": "Is there non-linear structure?"},
    {"type": "p", "text": f"A nested F-test strongly prefers the degree-2 model over the linear one (F = {R['f_test']['F']:.1f}; the p-value underflows to zero and is optimistic under autocorrelation). The held-out evidence is more convincing: test R² rises from {f(M[LIN]['test_r2'])} to {f(M[POL]['test_r2'])} and CV RMSE falls from {f(M[LIN]['cv_rmse'])} to {f(M[POL]['cv_rmse'])}."},
    {"type": "table", "headers": ["Feature set", "Linear", f"Poly d={DEG}", "Tuned tree"], "widths": [0.46, 0.18, 0.18, 0.18], "size": "small",
     "rows": [[r["feature set"], f(r["Linear"], 3), f(r[f"Poly(d={DEG})"], 3), f(r["Tree (tuned)"], 3)] for r in R["single_feat"][:2]] + [["Relative gain from adding the other seven features", pct(R["single_feat"][2]["Linear"]), pct(R["single_feat"][2][f"Poly(d={DEG})"]), pct(R["single_feat"][2]["Tree (tuned)"])]]},
    {"type": "p", "text": f"CV RMSE with `PT08.S2` alone and with all features. A tree is good at the curvature of a single variable (alone, the tuned tree at {f(SF['Best single sensor (PT08.S2)']['Tree (tuned)'])} beats the line at {f(SF['Best single sensor (PT08.S2)']['Linear'])}) but inefficient at adding several variables up: each split uses one feature, so adding the other seven features improves the linear model by {pct(R['single_feat'][2]['Linear'], 0)} and the tree by only {pct(R['single_feat'][2]['Tree (tuned)'], 0)}."},
    {"type": "p", "text": f"As a probe, a shallow tree was fitted to the residuals of the linear model (an extension, not an assignment model): test RMSE falls from {f(M[LIN]['test_rmse'])} to {f(R['hybrid_best']['test_rmse'])}, confirming that the line leaves non-linear signal, yet it does not reach the smooth polynomial ({f(M[POL]['test_rmse'])})."},
    {"type": "subsection", "title": "What each model family can represent (synthetic data)"},
    {"type": "p", "text": "One input, noise sd 0.15 and four known truths; each model is fitted on 70% of the points and tested on the rest, and then trained on x in [0, 1] and tested on [1, 1.5]. This isolates the mechanisms behind the real-data result."},
    fig("synthetic", "Linear, degree-2 and depth-4 tree fits to four synthetic truths (test RMSE in the legends; noise floor 0.15)."),
    fig("extrapolation", "Extrapolation beyond the training range (shaded) for a linear and a quadratic truth.", 0.85),
    {"type": "table", "headers": ["Truth", "Linear", "Poly d=2", "Tree depth 4", "Linear (extrap.)", "Poly (extrap.)", "Tree (extrap.)"], "widths": [0.24, 0.12, 0.12, 0.14, 0.13, 0.13, 0.12], "size": "small",
     "rows": [[t, f(sy(t, "Linear"), 2), f(sy(t, "Poly d=2"), 2), f(sy(t, "Tree depth 4"), 2)] + ([f(ex(t, "Linear"), 2), f(ex(t, "Poly d=2"), 2), f(ex(t, "Tree depth 4"), 2)] if f"{t}|Linear" in R["ext"] else ["-", "-", "-"])
              for t in ["linear: 2x", "quadratic: 4(x−½)²", "smooth wave: sin(2πx)", "step: 1[x>½]"]]},
    {"type": "p", "text": "Trees win on abrupt or irregular shapes (sine, step); the degree-2 polynomial wins on the quadratic; a straight line fails on every non-linear truth; and outside the training range a tree freezes at its last leaf value. The air-quality relation resembles the smooth, nearly linear or quadratic case, which is why linear and polynomial models win."},
    {"type": "subsection", "title": "Where each model wins, and ablations"},
    {"type": "table", "headers": ["PT08.S2 decile", "Mean CO", "Linear", f"Poly d={DEG}", "Tree depth 5", "Tuned tree", "Best"], "widths": [0.16, 0.13, 0.13, 0.14, 0.15, 0.15, 0.14], "size": "small",
     "rows": [[r["PT08.S2(NMHC)"], f(r["mean CO"], 2), f(r["Linear"]), f(r["Poly d=2"]), f(r["Tree depth 5"]), f(r["Tree tuned"]), r["winner"].replace("Poly d=2", "Poly")] for r in R["decile"]]},
    {"type": "p", "text": "Test RMSE per decile of the dominant sensor. The polynomial is best in all ten deciles; between the two models asked about, linear beats the tuned tree in nine of ten (the tree wins only in the lowest decile, where the relation flattens)."},
    fig("feature_ablation", "Feature ablation: CV RMSE with all features, sensors only, weather only, a single sensor, and with each feature dropped."),
    {"type": "table", "headers": ["Ablation", "Change", "Model", "Δ", "Verdict"], "widths": [0.17, 0.35, 0.16, 0.1, 0.22], "size": "tiny",
     "rows": [[r["ablation"], r["change"], r["model"], f"{r['Δ']:+.3f}".replace("-", "−"), r["verdict"]] for r in R["ledger"]]},
    {"type": "p", "text": "Positive Δ means the ablated variant is worse than the reference (CV RMSE, training RMSE for the scaling row, test RMSE for the evaluation protocol); differences below 0.03 are within fold-to-fold noise. Sensors alone are as good as all features; weather alone is as bad as the mean predictor; degree 2 and tree depth 5 are the right sizes; ridge, tree criterion, pruning and the chronological split change little; skipping standardisation hurts gradient descent; a log target hurts the linear model."},
    {"type": "subsection", "title": "Scorecard"},
    {"type": "table", "headers": ["Measure", "Linear", "Tree depth 5", "Tuned tree", f"Poly d={DEG}"], "widths": [0.34, 0.16, 0.17, 0.17, 0.16], "size": "small",
     "rows": [[k, *(f(SC[k][c], 4 if "time" in k else 3) for c in ("Linear", "Tree depth 5", "Tree tuned", "Poly d=2"))] for k in SC]},
])

# ---------------------------------------------------------------- 7 discussion
q1_share = top
add("Discussion", [
    {"type": "qa", "question": "Which features were most important for predicting CO(GT)?",
     "answer": f"`{s2}` by far: rank 1 in all eight importance measures (mean share {pct(s2share, 0)}), the only feature the depth-3 tree uses, and removing it raises the CV RMSE by {f(dS2['Linear'], 3)} (linear), {f(dS2[f'Poly(d={DEG})'], 3)} (polynomial) and {f(dS2['Tree (tuned)'], 3)} (tree). Temperature and `PT08.S1(CO)` follow."},
    {"type": "list", "items": [
        f"**Linear view.** The standardised coefficient of `{s2}` is {f(CO[s2]['coef'])} (95% interval {f(CO[s2]['ci_low'])} to {f(CO[s2]['ci_high'])}), about {CO[s2]['coef'] / max(abs(v['coef']) for k, v in CO.items() if k != s2):.1f} times the next largest effect.",
        f"**Sensors versus weather.** Using only the five sensors costs just {f(FEAT_ABL['Sensors only (5)']['Linear'] - FEAT_ABL['All 8 features']['Linear'], 3)} RMSE; weather alone is no better than predicting the mean (CV RMSE {f(FEAT_ABL['Weather only (T, RH, AH)']['Linear'], 2)} against {f(M[MEAN]['cv_rmse'], 2)}). Temperature is the only weather variable with a measurable effect.",
        "**Collinearity.** `PT08.S1(CO)` and `PT08.S5(O3)` correlate strongly with `PT08.S2`; importance of redundant sensors is shared arbitrarily, so the finding should be read as the sensor cluster led by `PT08.S2`.",
    ]},
    {"type": "qa", "question": "Which performed better overall: Decision Tree or Linear Regression? Why?",
     "answer": f"Linear regression, although the margin over the best-sized tree is small (test RMSE {f(lin_h['test_rmse'])} against {f(M[T5]['test_rmse'])}). The degree-2 polynomial ({f(M[POL]['test_rmse'])}) was the best model of all."},
    {"type": "table", "headers": ["", "Linear", "Best tree (depth 5)", "Tuned tree", f"Poly d={DEG}"], "widths": [0.32, 0.16, 0.2, 0.16, 0.16], "size": "small", "rows": [
        ["Test RMSE (mg/m³)", f(M[LIN]["test_rmse"]), f(M[T5]["test_rmse"]), f(M[TT]["test_rmse"]), f(M[POL]["test_rmse"])],
        ["Test R²", f(M[LIN]["test_r2"]), f(M[T5]["test_r2"]), f(M[TT]["test_r2"]), f(M[POL]["test_r2"])],
        ["CV RMSE", f(M[LIN]["cv_rmse"]), f(M[T5]["cv_rmse"]), f(M[TT]["cv_rmse"]), f(M[POL]["cv_rmse"])],
        ["Train-test RMSE gap", f(M[LIN]["rmse_gap (test-train)"]), f(M[T5]["rmse_gap (test-train)"]), f(M[TT]["rmse_gap (test-train)"]), f(M[POL]["rmse_gap (test-train)"])],
        ["Parameters / leaves", "9", f"{int(M[T5]['complexity'])}", f"{int(M[TT]['complexity'])}", "45"]]},
    {"type": "list", "items": [
        f"**Strength of the evidence.** The best tree is {f(H['max_depth=5']['Δ test RMSE vs Linear'])} RMSE worse than linear (95% interval {f(H['max_depth=5']['95% CI low'])} to {f(H['max_depth=5']['95% CI high'])}; linear better in {pct(H['max_depth=5']['P(tree worse)'], 0)} of resamples) and linear wins 5 of 5 CV folds; against other tree depths the gap is larger and its interval excludes zero.",
        "**A smooth, additive signal.** CO rises smoothly and almost linearly with `PT08.S2`, with additive contributions from other sensors and temperature. A line represents this with 9 numbers; a tree approximates it with a staircase of constant steps (the depth-3 tree is an 8-step function of one sensor), losing precision inside every step.",
        f"**Trees add variables up inefficiently.** With `PT08.S2` alone the tuned tree beats the line, but adding the other seven features improves linear regression by {pct(R['single_feat'][2]['Linear'], 0)} and the tree by only {pct(R['single_feat'][2]['Tree (tuned)'], 0)}.",
        "**Bias and variance.** Shallow trees under-fit, deep trees fit the noise (depth None: training R² = 1.00, test R² 0.817); linear regression has low variance and, for this near-linear signal, little bias.",
        "**Where trees do win.** In the lowest sensor decile and on abrupt shapes (synthetic study); they need no scaling and give readable rules.",
        "**Linear versus non-linear patterns.** The linear model's residuals show a slight U-shape, a quadratic term gives a real gain, and a tree fitted to the linear residuals also helps: the non-linearity is real but smooth and mild, captured best by the degree-2 polynomial, second best by a linear backbone with tree corrections, and least efficiently by a stand-alone staircase.",
    ]},
    {"type": "subsection", "title": "Limitations and next steps"},
    {"type": "list", "items": [
        "The random 80/20 split on autocorrelated hourly data makes test scores somewhat optimistic; this was mitigated by week-grouped CV, week-clustered bootstrap intervals and a chronological check (a single split).",
        "One site, about 13 months and one sensor array; the `PT08.S4` sensor drifts over time, so a deployed model would need recalibration.",
        "Differences among the top models are small relative to uncertainty; coefficients and single-feature importances of correlated sensors are unreliable; the polynomial model is fragile on unrealistic feature combinations.",
        "Next steps: hour-of-day and day-of-week features, ensembles (random forest, gradient boosting), quantile or heteroscedastic regression for extreme episodes, time-aware nested CV, sensor-drift monitoring.",
    ]},
])

# ---------------------------------------------------------------- appendix
chk = [["Data preprocessing and handling of missing values", "Section 2"], ["Load the data; choose target; select features; split 80/20", "Section 2"],
       ["Linear and polynomial regression from scratch, degree choice, scikit-learn cross-check", "Section 3, Appendix A"], ["Decision Tree training with max_depth 3, 5, 10, None", "Section 3"],
       ["MSE, RMSE and R² for all models", "Section 4"], ["Performance metrics versus epoch; versus tree depth", "Section 4"],
       ["Decision tree plot for max_depth = 3; feature importances", "Section 5"], ["Linear regression compared with the Decision Tree; linear vs non-linear patterns", "Section 6"],
       ["Short discussion: important features; Decision Tree vs Linear Regression", "Section 7"], ["Neat, commented code", "Appendix A and the notebook"]]
add("Appendix A: code excerpts", [
    {"type": "qa", "question": "Where is the code?", "answer": "The complete code and outputs are in the Jupyter notebook `src/notebooks/air_quality_regression.ipynb` (and its PDF export `docs/report/air_quality_regression_notebook.pdf`); the central from-scratch parts are quoted here, copied from the notebook cells."},
    {"type": "subsection", "title": "A.1 Evaluation metrics from scratch", "numbered": False}, {"type": "code", "text": code("metrics")},
    {"type": "subsection", "title": "A.2 Linear regression from scratch (normal equation, gradient descent, Adam)", "numbered": False}, {"type": "code", "text": code("linear_scratch")},
    {"type": "subsection", "title": "A.3 Polynomial features and polynomial regression", "numbered": False}, {"type": "code", "text": code("poly_scratch")},
    {"type": "subsection", "title": "A.4 Degree selection by week-grouped cross-validation", "numbered": False}, {"type": "code", "text": code("poly_cv")},
    {"type": "subsection", "title": "A.5 Decision Tree depth experiment", "numbered": False}, {"type": "code", "text": code("tree_depths")},
    {"type": "subsection", "title": "A.6 Week-clustered bootstrap", "numbered": False}, {"type": "code", "text": code("bootstrap")},
])
add("Appendix B: deliverables and reproducibility", [
    {"type": "table", "headers": ["Requirement of the brief", "Where"], "widths": [0.78, 0.22], "size": "small", "rows": chk},
    {"type": "p", "text": f"The notebook runs top to bottom from the raw CSV in about 30 seconds (Python {R['py']}, numpy {R['versions']['numpy']}, pandas {R['versions']['pandas']}, scikit-learn {R['versions']['sklearn']}, random seed 42). This report is generated from the executed notebook by `src/report/`."},
])

spec["references"] = [
    "S. De Vito, E. Massera, M. Piga, L. Martinotto and G. Di Francia, “On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario,” //Sensors and Actuators B: Chemical//, vol. 129, no. 2, pp. 750-757, 2008. Dataset: “Air Quality,” UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/360/air+quality.",
    "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” //Journal of Machine Learning Research//, vol. 12, pp. 2825-2830, 2011.",
    "T. Hastie, R. Tibshirani and J. Friedman, //The Elements of Statistical Learning//, 2nd ed. Springer, 2009.",
    "L. Breiman, J. Friedman, R. Olshen and C. Stone, //Classification and Regression Trees//. Wadsworth, 1984.",
    "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in //Proc. ICLR//, 2015.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="JSON with the cover details; keep it out of git")
    ap.add_argument("--save", action="store_true", help="remember the stable answers (name, ID, degree) for next time")
    a = ap.parse_args()
    (BUILD / "report_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    cmd = [sys.executable, str(BUILDER), str(BUILD / "report_spec.json"), "--profile", "msc-assignment", "--answers", a.answers,
           "--out", str(BUILD / "out"), "--name", "air_quality_regression_report"] + (["--save"] if a.save else [])
    done = subprocess.run(cmd, text=True)
    if done.returncode == 0:
        shutil.copy(BUILD / "out/air_quality_regression_report.pdf", ROOT / "docs/report/air_quality_regression_report.pdf")
        print("copied to docs/report/air_quality_regression_report.pdf")
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())

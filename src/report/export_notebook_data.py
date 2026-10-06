"""Export what the report needs from the executed notebook, so no number or figure is typed or redrawn by hand.

    .venv/bin/python src/report/export_notebook_data.py

Writes to src/report/build/ (git-ignored):
  figures/<name>.png   figures taken from the saved notebook outputs
  results.json         result tables, computed by re-running the notebook with one extra final cell
  code/<name>.py       code excerpts quoted in the report appendix (read from the notebook cells)
"""
import base64
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parents[1] / "src/notebooks/air_quality_regression.ipynb"
BUILD = HERE / "build"

# figure name -> (substring identifying the notebook cell, index of the image among that cell's outputs)
FIGURES = {
    "missingness": ("Missingness over time (weekly)", 0),
    "class_balance": ("Class-wise balance of CO(GT) regimes", 0),
    "correlations": ('ax[0].set_title("Pearson (linear)")', 0),
    "lr_sweep": ("Learning-rate sweep (batch GD)", 0),
    "poly_degree": ("Bias-variance: degree selection", 0),
    "epochs": ("Metrics vs. epoch (dotted = closed-form optimum)", 0),
    "depth": ('a.set_title(f"{lab} vs. tree depth")', 0),
    "forest": ("Test RMSE (mg/m³) with 95% week-bootstrap CI", 0),
    "residuals": ("residuals vs fitted", 0),
    "regime_bars": ("Test RMSE by CO regime", 0),
    "hour_month": ("Mean residual by hour (actual", 0),
    "tree3": ("Decision Tree Regressor, max_depth = 3", 0),
    "importance": ("Importance share by method", 0),
    "grouped_perm": ("Grouped permutation importance (test set)", 0),
    "pdp": ("average predicted CO(GT)", 0),
    "mplot": ("On-data binned means", 0),
    "feature_ablation": ("Feature ablation (group-CV, training set)", 0),
    "synthetic": ("Synthetic truths, test RMSE in legend", 0),
    "extrapolation": ("Synthetic truths, test RMSE in legend", 1),
    "episode": ("Eight days around the highest test CO value", 0),
}
CODE = {   # name -> (substring identifying the cell, start marker, end marker or None)
    "metrics": ("# ---- evaluation metrics implemented from scratch", "# ---- evaluation metrics implemented from scratch", "_a, _b = "),
    "linear_scratch": ("class LinearRegressionScratch", "class LinearRegressionScratch", "# ---- unit tests of the implementation ----"),
    "poly_scratch": ("class PolynomialFeaturesScratch", "class PolynomialFeaturesScratch", "# ---- verify the expansion against sklearn ----"),
    "poly_cv": ("def poly_cv_curve", "def poly_cv_curve", "t0 = time.perf_counter()"),
    "tree_depths": ("REQ_DEPTHS = [3, 5, 10, None]", "REQ_DEPTHS = [3, 5, 10, None]", "req = pd.DataFrame"),
    "bootstrap": ("def boot_dist(name)", "test_weeks = ", "BOOT = {nm"),
}

DUMP = '''
import json as _json
def _t(df, nd=6):
    df = df.copy()
    if not isinstance(df.index, __import__("pandas").RangeIndex): df = df.reset_index()
    df.columns = [str(c) if not isinstance(c, tuple) else " ".join(map(str, c)) for c in df.columns]
    return _json.loads(df.to_json(orient="records", double_precision=nd))
_out = {
 "n": {"raw_rows": int(len(pd.read_csv(DATA_PATH, sep=";", decimal=","))), "clean_rows": int(len(df)), "model_rows": int(len(data)), "train": int(len(X_train)), "test": int(len(X_test))},
 "miss": _t(miss), "bal": _t(bal), "split_bal": _t(sb), "feat_stats": _t(fs[["mean", "std", "min", "max", "skew"]]), "vif": _t(vif.to_frame()),
 "target_rel": _t(rel), "lr_tbl": _t(lr_tbl), "lr_stable": float(lr_stable), "lr_lin": float(LR_LIN),
 "master": _t(MASTER.drop(columns=["family"], errors="ignore").reset_index(drop=True)),
 "verdict": _t(pd.DataFrame({lbl: {**compare_pair(a, b), "scratch test RMSE": RESULTS[a]["test_rmse"], "sklearn test RMSE": RESULTS[b]["test_rmse"]} for lbl, a, b in pairs}).T), "poly_unreg": _t(unreg), "poly_best": [int(DEG_R), float(ALPHA_R)], "deg_star": int(DEG_STAR),
 "tree_sweep": _t(TREE_SWEEP), "tree_best": TREE_BEST, "ccp_best": float(CCP_BEST), "ridge_best": float(RIDGE_CV["cv_rmse"].idxmin()), "ridge_cv_min": float(RIDGE_CV["cv_rmse"].min()), "ridge_cv_ols": float(RIDGE_CV.iloc[0]["cv_rmse"]),
 "boot": _t(BOOT_TBL), "paired": _t(PAIRED), "fold_rmse": _t(FOLD_RMSE), "conv": _t(conv), "req_depth": _t(req_tbl), "resid_stats": _t(RESID_STATS.drop(columns=["sd low/mid/high fitted"])),
 "regime": _t(REGIME), "oot": _t(OOT), "lc": {k: _t(v) for k, v in LC.items()},
 "leaves": _t(LEAVES), "imp_n": _t(IMP_N), "imp": _t(IMP), "rank": _t(RANK), "gperm": _t(GPERM), "top_share": {k: float(v) for k, v in top.items()},
 "coef": _t(COEF_BOOT), "h2h": _t(H2H), "topterms": _t(TOPTERMS), "single_feat": _t(sf), "hybrid": _t(HYB), "hybrid_best": HYBRID,
 "f_test": {"F": float(F), "p": float(pval)}, "decile": _t(DECILE), "extrap": _t(EXTRAP), "ledger": _t(LEDGER), "score": _t(SCORE.drop(columns=["best"])),
 "abl_feat": _t(tbl), "abl_log": _t(ABL_LOG), "abl_crit": _t(ABL_CRIT), "abl_scale": _t(ABL_SCALE),
 "syn": {f"{a}|{b}": float(v) for (a, b), v in SYN.items()}, "ext": {f"{a}|{b}": float(v) for (a, b), v in EXT.items()},
 "pdp_range": {f"{a}|{b}": float(__import__("numpy").ptp(PDP[(a, b)])) for (a, b) in PDP}, "mplot_dev": {lbl: float(__import__("numpy").mean([__import__("numpy").abs(MPLOT[(f, lbl)] - MPLOT[(f, "actual")]).max() for f in top4])) for lbl in PDP_MODELS},
 "gd_epochs": {"linear": int(epochs_to_tol(HISTORY[N_LIN_GD])), "poly": int(epochs_to_tol(HISTORY[N_POLY_GD]))},
 "fit_time": {k: float(v["fit_time_s"]) for k, v in RESULTS.items()},
 "weeks_test": int(len(members)), "groups": int(len(__import__("numpy").unique(groups))), "py": platform.python_version(),
 "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
}
open(r"@@OUT@@", "w").write(_json.dumps(_out, ensure_ascii=False))
'''


def cell_with(nb, needle):
    for c in nb.cells:
        if c.cell_type == "code" and needle in c.source:
            return c
    raise SystemExit(f"notebook cell not found for: {needle!r}")


def main():
    (BUILD / "figures").mkdir(parents=True, exist_ok=True)
    (BUILD / "code").mkdir(exist_ok=True)
    nb = nbformat.read(NOTEBOOK, 4)
    for name, (needle, idx) in FIGURES.items():
        cells = [c for c in nb.cells if c.cell_type == "code" and needle in (c.source + "".join(str(o.get("data", {}).get("text/plain", "")) for o in c.outputs))]
        if not cells:
            cells = [cell_with(nb, needle)]
        imgs = [o.data["image/png"] for c in cells[:1] for o in c.outputs if o.output_type == "display_data" and "image/png" in o.get("data", {})]
        if name in ("hour_month",):
            idx = 1 if len(imgs) > 1 else 0          # second figure of the error-breakdown cell
        if name == "regime_bars":
            idx = 0
        (BUILD / "figures" / f"{name}.png").write_bytes(base64.b64decode(imgs[idx]))
    for name, (needle, start, end) in CODE.items():
        src = cell_with(nb, needle).source
        a = src.index(start); b = src.index(end, a) if end else len(src)
        (BUILD / "code" / f"{name}.py").write_text(src[a:b].rstrip() + "\n")
    # results: re-run the notebook with a final cell that dumps the tables
    out = BUILD / "results.json"
    probe = nbformat.v4.new_notebook(); probe.metadata = nb.metadata
    probe.cells = [c for c in nb.cells] + [nbformat.v4.new_code_cell("import platform\n" + DUMP.replace("@@OUT@@", str(out)))]
    for c in probe.cells:
        if c.cell_type == "code":
            c.outputs = []; c.execution_count = None
    NotebookClient(probe, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    print("wrote", out, "and", len(list((BUILD / 'figures').glob('*.png'))), "figures,", len(list((BUILD / 'code').glob('*.py'))), "code excerpts")


if __name__ == "__main__":
    main()

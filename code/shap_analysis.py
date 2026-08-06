"""
shap_analysis.py

Step 3 of the experimental design: adds the explainability layer (SHAP) on top of
the RandomForest selector already validated in train_eval.py, and computes the
stability of meta-feature importance attributions across the 10 ASlib folds.

Requires that aslib_loader.py has already been run (uses data/<scenario>/dataset.csv).

Usage:
    python shap_analysis.py                      # run the 3 default scenarios
    python shap_analysis.py --scenario CSP-2010   # run only one

Output (per scenario), in results/<scenario>/:
    - shap_importance_by_fold.csv   mean |SHAP| importance per feature and fold
    - shap_top_features.csv         top-10 features by mean SHAP importance (Table 4)
    - stability.csv                 rho_stab: mean Spearman correlation across folds (Table 5)
    - shap_summary_plot.png         SHAP summary plot for the representative fold (Figure 3)
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

DEFAULT_SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]
RANDOM_STATE = 42
TOP_K = 10


def mean_abs_shap_per_feature(shap_values, feature_cols):
    """
    Normalizes TreeExplainer output (whose shape varies with binary/
    multiclass classification and SHAP version) into a per-feature
    importance vector: mean |SHAP| over instances and, if applicable,
    over classes.
    """
    arr = np.asarray(shap_values.values) if hasattr(shap_values, "values") else np.asarray(shap_values)

    if isinstance(shap_values, list):
        # List of arrays (n_samples, n_features), one per class
        stacked = np.stack([np.abs(a) for a in shap_values], axis=0)  # (n_classes, n_samples, n_features)
        importance = stacked.mean(axis=(0, 1))
    elif arr.ndim == 3:
        # (n_samples, n_features, n_classes) -- typical shape in SHAP >= 0.4x for multiclass
        importance = np.abs(arr).mean(axis=(0, 2))
    else:
        # (n_samples, n_features) -- binary case
        importance = np.abs(arr).mean(axis=0)

    assert importance.shape[0] == len(feature_cols), \
        f"Importance dimension ({importance.shape[0]}) does not match # features ({len(feature_cols)})"
    return importance


def run_scenario(scenario: str, data_root: Path, results_root: Path):
    print(f"[{scenario}] loading dataset...")
    dataset = pd.read_csv(data_root / scenario / "dataset.csv")

    algo_cols = [c for c in dataset.columns if c.startswith("perf__")]
    non_feature_cols = set(algo_cols) | {"instance_id", "best_algorithm", "cv_fold"}
    feature_cols = [c for c in dataset.columns if c not in non_feature_cols]

    folds = sorted(dataset["cv_fold"].dropna().unique())
    importance_by_fold = {}
    representative_shap = None  # keep one for Figure 3

    for fold in folds:
        train_df = dataset[dataset["cv_fold"] != fold].reset_index(drop=True)
        test_df = dataset[dataset["cv_fold"] == fold].reset_index(drop=True)

        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df["best_algorithm"].values

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        label_enc = LabelEncoder()
        y_train_enc = label_enc.fit_transform(y_train)

        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train_imp, y_train_enc)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_imp)  # modern shap API (Explanation object)

        importance = mean_abs_shap_per_feature(shap_values, feature_cols)
        importance_by_fold[int(fold)] = importance

        if representative_shap is None:
            representative_shap = (shap_values, X_test_imp, feature_cols, int(fold))

        print(f"  fold {int(fold)}/{len(folds)} -> SHAP computed")

    # --- Table: importance per feature and fold ---
    imp_df = pd.DataFrame(importance_by_fold, index=feature_cols)
    imp_df.index.name = "feature"
    out_dir = results_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(out_dir / "shap_importance_by_fold.csv")

    # --- Table 4: top-K features by mean SHAP importance ---
    mean_importance = imp_df.mean(axis=1).sort_values(ascending=False)
    top_features = mean_importance.head(TOP_K).reset_index()
    top_features.columns = ["feature", "mean_abs_shap"]
    top_features["scenario"] = scenario
    top_features.to_csv(out_dir / "shap_top_features.csv", index=False)

    # --- Table 5: stability (rho_stab) across folds ---
    # Feature importance ranking, per fold
    rankings = imp_df.rank(ascending=False, axis=0)
    fold_ids = list(rankings.columns)
    pairwise_corrs = []
    for f1, f2 in itertools.combinations(fold_ids, 2):
        rho, _ = spearmanr(rankings[f1], rankings[f2])
        pairwise_corrs.append(rho)
    rho_stab = float(np.mean(pairwise_corrs))
    rho_stab_std = float(np.std(pairwise_corrs))

    stability_df = pd.DataFrame([{
        "scenario": scenario,
        "rho_stab_mean": rho_stab,
        "rho_stab_std": rho_stab_std,
        "n_fold_pairs": len(pairwise_corrs),
    }])
    stability_df.to_csv(out_dir / "stability.csv", index=False)

    # --- Figure 3: SHAP summary plot for the representative fold ---
    shap_values, X_test_imp, feat_cols, fold_id = representative_shap
    plt.figure()
    try:
        # If multiclass, average |SHAP| over classes for a readable summary plot
        vals = np.asarray(shap_values.values)
        if vals.ndim == 3:
            vals_for_plot = np.abs(vals).mean(axis=2)
            shap.summary_plot(vals_for_plot, X_test_imp, feature_names=feat_cols, show=False)
        else:
            shap.summary_plot(shap_values, X_test_imp, feature_names=feat_cols, show=False)
        plt.title(f"{scenario} - SHAP summary (fold {fold_id})")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_plot.png", dpi=150)
    finally:
        plt.close()

    print(f"[{scenario}] OK -> rho_stab = {rho_stab:.3f} (+/- {rho_stab_std:.3f})")
    print(f"  top-5 features: {list(top_features['feature'].head(5))}")
    print()

    return top_features, stability_df


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability layer + attribution stability")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else DEFAULT_SCENARIOS
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    all_top, all_stab = [], []
    for scenario in scenarios:
        try:
            top_features, stability_df = run_scenario(scenario, data_root, results_root)
            all_top.append(top_features)
            all_stab.append(stability_df)
        except Exception as e:
            print(f"[{scenario}] ERROR: {e}", file=sys.stderr)
            raise

    if all_top:
        pd.concat(all_top, ignore_index=True).to_csv(
            results_root / "shap_top_features_all_scenarios.csv", index=False)
    if all_stab:
        pd.concat(all_stab, ignore_index=True).to_csv(
            results_root / "stability_all_scenarios.csv", index=False)
        print("Combined stability summary -> results/stability_all_scenarios.csv")


if __name__ == "__main__":
    main()

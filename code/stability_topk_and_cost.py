"""
stability_topk_and_cost.py

Two camera-ready additions bundled together since both reuse the same
per-fold SHAP computation:

  1. Top-k Jaccard stability (response to reviewer Comment #3): the
     mean pairwise Jaccard overlap between each pair of folds' top-5
     and top-10 most important features, complementing the aggregate
     rho_stab already computed by shap_analysis.py.

  2. SHAP computational cost (response to reviewer Comment #1, RQ1
     reformulation): wall-clock time for TreeExplainer vs. plain
     prediction, on one representative fold per scenario.

Requires that shap_analysis.py has already produced
results/<scenario>/shap_importance_by_fold.csv (used for the Jaccard
computation) and that aslib_loader.py has produced
data/<scenario>/dataset.csv (used for the timing measurement, which
retrains a fresh RandomForest to get a clean wall-clock reading).

Usage:
    python stability_topk_and_cost.py

Output, in results/:
    - topk_jaccard_stability.csv     Jaccard@5 / Jaccard@10 per scenario
    - shap_computational_cost.csv    predict vs. SHAP timing per scenario
"""

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]
RANDOM_STATE = 42
REPRESENTATIVE_FOLD = 1

DATA_ROOT = Path("data")
RESULTS_ROOT = Path("results")


def compute_topk_jaccard():
    rows = []
    for scenario in SCENARIOS:
        imp = pd.read_csv(RESULTS_ROOT / scenario / "shap_importance_by_fold.csv", index_col=0)
        fold_ids = list(imp.columns)
        for k in [5, 10]:
            jaccards = []
            for f1, f2 in itertools.combinations(fold_ids, 2):
                top1 = set(imp[f1].sort_values(ascending=False).head(k).index)
                top2 = set(imp[f2].sort_values(ascending=False).head(k).index)
                jaccards.append(len(top1 & top2) / len(top1 | top2))
            rows.append({
                "scenario": scenario, "k": k,
                "jaccard_mean": np.mean(jaccards), "jaccard_std": np.std(jaccards),
            })
    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "topk_jaccard_stability.csv"
    df.to_csv(out, index=False)
    print(f"OK -> {out}")
    print(df.to_string(index=False))


def compute_shap_cost():
    rows = []
    for scenario in SCENARIOS:
        dataset = pd.read_csv(DATA_ROOT / scenario / "dataset.csv")
        algo_cols = [c for c in dataset.columns if c.startswith("perf__")]
        non_feature_cols = set(algo_cols) | {"instance_id", "best_algorithm", "cv_fold"}
        feature_cols = [c for c in dataset.columns if c not in non_feature_cols]

        train_df = dataset[dataset["cv_fold"] != REPRESENTATIVE_FOLD].reset_index(drop=True)
        test_df = dataset[dataset["cv_fold"] == REPRESENTATIVE_FOLD].reset_index(drop=True)
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df["best_algorithm"].values

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        label_enc = LabelEncoder()
        y_train_enc = label_enc.fit_transform(y_train)

        t0 = time.perf_counter()
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train_imp, y_train_enc)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.predict(X_test_imp)
        predict_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        explainer = shap.TreeExplainer(model)
        explainer(X_test_imp)
        shap_time = time.perf_counter() - t0

        rows.append({
            "scenario": scenario, "n_test": len(X_test_imp), "n_features": len(feature_cols),
            "train_time_s": train_time, "predict_time_s": predict_time, "shap_time_s": shap_time,
            "shap_overhead_ratio": shap_time / predict_time if predict_time > 0 else None,
            "shap_time_per_instance_ms": (shap_time / len(X_test_imp)) * 1000,
        })
        print(f"{scenario}: predict={predict_time*1000:.1f}ms (fold), "
              f"SHAP={shap_time:.2f}s ({(shap_time/len(X_test_imp))*1000:.1f} ms/instance)")

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "shap_computational_cost.csv"
    df.to_csv(out, index=False)
    print(f"OK -> {out}")


if __name__ == "__main__":
    compute_topk_jaccard()
    print()
    compute_shap_cost()

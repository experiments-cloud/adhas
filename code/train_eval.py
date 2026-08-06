"""
train_eval.py

Step 2 of the experimental design: trains per-instance selectors (Random Forest,
XGBoost) over the official ASlib folds and compares them against the standard
baselines (Single Best Solver, Virtual Best Solver/Oracle, random selection).

Requires that aslib_loader.py has already been run (uses data/<scenario>/dataset.csv
and data/<scenario>/raw/description.txt).

Usage:
    python train_eval.py                      # run the 3 default scenarios
    python train_eval.py --scenario CSP-2010   # run only one

Output (per scenario), in results/<scenario>/:
    - fold_results.csv     one row per (fold, method) with accuracy and PAR10
    - summary.csv          mean +/- std per method, ready for Table 2 of the paper
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DEFAULT_SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]
RANDOM_STATE = 42


def get_cutoff_time(scenario: str, data_root: Path) -> float:
    """Reads algorithm_cutoff_time from description.txt (cached by aslib_loader.py)."""
    desc_path = data_root / scenario / "raw" / "description.txt"
    text = desc_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"algorithm_cutoff_time:\s*([0-9.]+)", text)
    if not m:
        raise ValueError(f"algorithm_cutoff_time not found in {desc_path}")
    return float(m.group(1))


def par10(runtimes: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies the standard PAR10 transformation: runtime if <= cutoff, 10*cutoff otherwise."""
    r = runtimes.copy().astype(float)
    timeout_mask = (r >= cutoff) | np.isnan(r)
    r[timeout_mask] = 10.0 * cutoff
    return r


def evaluate_fold(train_df, test_df, feature_cols, algo_cols, cutoff, models):
    """Trains and evaluates all methods (models + baselines) for one fold."""
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df["best_algorithm"].values
    y_test = test_df["best_algorithm"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    results = []

    # --- Baseline: Single Best Solver (chosen using training data) ---
    train_perf = train_df[algo_cols]
    train_par10 = train_perf.apply(lambda col: par10(col.values, cutoff), axis=0)
    sbs_algo = train_par10.mean(axis=0).idxmin()
    sbs_runtimes = test_df[sbs_algo].values
    results.append({
        "method": "SBS",
        "accuracy": float((y_test == sbs_algo.replace("perf__", "")).mean())
        if sbs_algo.startswith("perf__") else np.nan,
        "par10_mean": par10(sbs_runtimes, cutoff).mean(),
    })

    # --- Baseline: Virtual Best Solver / Oracle ---
    test_perf = test_df[algo_cols].values
    vbs_runtimes = np.nanmin(test_perf, axis=1)
    results.append({
        "method": "VBS_Oracle",
        "accuracy": 1.0,
        "par10_mean": par10(vbs_runtimes, cutoff).mean(),
    })

    # --- Baseline: random selection (expected value = average over algorithms) ---
    random_par10 = par10(test_perf.flatten(), cutoff).reshape(test_perf.shape).mean(axis=1)
    results.append({
        "method": "Random",
        "accuracy": 1.0 / len(algo_cols),
        "par10_mean": random_par10.mean(),
    })

    # --- ML selection models ---
    label_enc = LabelEncoder()
    y_train_enc = label_enc.fit_transform(y_train)

    for name, model in models.items():
        model.fit(X_train_imp, y_train_enc)
        y_pred_enc = model.predict(X_test_imp)
        y_pred = label_enc.inverse_transform(y_pred_enc)

        chosen_cols = ["perf__" + a for a in y_pred]
        chosen_runtimes = np.array([
            test_df.iloc[i][chosen_cols[i]] for i in range(len(test_df))
        ])
        results.append({
            "method": name,
            "accuracy": float((y_pred == y_test).mean()),
            "par10_mean": par10(chosen_runtimes, cutoff).mean(),
        })

    return results


def run_scenario(scenario: str, data_root: Path, results_root: Path):
    print(f"[{scenario}] loading dataset...")
    dataset = pd.read_csv(data_root / scenario / "dataset.csv")
    cutoff = get_cutoff_time(scenario, data_root)

    algo_cols = [c for c in dataset.columns if c.startswith("perf__")]
    non_feature_cols = set(algo_cols) | {"instance_id", "best_algorithm", "cv_fold"}
    feature_cols = [c for c in dataset.columns if c not in non_feature_cols]

    all_results = []
    folds = sorted(dataset["cv_fold"].dropna().unique())
    for fold in folds:
        train_df = dataset[dataset["cv_fold"] != fold].reset_index(drop=True)
        test_df = dataset[dataset["cv_fold"] == fold].reset_index(drop=True)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, random_state=RANDOM_STATE,
                eval_metric="mlogloss", verbosity=0
            ),
        }

        fold_results = evaluate_fold(train_df, test_df, feature_cols, algo_cols, cutoff, models)
        for r in fold_results:
            r["fold"] = int(fold)
            r["scenario"] = scenario
        all_results.extend(fold_results)
        print(f"  fold {int(fold)}/{len(folds)} done")

    fold_df = pd.DataFrame(all_results)
    out_dir = results_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "fold_results.csv", index=False)

    summary = fold_df.groupby("method").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        par10_mean=("par10_mean", "mean"),
        par10_std=("par10_mean", "std"),
    ).reset_index()
    summary["scenario"] = scenario
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"[{scenario}] OK -> {out_dir}/summary.csv")
    print(summary.to_string(index=False))
    print()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate per-instance selectors on ASlib")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else DEFAULT_SCENARIOS
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for scenario in scenarios:
        try:
            summary = run_scenario(scenario, data_root, results_root)
            all_summaries.append(summary)
        except Exception as e:
            print(f"[{scenario}] ERROR: {e}", file=sys.stderr)
            raise

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(results_root / "summary_all_scenarios.csv", index=False)
        print(f"Combined summary -> {results_root}/summary_all_scenarios.csv")


if __name__ == "__main__":
    main()

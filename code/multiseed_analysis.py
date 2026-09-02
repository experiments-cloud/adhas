"""
multiseed_analysis.py

Camera-ready addition (response to reviewer Comment #7): repeats
train_eval.py's training with multiple random seeds per fold to (a)
separate data-partition variability from training stochasticity, and
(b) run a proper TOST equivalence test on the Random Forest vs.
XGBoost comparison, replacing the earlier "failed to reject the null"
framing with a formal equivalence bound.

Requires that aslib_loader.py has already been run (uses
data/<scenario>/dataset.csv) and that train_eval.py has already
produced results/<scenario>/fold_results.csv (used for the SBS values,
which are deterministic and not re-run here).

Usage:
    python multiseed_analysis.py                  # all 3 scenarios, all folds
    python multiseed_analysis.py --scenario CSP-2010
    python multiseed_analysis.py --scenario SAT12-ALL --fold-start 1 --fold-end 5
        (SAT12-ALL is the slowest scenario; splitting by fold range
        avoids long single invocations)

Output, in results/:
    - multiseed_<SCENARIO>[_f<start>-<end>].csv   per-run PAR10/accuracy
    - multiseed_all.csv                            all scenarios combined
                                                     (run with --combine
                                                     after all scenarios
                                                     are done)
    - multiseed_statistical_tests.csv              Friedman/CD/TOST summary
    - multiseed_nemenyi.csv, multiseed_wilcoxon.csv
    - table3_multiseed_std.csv                     mean +/- SD for Table 3
    - friedman_3block.csv                          conservative 3-block
                                                     Friedman (Section 6)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]
SEEDS = [42, 123, 2024]
METHODS = ["SBS", "RandomForest", "XGBoost"]


def get_cutoff_time(scenario, data_root):
    desc_path = Path(data_root) / scenario / "raw" / "description.txt"
    text = desc_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"algorithm_cutoff_time:\s*([0-9.]+)", text)
    return float(m.group(1))


def par10(runtimes, cutoff):
    r = runtimes.copy().astype(float)
    timeout_mask = (r >= cutoff) | np.isnan(r)
    r[timeout_mask] = 10.0 * cutoff
    return r


def run_multiseed(scenario, data_root, results_root, fold_start, fold_end):
    """Step 1: retrain RandomForest/XGBoost with 3 seeds per fold."""
    dataset = pd.read_csv(Path(data_root) / scenario / "dataset.csv")
    cutoff = get_cutoff_time(scenario, data_root)
    algo_cols = [c for c in dataset.columns if c.startswith("perf__")]
    non_feature_cols = set(algo_cols) | {"instance_id", "best_algorithm", "cv_fold"}
    feature_cols = [c for c in dataset.columns if c not in non_feature_cols]
    folds = sorted(dataset["cv_fold"].dropna().unique())
    folds = [f for f in folds if fold_start <= f <= fold_end]

    rows = []
    for fold in folds:
        train_df = dataset[dataset["cv_fold"] != fold].reset_index(drop=True)
        test_df = dataset[dataset["cv_fold"] == fold].reset_index(drop=True)
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        y_train = train_df["best_algorithm"].values
        y_test = test_df["best_algorithm"].values

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        label_enc = LabelEncoder()
        y_train_enc = label_enc.fit_transform(y_train)

        for seed in SEEDS:
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                "XGBoost": XGBClassifier(n_estimators=200, random_state=seed, eval_metric="mlogloss", verbosity=0),
            }
            for name, model in models.items():
                model.fit(X_train_imp, y_train_enc)
                y_pred_enc = model.predict(X_test_imp)
                y_pred = label_enc.inverse_transform(y_pred_enc)
                chosen_cols = ["perf__" + a for a in y_pred]
                chosen_runtimes = np.array([test_df.iloc[i][chosen_cols[i]] for i in range(len(test_df))])
                rows.append({
                    "scenario": scenario, "fold": int(fold), "seed": seed, "method": name,
                    "accuracy": float((y_pred == y_test).mean()),
                    "par10_mean": par10(chosen_runtimes, cutoff).mean(),
                })
        print(f"  fold {int(fold)} done", flush=True)

    df = pd.DataFrame(rows)
    out = Path(results_root) / f"multiseed_{scenario}_f{fold_start}-{fold_end}.csv"
    df.to_csv(out, index=False)
    print(f"OK -> {out}")
    return df


def combine_and_analyze(results_root):
    """Step 2: combine per-scenario multiseed files, run Table 3 std devs,
    the n=90 Friedman/Nemenyi/Wilcoxon/CD/TOST analysis, and the
    conservative 3-block Friedman check reported in Section 6."""
    results_root = Path(results_root)
    exclude = {"multiseed_all.csv", "multiseed_nemenyi.csv",
               "multiseed_statistical_tests.csv", "multiseed_wilcoxon.csv"}
    files = [f for f in results_root.glob("multiseed_*.csv") if f.name not in exclude]
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(results_root / "multiseed_all.csv", index=False)

    # --- Table 3: mean +/- SD over 3 seeds x 10 folds ---
    rows = []
    for s in SCENARIOS:
        for method in ["RandomForest", "XGBoost"]:
            sub = df[(df.scenario == s) & (df.method == method)]
            rows.append({
                "scenario": s, "method": method,
                "accuracy_mean": sub.accuracy.mean(), "accuracy_std": sub.accuracy.std(),
                "par10_mean": sub.par10_mean.mean(), "par10_std": sub.par10_mean.std(),
            })
    pd.DataFrame(rows).to_csv(results_root / "table3_multiseed_std.csv", index=False)

    # --- SBS is deterministic given the fold; pull from the original fold_results.csv ---
    sbs_frames = []
    for s in SCENARIOS:
        fr = pd.read_csv(results_root / s / "fold_results.csv")
        sbs = fr[fr.method == "SBS"][["scenario", "fold", "par10_mean"]].copy()
        sbs["method"] = "SBS"
        sbs_frames.append(sbs)
    sbs_df = pd.concat(sbs_frames, ignore_index=True)
    sbs_expanded = pd.concat(
        [sbs_df.assign(seed=seed) for seed in SEEDS], ignore_index=True)

    full = pd.concat(
        [df[["scenario", "fold", "seed", "method", "par10_mean"]], sbs_expanded],
        ignore_index=True)
    wide = full.pivot_table(index=["scenario", "fold", "seed"], columns="method", values="par10_mean")[METHODS]
    n_blocks = len(wide)

    stat, pvalue = friedmanchisquare(*[wide[m].values for m in METHODS])
    ranks = wide.rank(axis=1, ascending=True)
    avg_ranks = ranks.mean(axis=0)

    nemenyi = sp.posthoc_nemenyi_friedman(wide[METHODS].values)
    nemenyi.index = METHODS
    nemenyi.columns = METHODS
    nemenyi.to_csv(results_root / "multiseed_nemenyi.csv")

    wpairs = [("RandomForest", "XGBoost"), ("RandomForest", "SBS"), ("XGBoost", "SBS")]
    wres = []
    for m1, m2 in wpairs:
        st, p = wilcoxon(wide[m1].values, wide[m2].values)
        wres.append({"m1": m1, "m2": m2, "stat": st, "p": p})
    pd.DataFrame(wres).to_csv(results_root / "multiseed_wilcoxon.csv", index=False)

    k = 3
    q_alpha = 2.343
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    diff = ranks["RandomForest"] - ranks["XGBoost"]
    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    df_t = n - 1
    delta = cd
    t1 = (mean_diff - (-delta)) / se
    p1 = 1 - stats.t.cdf(t1, df_t)
    t2 = (mean_diff - delta) / se
    p2 = stats.t.cdf(t2, df_t)
    p_tost = max(p1, p2)

    pd.DataFrame([{
        "n_blocks": n_blocks, "friedman_chi2": stat, "friedman_p": pvalue,
        "rf_rank": avg_ranks["RandomForest"], "xgb_rank": avg_ranks["XGBoost"], "sbs_rank": avg_ranks["SBS"],
        "cd": cd, "tost_delta": delta, "tost_mean_diff": mean_diff, "tost_se": se,
        "tost_p1": p1, "tost_p2": p2, "tost_p": p_tost, "tost_equivalent_0.05": p_tost < 0.05,
    }]).to_csv(results_root / "multiseed_statistical_tests.csv", index=False)

    # --- Variance decomposition: fold vs. seed (example reported in Section 5.4) ---
    sub = df[(df.scenario == "SAT12-ALL") & (df.method == "RandomForest")]
    var_between_folds = sub.groupby("fold").par10_mean.mean().var()
    var_within_fold = sub.groupby("fold").par10_mean.var().mean()
    print(f"SAT12-ALL RandomForest: var_between_folds={var_between_folds:.1f}, "
          f"var_within_fold_across_seeds={var_within_fold:.1f}, "
          f"ratio={var_between_folds/var_within_fold:.2f}x")

    # --- Conservative 3-block Friedman (scenario-as-block), reported in Section 6 ---
    summary = pd.read_csv(results_root / "table3_multiseed_std.csv")
    sbs_means = sbs_df.groupby("scenario").par10_mean.mean().reset_index()
    sbs_means["method"] = "SBS"
    sbs_means = sbs_means.rename(columns={"par10_mean": "par10_mean"})
    block_df = pd.concat([
        summary[["scenario", "method", "par10_mean"]],
        sbs_means[["scenario", "method", "par10_mean"]],
    ], ignore_index=True)
    wide3 = block_df.pivot(index="scenario", columns="method", values="par10_mean")[METHODS]
    stat3, p3 = friedmanchisquare(*[wide3[m].values for m in METHODS])
    pd.DataFrame([{"n_blocks": 3, "friedman_chi2": stat3, "friedman_p": p3}]
                 ).to_csv(results_root / "friedman_3block.csv", index=False)
    print(f"3-block Friedman (scenario-as-block): chi2={stat3:.4f}, p={p3:.4f}")

    print("\nDone. Key outputs: table3_multiseed_std.csv, multiseed_statistical_tests.csv, "
          "multiseed_nemenyi.csv, multiseed_wilcoxon.csv, friedman_3block.csv")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed replication + TOST equivalence test")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--combine", action="store_true",
                         help="Skip retraining; combine existing multiseed_*.csv files and run the analysis.")
    args = parser.parse_args()

    if args.combine:
        combine_and_analyze(args.results_root)
        return

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    for scenario in scenarios:
        run_multiseed(scenario, args.data_root, args.results_root, args.fold_start, args.fold_end)


if __name__ == "__main__":
    main()

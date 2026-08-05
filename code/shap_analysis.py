"""
shap_analysis.py

Paso 3 del diseño experimental: añade la capa de explicabilidad (SHAP) sobre el
selector RandomForest ya validado en train_eval.py, y calcula la estabilidad de
las atribuciones de importancia de meta-features entre los 10 folds de ASlib.

Requiere que aslib_loader.py ya se haya corrido (usa data/<escenario>/dataset.csv).

Uso:
    python shap_analysis.py                      # corre los 3 escenarios por defecto
    python shap_analysis.py --scenario CSP-2010   # corre solo uno

Salida (por escenario), en results/<escenario>/:
    - shap_importance_by_fold.csv   importancia SHAP media |valor| por feature y fold
    - shap_top_features.csv         top-10 features por importancia SHAP promedio (Tabla 4)
    - stability.csv                 rho_stab: correlación de Spearman promedio entre folds (Tabla 5)
    - shap_summary_plot.png         SHAP summary plot para el fold representativo (Figura 3)
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
    Normaliza la salida de TreeExplainer (que varía de forma según binaria/
    multiclase y versión de SHAP) a un vector de importancia por feature:
    promedio de |SHAP| sobre instancias y, si aplica, sobre clases.
    """
    arr = np.asarray(shap_values.values) if hasattr(shap_values, "values") else np.asarray(shap_values)

    if isinstance(shap_values, list):
        # Lista de arrays (n_samples, n_features), uno por clase
        stacked = np.stack([np.abs(a) for a in shap_values], axis=0)  # (n_classes, n_samples, n_features)
        importance = stacked.mean(axis=(0, 1))
    elif arr.ndim == 3:
        # (n_samples, n_features, n_classes) -- shape típico en SHAP >= 0.4x para multiclase
        importance = np.abs(arr).mean(axis=(0, 2))
    else:
        # (n_samples, n_features) -- caso binario
        importance = np.abs(arr).mean(axis=0)

    assert importance.shape[0] == len(feature_cols), \
        f"Dimensión de importancia ({importance.shape[0]}) no coincide con # features ({len(feature_cols)})"
    return importance


def run_scenario(scenario: str, data_root: Path, results_root: Path):
    print(f"[{scenario}] cargando dataset...")
    dataset = pd.read_csv(data_root / scenario / "dataset.csv")

    algo_cols = [c for c in dataset.columns if c.startswith("perf__")]
    non_feature_cols = set(algo_cols) | {"instance_id", "best_algorithm", "cv_fold"}
    feature_cols = [c for c in dataset.columns if c not in non_feature_cols]

    folds = sorted(dataset["cv_fold"].dropna().unique())
    importance_by_fold = {}
    representative_shap = None  # guardamos uno para la Figura 3

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
        shap_values = explainer(X_test_imp)  # API moderna de shap (Explanation object)

        importance = mean_abs_shap_per_feature(shap_values, feature_cols)
        importance_by_fold[int(fold)] = importance

        if representative_shap is None:
            representative_shap = (shap_values, X_test_imp, feature_cols, int(fold))

        print(f"  fold {int(fold)}/{len(folds)} -> SHAP calculado")

    # --- Tabla: importancia por feature y fold ---
    imp_df = pd.DataFrame(importance_by_fold, index=feature_cols)
    imp_df.index.name = "feature"
    out_dir = results_root / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(out_dir / "shap_importance_by_fold.csv")

    # --- Tabla 4: top-K features por importancia SHAP promedio ---
    mean_importance = imp_df.mean(axis=1).sort_values(ascending=False)
    top_features = mean_importance.head(TOP_K).reset_index()
    top_features.columns = ["feature", "mean_abs_shap"]
    top_features["scenario"] = scenario
    top_features.to_csv(out_dir / "shap_top_features.csv", index=False)

    # --- Tabla 5: estabilidad (rho_stab) entre folds ---
    # Ranking de features por importancia, por fold
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

    # --- Figura 3: SHAP summary plot para el fold representativo ---
    shap_values, X_test_imp, feat_cols, fold_id = representative_shap
    plt.figure()
    try:
        # Si es multiclase, promediamos |SHAP| sobre clases para un summary plot legible
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
    parser = argparse.ArgumentParser(description="Capa de explicabilidad SHAP + estabilidad de atribuciones")
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
        print("Resumen de estabilidad combinado -> results/stability_all_scenarios.csv")


if __name__ == "__main__":
    main()

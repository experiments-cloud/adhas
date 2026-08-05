"""
statistical_tests.py

Paso 4 del diseño experimental: valida estadísticamente las diferencias de
desempeño (PAR10) entre métodos, usando los fold_results.csv generados por
train_eval.py en los 3 escenarios.

Unidad de bloque para Friedman/Nemenyi: cada combinación (escenario, fold),
es decir 3 escenarios x 10 folds = 30 bloques. Esto sigue la práctica común
en la literatura de algorithm selection para obtener suficiente poder
estadístico cuando se dispone de pocos escenarios (Demsar, 2006 recomienda
bloques = datasets, pero con pocos escenarios se usa fold x escenario).

Uso:
    python statistical_tests.py

Salida, en results/:
    - friedman_test.csv              estadístico y p-valor del test de Friedman
    - nemenyi_pvalues.csv             matriz de p-valores post-hoc de Nemenyi
    - average_ranks.csv               rango promedio de cada método (para Tabla 3)
    - wilcoxon_pairwise.csv           tests pareados de Wilcoxon (métodos ML vs baselines)
    - critical_difference_diagram.png Figura 4
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon

SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]
# Métodos "reales" de selección a comparar en Friedman/Nemenyi.
# Se excluyen VBS_Oracle (cota superior trivial) y Random (cota inferior trivial),
# ya que no son selectores comparables -- se mantienen solo como referencia en la Tabla 2.
METHODS = ["SBS", "RandomForest", "XGBoost"]


def load_wide_table(results_root: Path) -> pd.DataFrame:
    """Combina los fold_results.csv de los 3 escenarios en una tabla ancha:
    filas = (scenario, fold), columnas = método, valores = PAR10."""
    frames = []
    for s in SCENARIOS:
        df = pd.read_csv(results_root / s / "fold_results.csv")
        df = df[df["method"].isin(METHODS)]
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)

    wide = long_df.pivot_table(index=["scenario", "fold"], columns="method", values="par10_mean")
    wide = wide[METHODS]  # orden consistente
    return wide


def critical_difference_diagram(avg_ranks: pd.Series, cd: float, n_blocks: int, out_path: Path):
    """Implementación estándar (Demsar, 2006) del diagrama de diferencia crítica."""
    methods = list(avg_ranks.index)
    ranks = avg_ranks.values
    k = len(methods)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.6, color="black", linewidth=1)

    for m, r in zip(methods, ranks):
        ax.plot([r, r], [0.55, 0.65], color="black")
        ax.text(r, 0.7, f"{m}\n({r:.2f})", ha="center", va="bottom", fontsize=9)

    # Barra de CD (ancho = distancia mínima significativa)
    x0 = min(ranks)
    ax.plot([x0, x0 + cd], [0.3, 0.3], color="red", linewidth=2)
    ax.text(x0 + cd / 2, 0.2, f"CD = {cd:.3f}", ha="center", color="red", fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower PAR10 rank = better)")
    ax.set_title(f"Critical Difference Diagram (Nemenyi, n={n_blocks} blocks)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    results_root = Path("results")
    wide = load_wide_table(results_root)
    n_blocks = wide.shape[0]
    print(f"Bloques (escenario x fold): {n_blocks}")
    print(wide.head())
    print()

    # --- Test de Friedman ---
    stat, pvalue = friedmanchisquare(*[wide[m].values for m in METHODS])
    friedman_df = pd.DataFrame([{"statistic": stat, "p_value": pvalue, "n_blocks": n_blocks,
                                  "methods": ", ".join(METHODS)}])
    friedman_df.to_csv(results_root / "friedman_test.csv", index=False)
    print(f"Friedman: chi2={stat:.4f}, p={pvalue:.6g}")

    # --- Rangos promedio (para Tabla 3 y el diagrama CD) ---
    ranks = wide.rank(axis=1, ascending=True)  # rank 1 = menor PAR10 = mejor
    avg_ranks = ranks.mean(axis=0).sort_values()
    avg_ranks.to_csv(results_root / "average_ranks.csv", header=["avg_rank"])
    print("Rangos promedio:")
    print(avg_ranks)
    print()

    # --- Post-hoc Nemenyi ---
    nemenyi = sp.posthoc_nemenyi_friedman(wide[METHODS].values)
    nemenyi.index = METHODS
    nemenyi.columns = METHODS
    nemenyi.to_csv(results_root / "nemenyi_pvalues.csv")
    print("P-valores Nemenyi:")
    print(nemenyi)
    print()

    # --- Critical Difference (fórmula estándar de Demsar 2006) ---
    # q_alpha para Nemenyi con k métodos, alpha=0.05 (tabla estándar)
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    k = len(METHODS)
    q_alpha = q_alpha_table.get(k, 2.343)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    print(f"Critical Difference (alpha=0.05): {cd:.4f}")

    critical_difference_diagram(avg_ranks, cd, n_blocks,
                                 results_root / "critical_difference_diagram.png")

    # --- Wilcoxon pareado (comparaciones específicas de interés) ---
    pairs = [
        ("RandomForest", "XGBoost"),
        ("RandomForest", "SBS"),
        ("XGBoost", "SBS"),
    ]
    wilcoxon_rows = []
    for m1, m2 in pairs:
        stat_w, p_w = wilcoxon(wide[m1].values, wide[m2].values)
        wilcoxon_rows.append({
            "method_1": m1, "method_2": m2, "statistic": stat_w, "p_value": p_w,
            "significant_0.05": p_w < 0.05,
        })
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(results_root / "wilcoxon_pairwise.csv", index=False)
    print("Wilcoxon pareado:")
    print(wilcoxon_df.to_string(index=False))


if __name__ == "__main__":
    main()

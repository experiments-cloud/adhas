# Explainable Per-Instance Algorithm Selection — Reproducibility Package

This repository contains the full experimental pipeline behind the paper
*"Explainable Per-Instance Algorithm Selection: Assessing the Reliability
of Meta-Feature Attributions"* 


## Repository structure

```
.
├── README.md
├── requirements.txt
├── code/
    ├── aslib_loader.py       # Step 1: download + parse ASlib scenarios
    ├── train_eval.py         # Step 2: train selectors, compute PAR10/accuracy
    ├── shap_analysis.py      # Step 3: SHAP explainability + stability metric
    └── statistical_tests.py  # Step 4: Friedman/Nemenyi/Wilcoxon + CD diagram


```

Running the pipeline additionally produces `data/` (raw + processed ASlib
scenario data) and `results/` (all CSV outputs referenced in the
paper's tables in the working directory — these are not
version-controlled here since they are fully regenerable from the ASlib
source and are already provided separately alongside this package for
convenience.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12. `aslib_loader.py` requires outbound network
access to `raw.githubusercontent.com` (the official
[`coseal/aslib_data`](https://github.com/coseal/aslib_data) mirror).

## Running the full pipeline

All scripts default to relative `data/` and `results/` folders under the
**current working directory**, so run every step from the **repository
root** (not from inside `code/`):

```bash
# Step 1 — download and parse the 3 ASlib scenarios used in the paper
python code/aslib_loader.py
# -> data/<scenario>/dataset.csv, data/<scenario>/raw/*.arff

# Step 2 — train Random Forest / XGBoost selectors, compute PAR10, accuracy,
#          and the SBS / VBS / Random baselines (10-fold CV, per Section 4.1)
python code/train_eval.py
# -> results/<scenario>/{fold_results,summary}.csv
# -> results/summary_all_scenarios.csv  (Table 3)

# Step 3 — SHAP explainability layer + rho_stab cross-fold stability metric
python code/shap_analysis.py
# -> results/<scenario>/{shap_importance_by_fold,shap_top_features,stability}.csv
# -> results/<scenario>/shap_summary_plot.png
# -> results/{shap_top_features_all_scenarios,stability_all_scenarios}.csv (Tables 4-5)

# Step 4 — Friedman / Nemenyi / Wilcoxon tests + critical difference diagram
python code/statistical_tests.py
# -> results/{friedman_test,nemenyi_pvalues,average_ranks,wilcoxon_pairwise}.csv (Table 6)
# -> results/critical_difference_diagram.png (Figure 3)

```

Each script also accepts `--data-root` / `--results-root` (or
`--scenario NAME` to run a single scenario) if you prefer a different
layout — see `python code/<script>.py --help`.

Total runtime: a few minutes on a laptop CPU (SAT12-ALL, with 31
algorithms and 115 features, is the slowest step due to per-fold
TreeExplainer computation in `shap_analysis.py`).

## Mapping to the paper

| Script | Paper section(s) | Key outputs |
|---|---|---|
| `aslib_loader.py` | §4.1 (Benchmarks) | Table 2 |
| `train_eval.py` | §3.3, §4.2, §5.1, §5.4 | Table 3 |
| `shap_analysis.py` | §3.4, §3.5, §5.2, §5.3 | Table 4, Table 5, Figure 2 |
| `statistical_tests.py` | §4.4, §5.4 | Table 6, Figure 3 |

## License / data provenance

Scenario data originates from [ASlib](https://www.coseal.net/aslib/)
(Bischl et al., 2016) via the `coseal/aslib_data` GitHub mirror, and is
redistributed here only as intermediate pipeline output for
reproducibility. Original scenario authorship: CSP-2010 (CPHydra,
O'Mahony et al. 2008), QBF-2011 (AQME, Pulina & Tacchella 2009),
SAT12-ALL (Nudelman et al. 2004 features, as used by SATzilla).

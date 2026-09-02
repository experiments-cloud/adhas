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
    ├── statistical_tests.py  # Step 4: Friedman / Nemenyi / Wilcoxon + CD (n=30)                                   
    ├── multiseed_analysis.py # Step 5: 3-seed replication + TOST equivalence
    │                                        + conservative 3-block Friedman
    └── stability_topk_and_cost.py # Step 6: top-k Jaccard stability + SHAP
                                             computational cost (Table 6)

```

Running the pipeline additionally produces `data/` (raw + processed
ASlib scenario data) and `results/` (all CSV/PNG outputs referenced in
the paper's tables and figures) in the working directory — these are
regenerable from the ASlib source and are not version-controlled here.

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

# Step 2 — train Random Forest / XGBoost selectors, compute PAR10, accuracy,
#          and the SBS / VBS / Random baselines (10-fold CV, single seed)
python code/train_eval.py
# -> results/summary_all_scenarios.csv

# Step 3 — SHAP explainability layer + rho_stab cross-fold stability metric
python code/shap_analysis.py
# -> results/shap_top_features_all_scenarios.csv, stability_all_scenarios.csv

# Step 4 — Friedman / Nemenyi / Wilcoxon tests + critical difference diagram (n=30)
python code/statistical_tests.py
# -> results/friedman_test.csv, nemenyi_pvalues.csv, average_ranks.csv

# Step 5 — 3-seed replication, TOST equivalence test, 3-block Friedman
#          (retraining; SAT12-ALL is slow, ~3 min per fold range)
python code/multiseed_analysis.py --scenario CSP-2010
python code/multiseed_analysis.py --scenario QBF-2011
python code/multiseed_analysis.py --scenario SAT12-ALL
python code/multiseed_analysis.py --combine
# -> results/table3_multiseed_std.csv (Table 3)
# -> results/multiseed_statistical_tests.csv (Section 5.4 "Seed sensitivity")
# -> results/friedman_3block.csv (Section 6, conservative check)

# Step 6 — top-k Jaccard stability (Table 5) + SHAP computational cost (Table 6)
python code/stability_topk_and_cost.py
# -> results/topk_jaccard_stability.csv, shap_computational_cost.csv

```

Each script also accepts `--data-root` / `--results-root` if you prefer
a different layout — see `python code/<script>.py --help`.

## Mapping to the paper

| Script | Paper section(s) | Key outputs |
|---|---|---|
| `aslib_loader.py` | §4.1 (Benchmarks) | Table 2 |
| `train_eval.py` | §3.3, §4.2, §5.1 | Table 3 (single-seed baseline) |
| `shap_analysis.py` | §3.4, §3.5, §5.2, §5.3 | Table 4, Figures 2-4 (raw SHAP) |
| `statistical_tests.py` | §4.4, §5.4 | Table 7 ($n=30$), Figure 3 |
| `multiseed_analysis.py` | §5.4 ("Seed sensitivity"), §6 | Table 3 (final, multi-seed), TOST result, 3-block Friedman |
| `stability_topk_and_cost.py` | §5.3, §5.4 (RQ1) | Table 5 (Jaccard@5/@10), Table 6 (computational cost) |

## License / data provenance

Scenario data originates from [ASlib](https://www.coseal.net/aslib/)
(Bischl et al., 2016) via the `coseal/aslib_data` GitHub mirror, and is
redistributed here only as intermediate pipeline output for
reproducibility. Original scenario authorship: CSP-2010 (CPHydra,
O'Mahony et al. 2008), QBF-2011 (AQME, Pulina & Tacchella 2009),
SAT12-ALL (Nudelman et al. 2004 features, as used by SATzilla).

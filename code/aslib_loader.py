"""
aslib_loader.py

Downloads and parses ASlib scenarios (https://github.com/coseal/aslib_data)
for the explainable per-instance algorithm selection project.

Usage:
    python aslib_loader.py                     # download the 3 default scenarios
    python aslib_loader.py --scenario CSP-2010  # download only one

Output:
    For each scenario, saves under data/<SCENARIO>/:
        - raw/                  original .arff files (local cache)
        - dataset.csv           one row per instance, with:
                                   * meta-features
                                   * runtime/performance of each algorithm
                                   * best algorithm (column 'best_algorithm')
                                   * CV fold assigned by ASlib (column 'cv_fold')

Dependencies: pandas, numpy, requests (all standard/pip)
"""

import argparse
import csv
import io
import os
import re
import sys
from pathlib import Path

import pandas as pd
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/coseal/aslib_data/master"

# The 3 scenarios selected in the experimental design.
DEFAULT_SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]

# ARFF files we need from each scenario.
REQUIRED_FILES = ["description.txt", "algorithm_runs.arff", "feature_values.arff", "cv.arff"]


def download_file(scenario: str, filename: str, cache_dir: Path) -> Path:
    """Download a file from an ASlib scenario if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename
    if dest.exists():
        return dest
    url = f"{BASE_URL}/{scenario}/{filename}"
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read()
    except Exception as e:
        raise RuntimeError(f"Could not download {url}: {e}")
    dest.write_bytes(content)
    return dest


def parse_arff(path: Path) -> pd.DataFrame:
    """
    Minimalist ARFF parser (sufficient for ASlib files, which are
    standard ARFF without sparse data or complex types).
    Avoids depending on the external 'liac-arff' library to keep
    dependencies to a minimum.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    attributes = []
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        low = stripped.lower()
        if low.startswith("@attribute"):
            # @ATTRIBUTE name TYPE   (TYPE can include spaces if it is {a,b,c})
            m = re.match(r"@attribute\s+('[^']+'|\"[^\"]+\"|\S+)\s+(.*)", stripped, re.IGNORECASE)
            if m:
                name = m.group(1).strip("'\"")
                attributes.append(name)
        elif low.startswith("@data"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No @DATA section found in {path}")

    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        # Split while respecting quotes (some string values are quoted)
        fields = next(csv.reader([stripped]))
        rows.append(fields)

    df = pd.DataFrame(rows, columns=attributes)
    return df


def build_dataset(scenario: str, data_root: Path) -> pd.DataFrame:
    """
    Combines algorithm_runs, feature_values, and cv into a single
    DataFrame with one row per instance.
    """
    scen_dir = data_root / scenario
    raw_dir = scen_dir / "raw"

    print(f"[{scenario}] downloading ASlib files...")
    paths = {f: download_file(scenario, f, raw_dir) for f in REQUIRED_FILES}

    print(f"[{scenario}] parsing ARFF...")
    runs = parse_arff(paths["algorithm_runs.arff"])
    feats = parse_arff(paths["feature_values.arff"])
    cv = parse_arff(paths["cv.arff"])

    # --- Performance: one row per (instance, algorithm) -> pivot to wide ---
    # Typical columns: instance_id, repetition, algorithm, runtime, runstatus
    perf_col = "runtime" if "runtime" in runs.columns else runs.columns[-2]
    runs[perf_col] = pd.to_numeric(runs[perf_col], errors="coerce")

    perf_wide = runs.pivot_table(
        index="instance_id", columns="algorithm", values=perf_col, aggfunc="first"
    )
    perf_wide.columns = [f"perf__{c}" for c in perf_wide.columns]

    # Best algorithm per instance (assuming runtime: lower is better;
    # for quality scenarios the criterion would need to be inverted, check description.txt)
    algo_cols = list(perf_wide.columns)
    perf_wide["best_algorithm"] = perf_wide[algo_cols].idxmin(axis=1).str.replace("perf__", "", regex=False)

    # --- Meta-features: instance_id + feature columns ---
    feats = feats.set_index("instance_id")
    feature_cols = [c for c in feats.columns if c != "repetition"]
    feats_num = feats[feature_cols].apply(pd.to_numeric, errors="coerce")

    # --- CV folds ---
    cv = cv.set_index("instance_id")
    cv_col = "fold" if "fold" in cv.columns else cv.columns[-1]
    cv_folds = cv[[cv_col]].rename(columns={cv_col: "cv_fold"})

    # --- Final merge ---
    dataset = feats_num.join(perf_wide, how="inner").join(cv_folds, how="left")
    dataset.index.name = "instance_id"
    dataset = dataset.reset_index()

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ASlib scenarios")
    parser.add_argument("--scenario", type=str, default=None,
                         help="Name of a single scenario (default: the 3 in the experimental design)")
    parser.add_argument("--data-root", type=str, default="data",
                         help="Folder where downloaded/processed data is stored")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else DEFAULT_SCENARIOS
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        try:
            dataset = build_dataset(scenario, data_root)
        except Exception as e:
            print(f"[{scenario}] ERROR: {e}", file=sys.stderr)
            continue

        out_path = data_root / scenario / "dataset.csv"
        dataset.to_csv(out_path, index=False)
        n_algos = len([c for c in dataset.columns if c.startswith("perf__")])
        n_feats = len([c for c in dataset.columns
                        if not c.startswith("perf__") and c not in ("instance_id", "best_algorithm", "cv_fold")])
        print(f"[{scenario}] OK -> {out_path} "
              f"({len(dataset)} instances, {n_algos} algorithms, {n_feats} features)")


if __name__ == "__main__":
    main()

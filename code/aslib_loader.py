"""
aslib_loader.py

Descarga y parsea escenarios de ASlib (https://github.com/coseal/aslib_data)
para el proyecto de selección de algoritmos por instancia con explicabilidad.

Uso:
    python aslib_loader.py                     # descarga los 3 escenarios por defecto
    python aslib_loader.py --scenario CSP-2010  # descarga solo uno

Salida:
    Para cada escenario, guarda en data/<SCENARIO>/:
        - raw/                  archivos .arff originales (cache local)
        - dataset.csv           una fila por instancia, con:
                                   * meta-features
                                   * runtime/performance de cada algoritmo
                                   * mejor algoritmo (columna 'best_algorithm')
                                   * fold de CV asignado por ASlib (columna 'cv_fold')

Dependencias: pandas, numpy, requests (todas estándar/pip)
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

# Los 3 escenarios seleccionados en el diseño experimental.
DEFAULT_SCENARIOS = ["CSP-2010", "QBF-2011", "SAT12-ALL"]

# Archivos ARFF que necesitamos de cada escenario.
REQUIRED_FILES = ["description.txt", "algorithm_runs.arff", "feature_values.arff", "cv.arff"]


def download_file(scenario: str, filename: str, cache_dir: Path) -> Path:
    """Descarga un archivo de un escenario ASlib si no está ya en cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename
    if dest.exists():
        return dest
    url = f"{BASE_URL}/{scenario}/{filename}"
    print(f"  descargando {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read()
    except Exception as e:
        raise RuntimeError(f"No se pudo descargar {url}: {e}")
    dest.write_bytes(content)
    return dest


def parse_arff(path: Path) -> pd.DataFrame:
    """
    Parser ARFF minimalista (suficiente para los archivos de ASlib, que son
    ARFF estándar sin datos dispersos ni tipos complejos).
    Evita depender de la librería externa 'liac-arff' para no añadir
    dependencias adicionales.
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
            # @ATTRIBUTE name TYPE   (TYPE puede incluir espacios si es {a,b,c})
            m = re.match(r"@attribute\s+('[^']+'|\"[^\"]+\"|\S+)\s+(.*)", stripped, re.IGNORECASE)
            if m:
                name = m.group(1).strip("'\"")
                attributes.append(name)
        elif low.startswith("@data"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No se encontró sección @DATA en {path}")

    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        # Split respetando comillas simples (algunos valores string vienen citados)
        fields = next(csv.reader([stripped]))
        rows.append(fields)

    df = pd.DataFrame(rows, columns=attributes)
    return df


def build_dataset(scenario: str, data_root: Path) -> pd.DataFrame:
    """
    Combina algorithm_runs, feature_values y cv en un único DataFrame
    con una fila por instancia.
    """
    scen_dir = data_root / scenario
    raw_dir = scen_dir / "raw"

    print(f"[{scenario}] descargando archivos ASlib...")
    paths = {f: download_file(scenario, f, raw_dir) for f in REQUIRED_FILES}

    print(f"[{scenario}] parseando ARFF...")
    runs = parse_arff(paths["algorithm_runs.arff"])
    feats = parse_arff(paths["feature_values.arff"])
    cv = parse_arff(paths["cv.arff"])

    # --- Performance: una fila por (instancia, algoritmo) -> pivotear a ancho ---
    # Columnas típicas: instance_id, repetition, algorithm, runtime, runstatus
    perf_col = "runtime" if "runtime" in runs.columns else runs.columns[-2]
    runs[perf_col] = pd.to_numeric(runs[perf_col], errors="coerce")

    perf_wide = runs.pivot_table(
        index="instance_id", columns="algorithm", values=perf_col, aggfunc="first"
    )
    perf_wide.columns = [f"perf__{c}" for c in perf_wide.columns]

    # Mejor algoritmo por instancia (asumiendo runtime: menor es mejor;
    # para escenarios de calidad habría que invertir el criterio, revisar description.txt)
    algo_cols = list(perf_wide.columns)
    perf_wide["best_algorithm"] = perf_wide[algo_cols].idxmin(axis=1).str.replace("perf__", "", regex=False)

    # --- Meta-features: instance_id + columnas de features ---
    feats = feats.set_index("instance_id")
    feature_cols = [c for c in feats.columns if c != "repetition"]
    feats_num = feats[feature_cols].apply(pd.to_numeric, errors="coerce")

    # --- CV folds ---
    cv = cv.set_index("instance_id")
    cv_col = "fold" if "fold" in cv.columns else cv.columns[-1]
    cv_folds = cv[[cv_col]].rename(columns={cv_col: "cv_fold"})

    # --- Merge final ---
    dataset = feats_num.join(perf_wide, how="inner").join(cv_folds, how="left")
    dataset.index.name = "instance_id"
    dataset = dataset.reset_index()

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Descarga y prepara escenarios ASlib")
    parser.add_argument("--scenario", type=str, default=None,
                         help="Nombre de un único escenario (default: los 3 del diseño experimental)")
    parser.add_argument("--data-root", type=str, default="data",
                         help="Carpeta donde guardar los datos descargados/procesados")
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
              f"({len(dataset)} instancias, {n_algos} algoritmos, {n_feats} features)")


if __name__ == "__main__":
    main()

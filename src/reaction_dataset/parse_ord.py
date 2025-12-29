#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_ord_jsonl.py
ETL Pipeline para dataset ORD descomprimido a JSONL dentro de .gz.

1️⃣ Recorre recursivamente la carpeta base (processed_data).
2️⃣ Descomprime cada archivo .jsonl.gz o .jsonl.
3️⃣ Lee línea a línea sin cargar todo en memoria.
4️⃣ Extrae reactants, products, condiciones, yield, y metadatos.
5️⃣ Escribe todo en un archivo CSV comprimido .csv.gz.
"""

import os
import gzip
import csv
import json
import argparse
import logging
from typing import Iterator, Dict, Any, List, Optional

# ==============================
# Configuración del logger
# ==============================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("parse_ord_jsonl")

# ==============================
# Funciones auxiliares
# ==============================

def extract_smiles_from_inputs(inputs: Dict[str, Any]) -> List[str]:
    """Extrae todos los SMILES de reactivos/catalizadores dentro de 'inputs'."""
    smiles = []
    if not isinstance(inputs, dict):
        return smiles
    for _, group in inputs.items():
        comps = group.get("components", [])
        for comp in comps:
            ids = comp.get("identifiers", [])
            for identifier in ids:
                if identifier.get("type", "").upper() == "SMILES":
                    smiles.append(identifier.get("value"))
    return smiles


def extract_smiles_from_products(outcomes: List[Dict[str, Any]]) -> List[str]:
    """Extrae los SMILES de los productos en outcomes."""
    smiles = []
    if not outcomes:
        return smiles
    for oc in outcomes:
        for prod in oc.get("products", []):
            for identifier in prod.get("identifiers", []):
                if identifier.get("type", "").upper() == "SMILES":
                    smiles.append(identifier.get("value"))
    return smiles


def extract_yield(outcomes: List[Dict[str, Any]]) -> Optional[float]:
    """Devuelve el rendimiento (%), si está disponible."""
    try:
        for oc in outcomes:
            for prod in oc.get("products", []):
                for meas in prod.get("measurements", []):
                    if meas.get("type") == "YIELD":
                        return meas.get("percentage", {}).get("value")
    except Exception:
        return None
    return None


def extract_temperature(conditions: Dict[str, Any]) -> Optional[float]:
    """Extrae la temperatura en Celsius."""
    try:
        return conditions["temperature"]["setpoint"]["value"]
    except Exception:
        return None


def iterate_reactions_from_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Lee un archivo .jsonl (posiblemente comprimido) y genera un dict por reacción.
    """
    logger.debug(f"Leyendo archivo: {file_path}")

    open_func = gzip.open if file_path.endswith(".gz") else open
    with open_func(file_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON inválido en línea {line_num} ({file_path}): {e}")
                continue

            reaction_id = obj.get("reaction_id")
            reactant_smiles = extract_smiles_from_inputs(obj.get("inputs", {}))
            product_smiles = extract_smiles_from_products(obj.get("outcomes", []))
            yield_pct = extract_yield(obj.get("outcomes", []))
            temperature_c = extract_temperature(obj.get("conditions", {}))

            provenance = obj.get("provenance", {})
            org = provenance.get("experimenter", {}).get("organization")
            date = provenance.get("experiment_start", {}).get("value")

            row = {
                "reaction_id": reaction_id,
                "reactant_smiles_json": json.dumps(reactant_smiles, ensure_ascii=False),
                "product_smiles_json": json.dumps(product_smiles, ensure_ascii=False),
                "yield_percent": yield_pct,
                "temperature_c": temperature_c,
                "organization": org,
                "experiment_date": date,
                "source_file": os.path.basename(file_path),
                "source_path": os.path.abspath(file_path),
            }
            yield row


def walk_and_parse_jsonl(base_dir: str, output_csv_gz: str):
    """Recorre el directorio base y procesa todos los .jsonl y .jsonl.gz."""
    fieldnames = [
        "reaction_id",
        "reactant_smiles_json",
        "product_smiles_json",
        "yield_percent",
        "temperature_c",
        "organization",
        "experiment_date",
        "source_file",
        "source_path",
    ]

    logger.info(f"Buscando archivos JSONL en {base_dir}")
    jsonl_files = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".jsonl") or fn.endswith(".jsonl.gz"):
                jsonl_files.append(os.path.join(root, fn))

    logger.info(f"Se encontraron {len(jsonl_files)} archivos JSONL/GZ.")

    with gzip.open(output_csv_gz, "wt", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        total_rows = 0

        for fp in jsonl_files:
            for row in iterate_reactions_from_jsonl(fp):
                writer.writerow(row)
                total_rows += 1

        logger.info(f"Se escribieron {total_rows} filas en {output_csv_gz}")


# ==============================
# CLI
# ==============================
def main():
    parser = argparse.ArgumentParser(
        description="ETL Pipeline para ORD JSONL dentro de .gz"
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        required=True,
        help="Ruta base que contiene las subcarpetas con .jsonl.gz",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Ruta de salida para ord_parsed_reactions.csv.gz",
    )
    parser.add_argument("--log-level", default="INFO", help="Nivel de log")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    walk_and_parse_jsonl(args.base_dir, args.output)


if __name__ == "__main__":
    main()

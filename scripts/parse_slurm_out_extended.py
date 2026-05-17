#!/usr/bin/env python3
"""Extended parser for SLURM .out training logs.

Extracts:
- method
- dataset
- trigger_type
- clean_acc_800, back_acc_800 (from [800-epoch] line)
- linear_ACC, linear_ASR (linear classifier before replacements)
- knn_clean_acc, knn_back_asr (overall summary)
- linear_clean_acc, linear_back_asr (overall summary)

Usage:
  python scripts/parse_slurm_out_extended.py path/to/file.or.dir -o out.json -f json

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, Optional


def first_number(s: str) -> Optional[float]:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_file(path: str) -> Dict[str, Optional[object]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    out: Dict[str, Optional[object]] = {
        "method": None,
        "dataset": None,
        "trigger_type": None,
        "clean_acc_800": None,
        "back_acc_800": None,
        "linear_ACC": None,
        "linear_ASR": None,
        "knn_clean_acc": None,
        "knn_back_asr": None,
        "linear_clean_acc": None,
        "linear_back_asr": None,
    }

    # simple key: value lines
    for key in ("method", "dataset", "trigger_type"):
        m = re.search(
            rf"^\s*{re.escape(key)}\s*:\s*(.+)$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if m:
            out[key] = m.group(1).strip()

    # [800-epoch] line
    m800 = re.search(
        r"\[\s*800-epoch\s*\].*?clean acc:\s*([0-9]+(?:\.[0-9]+)?).*?back acc:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m800:
        out["clean_acc_800"] = float(m800.group(1))
        out["back_acc_800"] = float(m800.group(2))

    # linear classifier immediate report (before replacements)
    mlin = re.search(
        r"for linear classifier.*?ACC on clean val is:\s*([0-9]+(?:\.[0-9]+)?).*?ASR on poisoned val is:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if mlin:
        out["linear_ACC"] = float(mlin.group(1))
        out["linear_ASR"] = float(mlin.group(2))

    # overall summary lines like: knn_clean_acc: 54.9±0.3
    def extract_mean_std_str(keyname: str) -> Optional[str]:
        # matches 'key: 54.9±0.3' or 'key: 54.9 +/- 0.3' or 'key: 54.9+-0.3'
        pattern = rf"{re.escape(keyname)}\s*:\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:±|\+/-|\+-)\s*([0-9]+(?:\.[0-9]+)?))?"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            mean = m.group(1)
            std = m.group(2)
            if std:
                return f"{mean}±{std}"
            return mean
        return None

    out["knn_clean_acc"] = extract_mean_std_str("knn_clean_acc")
    out["knn_back_asr"] = extract_mean_std_str("knn_back_asr")
    out["linear_clean_acc"] = extract_mean_std_str("linear_clean_acc")
    out["linear_back_asr"] = extract_mean_std_str("linear_back_asr")

    return out


def process_path(path: str) -> Dict[str, Dict[str, Optional[object]]]:
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".out"))
        results: Dict[str, Dict[str, Optional[object]]] = {}
        for fn in files:
            full = os.path.join(path, fn)
            results[fn] = parse_file(full)
        return results
    else:
        return {os.path.basename(path): parse_file(path)}


def write_csv(results: Dict[str, Dict[str, Optional[object]]], outpath: str) -> None:
    fieldnames = [
        "filename",
        "method",
        "dataset",
        "trigger_type",
        "clean_acc_800",
        "back_acc_800",
        "linear_ACC",
        "linear_ASR",
        "knn_clean_acc",
        "knn_back_asr",
        "linear_clean_acc",
        "linear_back_asr",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for fn, vals in results.items():
            row = {k: vals.get(k) for k in fieldnames if k != "filename"}
            row["filename"] = fn
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(description="Extended .out parser")
    p.add_argument("path", help=".out file or directory containing .out files")
    p.add_argument("-o", "--output", help="Output file path (JSON or CSV)")
    p.add_argument(
        "-f",
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format when --output is provided (default: json)",
    )
    args = p.parse_args()

    results = process_path(args.path)

    if args.output:
        fmt = args.format.lower()
        if fmt == "json":
            with open(args.output, "w", encoding="utf-8") as jf:
                json.dump(results, jf, indent=2)
        else:
            write_csv(results, args.output)
        print(f"Wrote results to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""
Sample usage:
python3 scripts/parse_slurm_out_extended.py results/ablation_aug_1 -o results/results_ablation_aug_1.json -f json
"""

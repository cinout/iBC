#!/usr/bin/env python3
"""Parse SLURM .out training logs and extract key metrics.

Extracts:
- method
- dataset
- trigger_type
- clean/back acc at 800-epoch
- linear classifier ACC and ASR

Usage:
  python scripts/parse_slurm_out.py path/to/slurm-123.out

If a directory is provided, all files ending with `.out` will be processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Dict, Optional


def parse_out_file(path: str) -> Dict[str, Optional[float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    result: Dict[str, Optional[object]] = {
        "method": None,
        "dataset": None,
        "trigger_type": None,
        "clean_acc_800": None,
        "back_acc_800": None,
        "linear_ACC": None,
        "linear_ASR": None,
    }

    # Simple key:value lines like `method: byol`
    for key in ("method", "dataset", "trigger_type"):
        m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+)$", text, flags=re.MULTILINE)
        if m:
            result[key] = m.group(1).strip()

    # clean/back acc at 800-epoch
    m800 = re.search(
        r"\[\s*800-epoch\s*\].*?clean acc:\s*([0-9]+\.?[0-9]*)\s*\|?[^\n]*back acc:\s*([0-9]+\.?[0-9]*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m800:
        try:
            result["clean_acc_800"] = float(m800.group(1))
            result["back_acc_800"] = float(m800.group(2))
        except ValueError:
            pass

    # If exact 800-epoch entry not found, try to find last occurrence of "-epoch" with 800 in it
    if result["clean_acc_800"] is None or result["back_acc_800"] is None:
        # find lines like [800-epoch] or [ 800-epoch ] but be tolerant
        for line in text.splitlines():
            if re.search(r"\[\s*800-epoch\s*\]", line):
                m = re.search(
                    r"clean acc:\s*([0-9]+\.?[0-9]*).*back acc:\s*([0-9]+\.?[0-9]*)",
                    line,
                    flags=re.IGNORECASE,
                )
                if m:
                    try:
                        result["clean_acc_800"] = float(m.group(1))
                        result["back_acc_800"] = float(m.group(2))
                        break
                    except ValueError:
                        continue

    # linear classifier final summary
    mlin = re.search(
        r"ACC on clean val is:\s*([0-9]+\.?[0-9]*).*ASR on poisoned val is:\s*([0-9]+\.?[0-9]*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if mlin:
        try:
            result["linear_ACC"] = float(mlin.group(1))
            result["linear_ASR"] = float(mlin.group(2))
        except ValueError:
            pass

    # Some logs might instead phrase it differently; try a DOTALL tolerant match
    if result["linear_ACC"] is None or result["linear_ASR"] is None:
        m2 = re.search(
            r"for linear classifier.*ACC on clean val is:\s*([0-9]+\.?[0-9]*).*ASR on poisoned val is:\s*([0-9]+\.?[0-9]*)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m2:
            try:
                result["linear_ACC"] = float(m2.group(1))
                result["linear_ASR"] = float(m2.group(2))
            except ValueError:
                pass

    return result


def process_path(path: str):
    """Return a dict mapping filename -> parsed result (single file returns dict for that file)."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".out"))
        out: Dict[str, Dict[str, Optional[object]]] = {}
        for fn in files:
            full = os.path.join(path, fn)
            out[fn] = parse_out_file(full)
        return out
    else:
        res = parse_out_file(path)
        # return with the basename as key for consistency
        return {os.path.basename(path): res}


def write_csv(out: Dict[str, Dict[str, Optional[object]]], outpath: str) -> None:
    fieldnames = [
        "filename",
        "method",
        "dataset",
        "trigger_type",
        "clean_acc_800",
        "back_acc_800",
        "linear_ACC",
        "linear_ASR",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for fn, vals in out.items():
            row = {k: vals.get(k) for k in fieldnames if k != "filename"}
            row["filename"] = fn
            # ensure order
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(description="Parse .out training logs for metrics")
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
        outpath = args.output
        if fmt == "json":
            with open(outpath, "w", encoding="utf-8") as jf:
                json.dump(results, jf, indent=2)
        else:
            # write CSV
            write_csv(results, outpath)
        print(f"Wrote results to {outpath}")
    else:
        # print to stdout as JSON
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""
Example Usage:
python3 scripts/parse_slurm_out.py clean_encoders_results/ -o clean_encoders_results.json -f json 
"""

#!/usr/bin/env python3
"""
Scan slurm output files for `saved_path`, `adaptive_attack_lambda`, and
`adaptive_attack_mode`, then generate a shell script with commands that
replace the old saved path with a new path that encodes the adaptive
attack settings.

Usage:
  python3 scripts/generate_replace_commands.py --input-dir results --out replace_saved_paths.sh

By default it searches recursively under `results/` for files ending with
`.out`.
"""

import argparse
import os
import re
from pathlib import Path


def find_kv(text, key):
    m = re.search(rf"^{re.escape(key)}:\s*(\S+)", text, flags=re.M)
    return m.group(1) if m else None


def shell_quote_single(s: str) -> str:
    # Quote for inclusion inside single-quoted shell string
    return s.replace("'", "'\\''")


def make_new_path(old_path, mode, lam, timestamp=None):
    # Preserve directory, replace basename.
    dirn = os.path.dirname(old_path)
    base = os.path.basename(old_path)
    rest = base
    if timestamp and base.startswith(timestamp + "_"):
        rest = base[len(timestamp) + 1 :]
    new_base = f"ada_{mode}_{lam}_{rest}"
    return os.path.join(dirn, new_base)


def process_file(p: Path):
    text = p.read_text()
    saved = find_kv(text, "saved_path")
    lam = find_kv(text, "adaptive_attack_lambda")
    mode = find_kv(text, "adaptive_attack_mode")
    timestamp = find_kv(text, "timestamp")
    if not saved or not lam or not mode:
        return None
    new = make_new_path(saved, mode, lam, timestamp)
    return (saved, new, str(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir", default="results", help="root to search for .out files"
    )
    ap.add_argument(
        "--out", default="replace_saved_paths.sh", help="output shell script"
    )
    args = ap.parse_args()

    root = Path(args.input_dir)
    matches = []
    for p in root.rglob("*.out"):
        try:
            res = process_file(p)
        except Exception:
            res = None
        if res:
            matches.append(res)

    if not matches:
        print("No matching slurm .out files with required keys found.")
        return

    out_lines = [
        "#!/usr/bin/env bash\n",
        "# Generated rename commands. Run from project root.\n",
    ]
    out_lines.append("set -euo pipefail\n\n")
    out_lines.append("# For safety this script will check existence before moving.\n")
    for old, new, src in matches:
        old_q = sh_quote(old)
        new_q = sh_quote(new)
        parent_new = sh_quote(os.path.dirname(new) or ".")
        out_lines.append(
            f'echo "[INFO] Processing {sh_quote(src)} -> move {old_q} -> {new_q}"\n'
        )
        out_lines.append(f"if [ -e {old_q} ]; then\n")
        out_lines.append(f"  if [ -e {new_q} ]; then\n")
        out_lines.append(f'    echo "[SKIP] target exists: {new_q}"\n')
        out_lines.append(f"  else\n")
        out_lines.append(f"    mkdir -p {parent_new}\n")
        out_lines.append(f'    echo "[MOVE] {old_q} -> {new_q}"\n')
        out_lines.append(f"    mv -- {old_q} {new_q}\n")
        out_lines.append(f"  fi\n")
        out_lines.append(f"else\n")
        out_lines.append(f'  echo "[WARN] source not found: {old_q}"\n')
        out_lines.append(f"fi\n\n")

    Path(args.out).write_text("".join(out_lines))
    os.chmod(args.out, 0o755)
    print(f"Wrote {args.out} with {len(matches)} move commands")


def sh_quote(s: str) -> str:
    # simple shell quoting for filenames
    if re.search(r"[\s\\'\"$]", s):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    main()

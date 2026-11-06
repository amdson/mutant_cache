"""Content hashes of the processed stability benchmarks (and their raw inputs).

    python -m mutant_cache.data.verify_processed            # check $MUTANT_CACHE_SCRATCH
    python -m mutant_cache.data.verify_processed --record   # (re)write the expected file
    python -m mutant_cache.data.verify_processed --group esm-msr

Expected hashes live in ``datasets/processed_hashes.json``. Exit status is 1 if
any listed file is missing or its content hash differs.

Hash rules (``RULES`` below, also written into the expected file):

- **table** (``.parquet`` / ``.csv``): read into pandas (parquet via pyarrow;
  CSV with ``float_precision="round_trip"``). Columns are sorted by name, a
  header line records ``(column, type)`` with type ``int<bits>`` / ``float<bits>``
  / ``bool`` / ``str`` (object), so the parquet writer (polars vs pandas,
  ``string`` vs ``large_string``, row groups, compression, metadata) does not
  matter. Each row is serialized as a JSON array (floats as ``"f:" + repr``,
  i.e. exact and never equal to an int; ints as integers; null/NaN as ``null``), lines are sorted, and the
  hash is sha256 over header + sorted lines. Row order is therefore ignored;
  ``ordered_sha256`` (same, unsorted) is reported as a note only.
- **json**: parsed, re-serialized with sorted keys and fixed separators.
- **bytes**: plain sha256 of the file (raw inputs only).
- **paths**: in tables and JSON, any string that starts with the scratch root
  (as given, or its realpath) is rewritten to ``$MUTANT_CACHE_SCRATCH/...``.
  This affects ``manifest.json`` ``files.*.path`` / ``external_dataset_dir``,
  ``audit/provenance.json`` ``archive`` / ``source_csv`` / ``split_file`` and the
  ``split_file`` column of ``external_overlap.csv``. No field is dropped; a
  path outside scratch stays literal (and will not match). None of these
  files carry timestamps.
- ``rounded_sha256`` repeats the table/json hash with every float formatted to
  ``ROUND_SIG`` significant digits. It is a diagnostic for library-version
  float noise on another machine: a file whose exact hash differs but whose
  rounded hash matches is reported as ``ROUNDED`` (still exit 1 unless
  ``--accept-rounded``).
- **unstable** (``UNSTABLE``): files that differ between rebuilds of identical
  inputs because of a known nondeterminism; a mismatch prints ``UNSTABLE``
  and fails only under ``--strict``. Currently only the ESM-MSR homology audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import scratch

EXPECTED_FILE = Path(__file__).resolve().parents[2] / "datasets" / "processed_hashes.json"
ROUND_SIG = 10
SCRATCH_TOKEN = "$MUTANT_CACHE_SCRATCH"

ME = "processed/mutate_everything_cdna"
ME_EXT = "mutate_everything/external"
MSR = "processed/esm_msr_mega"

# group (make target) -> relative paths under scratch
GROUPS: dict[str, list[str]] = {
    "mutate-everything-raw": [
        "mutate_everything/mutate_everything_data.zip",
        "mutate_everything/data/cdna_train.csv",
        "mutate_everything/data/cdna2_test.csv",
    ],
    "mutate-everything": [
        f"{ME}/targets.parquet", f"{ME}/variants.parquet", f"{ME}/score_sets.parquet",
        f"{ME}/wt_report.parquet", f"{ME}/splits/mutate_everything_cdna.parquet",
        f"{ME}/exclusions_train.csv", f"{ME}/exclusions_test.csv",
        f"{ME}/comparison_labels_train.csv", f"{ME}/comparison_labels_test.csv",
        f"{ME}/external_overlap.csv", f"{ME}/manifest.json",
    ],
    "mutate-everything-holdouts": [
        f"{ME}/splits/mutate_everything_cdna_holdout1.parquet",
        f"{ME}/splits/mutate_everything_cdna_holdout2.parquet",
    ],
    "mutate-everything-external": [
        f"{ME_EXT}/{n}" for n in (
            "s669.csv", "protherm_multiple.csv", "cdna1_test.csv",
            "s669_labels.csv", "ptmul_labels.csv", "cdna1_labels.csv", "s669_reverse_labels.csv",
            "s669_exclusions.csv", "ptmul_exclusions.csv", "cdna1_exclusions.csv",
            "inventory.json")
    ],
    "esm-msr-raw": [
        "esm_msr/preprocessed_data.tar.gz",
        "esm_msr/preprocessed/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
        "esm_msr/repo/data/hyperopt_splits.csv",
    ],
    "esm-msr": [
        f"{MSR}/targets.parquet", f"{MSR}/variants.parquet", f"{MSR}/score_sets.parquet",
        f"{MSR}/wt_report.parquet", f"{MSR}/splits/esm_msr_hyperopt.parquet",
        *(f"{MSR}/audit/{n}.csv" for n in (
            "exclusions", "reconstruction_failures", "split_summary", "overlap", "proteins",
            "homology_ge30")),
        f"{MSR}/audit/provenance.json",
    ],
}
RAW_GROUPS = {"mutate-everything-raw", "esm-msr-raw"}

# Present in scratch but not produced by any datasets/Makefile rule.
NOT_VERIFIED = {
    f"{ME}/sign_and_overlap_audit.json": "no recorded generating script",
    f"{ME}/largest_external_label_differences.csv": "no recorded generating script",
    f"{ME_EXT}/(baseline|ours_joint_s0|report_*|homology|headline.csv|parity_checks.json|driver*)":
        "evaluation outputs, not data",
}

# Files whose content legitimately differs between rebuilds of identical
# inputs. A mismatch is printed as UNSTABLE (exit 0 unless --strict).
UNSTABLE = {
    f"{MSR}/audit/homology_ge30.csv":
        "schema.write_dataset dedups targets with polars unique() without maintain_order, so "
        "targets.parquet row order varies run to run; esm_msr.homology_audit keeps the first "
        "target per code_wt (drop_duplicates), so the representative sequence (natural domain vs "
        "an engineered background) and hence the >=30% pair list vary. Verified 2026-11-05: "
        "recomputing on the rebuilt tables in the recorded targets order reproduces the recorded "
        "file exactly. Audit only; never used to filter or split.",
}

RULES = {
    "table": "pandas read (parquet: pyarrow; csv: float_precision=round_trip); columns sorted by "
             "name; header [(column, int<bits>|float<bits>|bool|str)]; rows as JSON arrays "
             "(float repr exact, null for None/NaN); rows sorted; sha256(header + sorted rows)",
    "json": "json.load, re-dumped with sort_keys=True, separators=(',', ':'), ensure_ascii=True",
    "bytes": "sha256 of the file bytes",
    "paths": f"strings starting with the scratch root (literal or realpath) -> {SCRATCH_TOKEN}/...; "
             "no fields dropped; none of these files carry timestamps",
    "rounded_sha256": f"same as content_sha256 with floats formatted '%.{ROUND_SIG}g' (diagnostic only)",
    "ordered_sha256": "table hash without sorting rows (row order; informational only)",
    "unstable": "files listed with an 'unstable' reason are reported, not failed, on mismatch "
                "(--strict fails them)",
}


# --------------------------------------------------------------- canonical forms
class _Canon:
    def __init__(self, root: Path, round_sig: int | None = None):
        r = str(root).rstrip("/")
        self.prefixes = sorted({r, os.path.realpath(r)}, key=len, reverse=True)
        self.round_sig = round_sig

    def string(self, s: str) -> str:
        for p in self.prefixes:
            if s == p or s.startswith(p + "/"):
                return SCRATCH_TOKEN + s[len(p):]
        return s

    def value(self, v):
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            f = float(v)
            if math.isnan(f):
                return None
            if math.isinf(f):
                return "inf" if f > 0 else "-inf"
            # floats become tagged strings so 1.0 never collides with the int 1
            return "f:" + (format(f + 0.0, f".{self.round_sig}g") if self.round_sig else repr(f))
        if isinstance(v, str):
            return self.string(v)
        if isinstance(v, dict):
            return {self.string(str(k)): self.value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self.value(x) for x in v]
        if v is pd.NA or v is pd.NaT:
            return None
        return "?" + type(v).__name__ + ":" + str(v)


def _type_token(dtype) -> str:
    if dtype.kind in "iu":
        return f"{'u' if dtype.kind == 'u' else ''}int{dtype.itemsize * 8}"
    if dtype.kind == "f":
        return f"float{dtype.itemsize * 8}"
    if dtype.kind == "b":
        return "bool"
    if dtype.kind in "OSU":
        return "str"
    return str(dtype)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pandas()
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def table_lines(df: pd.DataFrame, canon: _Canon) -> tuple[str, list[str]]:
    cols = sorted(df.columns, key=str)
    header = json.dumps([[str(c), _type_token(df[c].dtype)] for c in cols])
    columns = [[canon.value(v) for v in df[c].tolist()] for c in cols]
    lines = [json.dumps(list(row), ensure_ascii=True, separators=(",", ":"))
             for row in zip(*columns)] if cols else []
    return header, lines


def _sha(header: str, lines) -> str:
    h = hashlib.sha256(header.encode())
    for line in lines:
        h.update(b"\n")
        h.update(line.encode())
    return h.hexdigest()


def json_text(path: Path, canon: _Canon) -> str:
    return json.dumps(canon.value(json.loads(path.read_text())), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(path: Path, root: Path, *, raw: bool) -> dict:
    """Hashes of one file under the rules above."""
    if raw:
        return {"kind": "bytes", "sha256": file_sha256(path)}
    exact, rounded = _Canon(root), _Canon(root, ROUND_SIG)
    if path.suffix == ".json":
        return {"kind": "json",
                "content_sha256": hashlib.sha256(json_text(path, exact).encode()).hexdigest(),
                "rounded_sha256": hashlib.sha256(json_text(path, rounded).encode()).hexdigest()}
    df = read_table(path)
    header, lines = table_lines(df, exact)
    _, rlines = table_lines(df, rounded)
    return {"kind": "table", "rows": len(df), "columns": sorted(map(str, df.columns)),
            "content_sha256": _sha(header, sorted(lines)),
            "rounded_sha256": _sha(header, sorted(rlines)),
            "ordered_sha256": _sha(header, lines)}


def _selected(groups):
    return {g: GROUPS[g] for g in (groups or GROUPS)}


def record(root: Path, out: Path, groups=None) -> dict:
    data = {"format": 1, "rules": RULES, "round_sig": ROUND_SIG, "groups": {},
            "not_verified": NOT_VERIFIED}
    if out.exists() and groups:
        data = json.loads(out.read_text())
    for g, rels in _selected(groups).items():
        data["groups"][g] = {}
        for rel in rels:
            p = root / rel
            if not p.exists():
                raise FileNotFoundError(p)
            data["groups"][g][rel] = describe(p, root, raw=g in RAW_GROUPS)
            if rel in UNSTABLE:
                data["groups"][g][rel]["unstable"] = UNSTABLE[rel]
            print(f"recorded {rel}", flush=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    return data


def verify(root: Path, expected: dict, groups=None, *, accept_rounded=False, strict=False) -> bool:
    ok = True
    for g, rels in expected["groups"].items():
        if groups and g not in groups:
            continue
        print(f"[{g}]")
        for rel, exp in rels.items():
            p = root / rel
            if not p.exists():
                print(f"  MISSING   {rel}")
                ok = False
                continue
            got = describe(p, root, raw=exp["kind"] == "bytes")
            key = "sha256" if exp["kind"] == "bytes" else "content_sha256"
            if got[key] == exp[key]:
                note = ""
                if exp.get("ordered_sha256") and got["ordered_sha256"] != exp["ordered_sha256"]:
                    note = "  (row order differs)"
                print(f"  OK        {rel}{note}")
            elif exp.get("rounded_sha256") and got["rounded_sha256"] == exp["rounded_sha256"]:
                print(f"  ROUNDED   {rel}  (equal only at {expected.get('round_sig')} significant digits)")
                ok = ok and accept_rounded
            elif exp.get("unstable"):
                print(f"  UNSTABLE  {rel}  (differs; known nondeterministic: {exp['unstable'][:90]}...)")
                ok = ok and not strict
            else:
                extra = ""
                if exp["kind"] == "table":
                    extra = f"  rows {got['rows']} vs {exp['rows']}"
                    if got["columns"] != exp["columns"]:
                        extra += f"; columns {got['columns']} vs {exp['columns']}"
                print(f"  MISMATCH  {rel}{extra}")
                ok = False
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scratch", type=Path, help="default: $MUTANT_CACHE_SCRATCH")
    p.add_argument("--expected", type=Path, default=EXPECTED_FILE)
    p.add_argument("--group", action="append", choices=sorted(GROUPS),
                   help="restrict to these groups (repeatable)")
    p.add_argument("--record", action="store_true", help="write --expected from --scratch")
    p.add_argument("--accept-rounded", action="store_true")
    p.add_argument("--strict", action="store_true", help="also fail files marked unstable")
    a = p.parse_args(argv)
    root = (a.scratch or scratch()).expanduser()
    if a.record:
        record(root, a.expected, a.group)
        return 0
    expected = json.loads(a.expected.read_text())
    print(f"scratch {root}\nexpected {a.expected}")
    ok = verify(root, expected, a.group, accept_rounded=a.accept_rounded, strict=a.strict)
    print("ALL MATCH" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

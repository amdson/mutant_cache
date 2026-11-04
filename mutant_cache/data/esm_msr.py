"""ESM-MSR's Megascale protocol -> canonical tables (TODO_ESM_MSR_EXPERIMENT.md §1).

Upstream: github.com/SKTeamLab/esm-msr @ ``REPO_COMMIT``, data archive
Zenodo 10.5281/zenodo.21539277 (``preprocessed_data.tar.gz``, MIT per the Zenodo
record), preprint biorxiv 10.64898/2026.06.04.730231. Both live under
``$MUTANT_CACHE_SCRATCH/esm_msr/``; no upstream code is run and no upstream
pickle is loaded (``hyperopt_splits.csv`` is parsed with ``ast.literal_eval``).

**Rows** reproduce ``MegaScaleDatasetPreprocessor.preprocess`` + ``_filter_data``
(src/esm_msr/preprocess_megascale.py:127-154, 236-239), in order, each step
logged in ``audit/exclusions.csv``:

1. ``ddG_ML`` numeric (drops the ``"-"`` out-of-range rows)
2. drop ``mut_type`` containing ``wt`` / ``ins`` / ``del``
3. drop ``WT_name == "1UBQ.pdb_L43A"`` (upstream: mislabelled)
4. ``code_wt`` = WT_name before ``.pdb``; ``code`` = code_wt[_background];
   duplicates of (aa_seq, mut_type, WT_name) averaged; ``|`` -> ``_``
5. drop PROSS backgrounds, wt == mut tokens, and uids whose mutations conflict
   at one position (``is_improper_mutation`` on ``code_mut_type``, i.e. a
   variant re-mutating its engineered background site inconsistently)

**Proteins**: ``target_id`` = upstream ``code`` (a natural domain or one of its
engineered backgrounds). The split assigns ``code_wt`` to train / val / test
(``test`` == ``test_internal``; ``test_external`` lists non-Megascale benchmark
proteins and matches no ``code_wt``), and every background inherits it.

**Reference sequence**: the CSV's wild-type row ``aa_seq`` for that WT_name
(upstream instead reads the AlphaFold structure's sequence and applies the
background mutation). Every kept variant is checked twice: the WT letters match
the reference, and mutating the reference reproduces the row's own ``aa_seq``.
Upstream silently drops mismatching constituents (a double can become a
"single"); here a mismatch is an exclusion, logged.

**Labels**: raw ``ddG_ML`` kcal/mol, WT-relative, *positive = stabilizing*
(upstream losses.py / stats.py convention; same sign as ``tsuboyama.py``).
No WT rows are kept: models are zero-anchored, d(WT) = 0.

Not reproduced (training-time, not data): reversion / mut_ctx augmentation (off
in the release config), the 2-D (ddG_additive, dddG) rebalancing of doubles in
upstream's sampler, and the three external benchmarks (ptmuld, s461, ssym)
inside upstream's validation metric.

Run: ``python -m mutant_cache.data.esm_msr``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .mutations import format_mutant, parse_mutant
from .paths import processed_dir, scratch
from .schema import rank_normalize, write_dataset

DATASET = "esm_msr_mega"
SPLIT = "esm_msr_hyperopt"
REPO_COMMIT = "c34c315913b539070095b8af63b4bd7fbcd503b7"
ARCHIVE_MD5 = "56ae9af3f800065b7a7ff4f4a42b7b92"
CSV_NAME = "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
EXCLUDED_WT_NAMES = ("1UBQ.pdb_L43A",)

# Paper / release-notebook reconciliation targets (TODO §1: checks, not filters)
EXPECTED = {"test": {"rows": 43_700, "singles": 32_968, "doubles": 10_732},
            "test_doubles_missing_a_single": 556}

_TOKEN = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_IMPROPER = re.compile(r"(?<!\d)([A-Z])(\d+)([A-Z])")


def root() -> Path:
    return scratch() / "esm_msr"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    h.update(p.read_bytes())
    return h.hexdigest()


def is_fake_mutation(mut_type: str) -> bool:
    """preprocess_megascale.py:21-35."""
    for tok in mut_type.split(":"):
        m = re.match(r"([A-Za-z])(\d+)([A-Za-z])", tok)
        if m and m.group(1) == m.group(3):
            return True
    return False


def is_improper_mutation(uid: str) -> bool:
    """preprocess_megascale.py:38-97 (True = conflict = excluded)."""
    by_pos: dict[int, list[tuple[str, str]]] = {}
    for m in _IMPROPER.finditer(uid):
        by_pos.setdefault(int(m.group(2)), []).append((m.group(1), m.group(3)))
    if not by_pos:
        return True
    for muts in by_pos.values():
        for (_, y1), (x2, _) in zip(muts, muts[1:]):
            if y1 != x2:
                return True
    return False


def load_split() -> dict[str, list[str]]:
    """``hyperopt_splits.csv``: one row per key, a stringified list per row."""
    p = root() / "repo" / "data" / "hyperopt_splits.csv"
    df = pd.read_csv(p, index_col=0)
    out = {k: ast.literal_eval(v) for k, v in df.iloc[:, 0].items()}
    if sorted(out["test"]) != sorted(out["test_internal"]):
        raise ValueError("hyperopt_splits: test != test_internal")
    return out


def _code_wt(wt_name: str) -> str:
    return wt_name.split(".pdb")[0].replace("|", "_")


def _code(wt_name: str) -> str:
    bg = wt_name.split(".pdb_")[1] if ".pdb_" in wt_name else None
    c = wt_name.split(".pdb")[0]
    return (f"{c}_{bg}" if bg is not None else c).replace("|", "_")


def _mutate(seq: str, subs) -> str:
    s = list(seq)
    for p, _, a in subs:
        s[p - 1] = a
    return "".join(s)


def build() -> dict:
    csv = root() / "preprocessed" / "data" / "tsuboyama" / CSV_NAME
    raw = pl.read_csv(csv, columns=["aa_seq", "mut_type", "WT_name", "ddG_ML"],
                      infer_schema_length=100_000).to_pandas()
    log: list[dict] = []

    def step(name, before, after, note=""):
        log.append({"step": name, "rows_before": len(before), "rows_after": len(after),
                    "removed": len(before) - len(after), "note": note})
        return after

    d = raw
    # reference sequences from the wild-type rows, before step 2 removes them
    wt_rows = d[d["mut_type"] == "wt"]
    ref_seq = wt_rows.groupby("WT_name")["aa_seq"].agg(lambda s: s.iloc[0])
    n_wt_seq_variants = wt_rows.groupby("WT_name")["aa_seq"].nunique()
    if (n_wt_seq_variants > 1).any():
        raise ValueError(f"WT rows disagree on aa_seq: {n_wt_seq_variants[n_wt_seq_variants > 1]}")

    d = d.assign(ddG_ML=pd.to_numeric(d["ddG_ML"], errors="coerce"))
    d = step("1 ddG_ML numeric", raw, d[d["ddG_ML"].notna()], 'drops "-" (out of dynamic range)')
    for tag in ("wt", "ins", "del"):
        d = step(f"2 mut_type contains {tag!r}", d, d[~d["mut_type"].str.contains(tag)])
    d = step("3 excluded WT_name", d, d[~d["WT_name"].isin(EXCLUDED_WT_NAMES)],
             ", ".join(EXCLUDED_WT_NAMES) + " (upstream: mislabelled rows)")
    before = d
    d = (d.groupby(["aa_seq", "mut_type", "WT_name"], sort=False, as_index=False)["ddG_ML"]
         .mean())
    d = step("4 duplicates averaged", before, d, "same (aa_seq, mut_type, WT_name)")
    d["code_wt"] = d["WT_name"].map(_code_wt)
    d["code"] = d["WT_name"].map(_code)
    d["mut_structure"] = d["WT_name"].map(lambda x: x.split(".pdb_")[1] if ".pdb_" in x else "")
    d = step("5a PROSS backgrounds", d, d[~d["mut_structure"].str.startswith("pross")])
    d = step("5b fake mutation (wt == mut)", d, d[~d["mut_type"].map(is_fake_mutation)])
    uid = d["code"] + "_" + d["mut_type"]
    d = step("5c improper (conflicting mutations at one site, incl. background)",
             d, d[~uid.map(is_improper_mutation)])

    # ── reconstruction against our references (stricter than upstream) ────────
    missing_ref = sorted(set(d["WT_name"]) - set(ref_seq.index))
    for name in missing_ref:      # no WT row: revert one single (tsuboyama.py does the same)
        r = d[(d["WT_name"] == name) & ~d["mut_type"].str.contains(":")].iloc[0]
        p, w, _ = (int(_TOKEN.match(r.mut_type).group(2)), _TOKEN.match(r.mut_type).group(1),
                   None)
        s = list(r.aa_seq)
        s[p - 1] = w
        ref_seq[name] = "".join(s)
    ok, why, canon = [], [], []
    for r in d.itertuples(index=False):
        seq = ref_seq[r.WT_name]
        try:
            subs = [(int(m.group(2)), m.group(1), m.group(3))
                    for m in (_TOKEN.fullmatch(t) for t in r.mut_type.split(":"))]
            bad = [f"{w}{p}" for p, w, _ in subs if p > len(seq) or seq[p - 1] != w]
            if bad:
                ok.append(False); why.append(f"WT letter mismatch {bad}"); canon.append(None)
            elif _mutate(seq, subs) != r.aa_seq:
                ok.append(False); why.append("mutated reference != row aa_seq"); canon.append(None)
            else:
                ok.append(True); why.append(""); canon.append(format_mutant(subs))
        except (AttributeError, ValueError) as e:
            ok.append(False); why.append(f"unparseable: {e}"); canon.append(None)
    d = d.assign(mutant=canon, _ok=ok, _why=why)
    bad_rows = d[~d["_ok"]]
    d = step("6 reconstruction (ours, stricter than upstream)", d, d[d["_ok"]],
             "; ".join(f"{k}: {v}" for k, v in bad_rows["_why"].str.split(" \\[").str[0]
                       .value_counts().items()))
    before = d
    d = d.drop_duplicates(["code", "mutant"], keep=False)
    d = step("7 same (code, canonical mutant) twice", before, d,
             "both copies dropped: equivalent strings with different labels")
    d["n_mut"] = d["mutant"].str.count(":") + 1

    # ── split ────────────────────────────────────────────────────────────────
    sp = load_split()
    assign = {}
    for k in ("train", "val", "test"):
        for pid in sp[k]:
            cw = pid.removesuffix(".pdb")
            if cw in assign:
                raise ValueError(f"{cw} in both {assign[cw]} and {k}")
            assign[cw] = k
    ext = {p.removesuffix(".pdb") for p in sp["test_external"]}
    if ext & set(d["code_wt"]):
        raise ValueError("test_external overlaps Megascale code_wt")
    unmatched = sorted(set(assign) - set(d["code_wt"]))
    d["split"] = d["code_wt"].map(assign)
    before = d
    d = d[d["split"].notna()]
    d = step("8 code_wt not in hyperopt train/val/test", before, d,
             f"{before.loc[before['split'].isna(), 'code_wt'].nunique()} code_wt unused")

    # ── tables ───────────────────────────────────────────────────────────────
    codes = d.drop_duplicates("code")[["code", "code_wt", "WT_name", "split"]]
    targets = pl.DataFrame({"target_id": codes["code"].tolist(),
                            "sequence": [ref_seq[w] for w in codes["WT_name"]],
                            "uniprot": [None] * len(codes),
                            "source": ["tsuboyama/esm_msr"] * len(codes)})
    variants = pl.from_pandas(pd.DataFrame({
        "score_set_id": "esm_msr_" + d["code"], "target_id": d["code"],
        "mutant": d["mutant"], "n_mut": d["n_mut"].astype("int32"),
        "score": d["ddG_ML"].astype(float), "score_se": np.nan, "assay_type": "ddG"}))
    variants = rank_normalize(variants.with_columns(pl.col("score_se").cast(pl.Float64)))
    notes = [f"{r['step']}: -{r['removed']:,} ({r['note']})" if r["note"]
             else f"{r['step']}: -{r['removed']:,}" for r in log]
    report = write_dataset(DATASET, targets, variants, None, notes)

    out = processed_dir(DATASET)
    (out / "splits").mkdir(exist_ok=True)
    split_df = codes.rename(columns={"code": "target_id"})[["target_id", "code_wt", "split"]]
    split_df.to_parquet(out / "splits" / f"{SPLIT}.parquet", index=False)
    audit = audit_tables(d, ref_seq, codes)
    ad = out / "audit"
    ad.mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(ad / "exclusions.csv", index=False)
    bad_rows.drop(columns=["_ok"]).to_csv(ad / "reconstruction_failures.csv", index=False)
    for k, v in audit.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(ad / f"{k}.csv", index=False)
    prov = provenance(csv)
    prov.update(unmatched_split_code_wt=unmatched, missing_wt_row=missing_ref,
                reconciliation=audit["reconciliation"])
    (ad / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return {"report": report, "log": log, "audit": audit, "provenance": prov}


def audit_tables(d: pd.DataFrame, ref_seq, codes) -> dict:
    """Per-partition counts, complete A/B/AB groups, overlap checks."""
    rows, groups = [], []
    for k in ("train", "val", "test"):
        s = d[d["split"] == k]
        rows.append({"split": k, "code_wt": s["code_wt"].nunique(), "code": s["code"].nunique(),
                     "unique_reference_sequences": len({ref_seq[w] for w in s["WT_name"].unique()}),
                     "rows": len(s), "singles": int((s["n_mut"] == 1).sum()),
                     "doubles": int((s["n_mut"] == 2).sum()),
                     "ddG_min": s["ddG_ML"].min(), "ddG_q05": s["ddG_ML"].quantile(.05),
                     "ddG_median": s["ddG_ML"].median(), "ddG_q95": s["ddG_ML"].quantile(.95),
                     "ddG_max": s["ddG_ML"].max(), "frac_stabilizing": (s["ddG_ML"] > 0).mean()})
        singles = set(zip(s.loc[s["n_mut"] == 1, "code"], s.loc[s["n_mut"] == 1, "mutant"]))
        dbl = s[s["n_mut"] == 2]
        have = [all((c, f"{w}{p}{a}") in singles for p, w, a in parse_mutant(m))
                for c, m in zip(dbl["code"], dbl["mutant"])]
        groups.append({"split": k, "doubles": len(dbl), "complete_groups": int(sum(have)),
                       "doubles_missing_a_single": len(dbl) - int(sum(have))})
    split_summary = pd.DataFrame(rows).merge(pd.DataFrame(groups), on=["split", "doubles"])

    # exact sequence / genotype overlap across partitions
    seq_of = {c: ref_seq[w] for c, w in zip(codes["code"], codes["WT_name"])}
    part = dict(zip(codes["code"], codes["split"]))
    by_seq: dict[str, set] = {}
    for c, s in seq_of.items():
        by_seq.setdefault(s, set()).add(part[c])
    shared_seq = sum(len(v) > 1 for v in by_seq.values())
    full = {}
    for k in ("train", "val", "test"):
        s = d[d["split"] == k]
        full[k] = {_mutate(seq_of[c], parse_mutant(m)) for c, m in zip(s["code"], s["mutant"])}
    overlap = pd.DataFrame([{"a": a, "b": b,
                             "shared_reference_sequences": sum(
                                 {a, b} <= v for v in by_seq.values()),
                             "shared_full_variant_sequences": len(full[a] & full[b])}
                            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))])
    t = split_summary.set_index("split").loc["test"]
    recon = {"test_rows": int(t["rows"]), "test_singles": int(t["singles"]),
             "test_doubles": int(t["doubles"]),
             "test_doubles_missing_a_single": int(t["doubles_missing_a_single"]),
             "expected": EXPECTED,
             "matches": (int(t["rows"]) == EXPECTED["test"]["rows"]
                         and int(t["singles"]) == EXPECTED["test"]["singles"]
                         and int(t["doubles"]) == EXPECTED["test"]["doubles"]
                         and int(t["doubles_missing_a_single"])
                         == EXPECTED["test_doubles_missing_a_single"]),
             "reference_sequences_in_more_than_one_split": shared_seq}
    proteins = (d.groupby(["split", "code_wt", "code"])
                .agg(rows=("mutant", "size"), singles=("n_mut", lambda x: int((x == 1).sum())),
                     doubles=("n_mut", lambda x: int((x == 2).sum())),
                     length=("WT_name", lambda x: len(ref_seq[x.iloc[0]])))
                .reset_index())
    return {"split_summary": split_summary, "overlap": overlap, "proteins": proteins,
            "reconciliation": recon}


def homology_audit(identity: float = 0.30) -> pd.DataFrame:
    """Cross-partition pairs at or above ``identity`` with upstream's local
    alignment (BLOSUM62, gap open -10 / extend -1, identity over the shorter
    sequence; split_tsuboyama.py:134-163). Reported, never used to filter."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score, aligner.extend_gap_score = -10, -1
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

    out = processed_dir(DATASET)
    t = pd.read_parquet(out / "targets.parquet")
    sp = pd.read_parquet(out / "splits" / f"{SPLIT}.parquet")
    t = t.merge(sp, on="target_id").drop_duplicates("code_wt")    # one per natural domain
    rows = []
    recs = list(t[["code_wt", "split", "sequence"]].itertuples(index=False))
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if a.split == b.split:
                continue
            aln = next(iter(aligner.align(a.sequence, b.sequence)), None)
            if aln is None:
                continue
            x, y = aln[0], aln[1]
            matches = sum(1 for u, v in zip(x, y) if u == v and u != "-")
            ident = matches / min(len(a.sequence), len(b.sequence))
            if ident >= identity:
                rows.append({"a": a.code_wt, "a_split": a.split, "b": b.code_wt,
                             "b_split": b.split, "identity": round(ident, 4)})
    df = pd.DataFrame(rows, columns=["a", "a_split", "b", "b_split", "identity"])
    df.to_csv(out / "audit" / f"homology_ge{int(identity * 100)}.csv", index=False)
    return df


def provenance(csv: Path) -> dict:
    repo = root() / "repo"
    try:
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        head = None
    arch = root() / "preprocessed_data.tar.gz"
    return {"repo_commit": head, "repo_commit_expected": REPO_COMMIT,
            "archive": str(arch), "archive_md5": _md5(arch) if arch.exists() else None,
            "archive_md5_expected": ARCHIVE_MD5,
            "zenodo": "10.5281/zenodo.21539277 (v0.1, 2026-06-04, MIT)",
            "preprint": "10.64898/2026.06.04.730231",
            "source_csv": str(csv), "source_csv_md5": _md5(csv),
            "split_file": str(repo / "data" / "hyperopt_splits.csv"),
            "split_file_sha1": _sha1(repo / "data" / "hyperopt_splits.csv"),
            "upstream_data": "Tsuboyama et al. 2023 Nature, Dataset 2+3 (Zenodo 7844779, "
                             "CC-BY-4.0); redistributed unchanged in the ESM-MSR archive"}


if __name__ == "__main__":
    res = build()
    print(res["report"])
    print(pd.DataFrame(res["log"]).to_string(index=False))
    for k in ("split_summary", "overlap"):
        print(res["audit"][k].to_string(index=False))
    print(json.dumps(res["audit"]["reconciliation"], indent=2))
    h = homology_audit()
    print(f"cross-partition pairs >= 30% identity: {len(h)}")
    if len(h):
        print(h.sort_values("identity", ascending=False).head(20).to_string(index=False))

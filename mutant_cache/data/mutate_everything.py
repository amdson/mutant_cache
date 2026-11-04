"""Released Mutate Everything CSVs -> canonical stability tables.

Run with explicit CSV paths; raw artifacts stay outside the repository. The
released test split is immutable. Validation is held out from training proteins
(including identical sequences under other names), never from cDNA2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .mutations import parse_mutant, format_mutant
from .paths import processed_dir
from .schema import rank_normalize, write_dataset

REPO_COMMIT = "034dd91e7a449e68b0f0993043bd32ce281f056c"
DATASET = "mutate_everything_cdna"
SPLIT = "mutate_everything_cdna"
DATA_ARCHIVE_ID = "1akfHt1WLbC345DgZwZ5kLeMUADq9XB7y"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize(raw, *, ddg_sign="destabilizing_positive"):
    """Return rows, target sequences and an exclusion audit; median replicates.

    ``ddg_sign`` is explicit: canonical ``score`` is positive stabilizing.
    No majority-sign heuristic is applied silently. WT mismatches are excluded,
    not repaired. Every excluded raw row retains its input row number.
    """
    required = {"pdb_id", "wt_seq", "mut_info", "ddg"}
    if required - set(raw):
        raise ValueError(f"missing columns: {sorted(required - set(raw))}")
    if ddg_sign not in ("destabilizing_positive", "stabilizing_positive"):
        raise ValueError("ddg_sign must be explicit")
    sequences = {}
    for tid, g in raw.dropna(subset=["pdb_id", "wt_seq"]).groupby("pdb_id"):
        seqs = g.wt_seq.str.upper().unique()
        if len(seqs) == 1:
            sequences[str(tid)] = seqs[0]
    rows, audit = [], []
    for row_no, r in enumerate(raw.to_dict("records")):
        reason = None
        try:
            tid = str(r["pdb_id"])
            if tid not in sequences:
                raise ValueError("missing_or_ambiguous_reference")
            seq = sequences[tid]
            if not isinstance(r["mut_info"], str):
                raise ValueError("missing_mutation")
            subs = parse_mutant(r["mut_info"].upper())
            if not subs or len({s[0] for s in subs}) != len(subs):
                raise ValueError("empty_or_conflicting_mutation")
            for pos, wt, alt in subs:
                if wt not in "ACDEFGHIKLMNPQRSTVWY" or alt not in "ACDEFGHIKLMNPQRSTVWY":
                    raise ValueError("noncanonical_substitution")
                if pos < 1 or pos > len(seq) or seq[pos-1] != wt:
                    raise ValueError("wt_mismatch")
                if wt == alt:
                    raise ValueError("self_substitution")
            ddg = float(r["ddg"])
            if not np.isfinite(ddg):
                raise ValueError("nonfinite_score")
            rows.append(dict(target_id=tid, mutant=format_mutant(subs), n_mut=len(subs),
                             score=ddg * (-1 if ddg_sign == "destabilizing_positive" else 1)))
        except (ValueError, TypeError) as exc:
            reason = str(exc)
        if reason:
            audit.append(dict(source_row=row_no, target_id=r["pdb_id"],
                              mutant=r["mut_info"], reason=reason))
    frame = pd.DataFrame(rows, columns=["target_id", "mutant", "n_mut", "score"])
    if len(frame):
        frame = frame.groupby(["target_id", "mutant", "n_mut"], as_index=False).score.median()
    return frame, sequences, pd.DataFrame(audit, columns=["source_row", "target_id", "mutant", "reason"])


def overlap(a, aseq, b, bseq):
    """Exact ID and sequence overlaps. Sequence identity clustering is separate."""
    at, bt = set(a.target_id), set(b.target_id)
    return {"target_ids": sorted(at & bt), "sequence_pairs": sorted(
        (x, y) for x in at for y in bt if aseq[x] == bseq[y])}


def audit_external(frame, sequences, assignments, external_dir):
    """Cross-split exact-sequence overlap in both directions and joined sign audit.

    External splits must contain target_id/split. Homology is not inferred from
    identifiers. Joined labels compare the same reference sequence and mutation.
    """
    d = Path(external_dir)
    targets = pd.read_parquet(d / "targets.parquet")
    variants = pd.read_parquet(d / "variants.parquet")
    external_sequences = dict(zip(targets.target_id, targets.sequence))
    matches = []
    for split_path in sorted((d / "splits").glob("*.parquet")):
        splits = pd.read_parquet(split_path)
        if not {"target_id", "split"} <= set(splits):
            continue
        for r in splits.to_dict("records"):
            for tid, part in assignments.items():
                if sequences[tid] == external_sequences.get(r["target_id"]):
                    matches.append(dict(our_target=tid, our_split=part,
                        external_target=r["target_id"], external_split=r["split"],
                        split_file=str(split_path)))
    a = frame.assign(sequence=frame.target_id.map(sequences))
    b = variants.assign(sequence=variants.target_id.map(external_sequences))
    a = a.groupby(["sequence", "mutant"], as_index=False).score.median()
    b = b.groupby(["sequence", "mutant"], as_index=False).score.median()
    joined = a.merge(b, on=["sequence", "mutant"], suffixes=("_ours", "_external"))
    sign = {"joined_rows": len(joined), "verified": False}
    if len(joined):
        x, y = joined.score_ours.to_numpy(), joined.score_external.to_numpy()
        sign.update(same_sign_mae=float(np.abs(x-y).mean()), flipped_sign_mae=float(np.abs(x+y).mean()),
                    verified=bool(np.allclose(x, y, atol=1e-4, rtol=0)))
    return matches, sign


def build(train_csv, test_csv, *, dataset=DATASET, val_fraction=0.1, seed=0,
          ddg_sign="destabilizing_positive", checkpoint=None, audit_dataset_dir=None):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    paths = {"train": Path(train_csv), "test": Path(test_csv)}
    loaded = {k: normalize(pd.read_csv(p), ddg_sign=ddg_sign) for k, p in paths.items()}
    tr, ts, ta = loaded["train"]
    te, es, ea = loaded["test"]
    ov = overlap(tr, ts, te, es)
    if ov["target_ids"] or ov["sequence_pairs"]:
        raise ValueError(f"released train/test overlap: {ov}")
    unique_sequences = sorted({ts[t] for t in tr.target_id})
    if len(unique_sequences) < 2:
        raise ValueError("need at least two training sequence groups for validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sequences)
    nval = min(len(unique_sequences)-1, max(1, round(len(unique_sequences)*val_fraction)))
    val = set(unique_sequences[:nval])
    assignments = {t: ("val" if ts[t] in val else "train") for t in tr.target_id.unique()}
    assignments.update({t: "test" for t in te.target_id.unique()})
    seqs = ts | es
    frame = pd.concat([tr, te], ignore_index=True)
    frame["score_set_id"] = frame.target_id
    frame["assay_type"] = "stability"
    targets = pl.DataFrame([dict(target_id=t, sequence=seqs[t], source="mutate_everything")
                            for t in sorted(assignments)])
    report = write_dataset(dataset, targets, rank_normalize(pl.from_pandas(frame)))
    out = processed_dir(dataset)
    (out / "splits").mkdir(exist_ok=True)
    pl.DataFrame({"target_id": list(assignments), "split": list(assignments.values())}).write_parquet(
        out / "splits" / f"{SPLIT}.parquet")
    for k, (_, _, audit) in loaded.items():
        audit.to_csv(out / f"exclusions_{k}.csv", index=False)
        f = loaded[k][0].copy()
        f["wt_seq"] = f.target_id.map(seqs)
        f.to_csv(out / f"comparison_labels_{k}.csv", index=False)
    files = {k: {"path": str(p.resolve()), "sha256": sha256(p)} for k, p in paths.items()}
    if checkpoint:
        files["checkpoint"] = {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)}
    note = dict(upstream_commit=REPO_COMMIT, files=files, ddg_sign=ddg_sign,
                canonical_sign="positive_stabilizing", seed=seed, val_fraction=val_fraction,
                overlap=ov, identity_ceiling_verified=False,
                partitions={k: {"rows": int(frame.target_id.map(assignments).eq(k).sum()),
                                "proteins": sum(s == k for s in assignments.values())}
                            for k in ("train", "val", "test")},
                released_shapes={k: {"raw_rows": len(pd.read_csv(paths[k])), "kept_rows": len(f),
                                    "proteins": int(f.target_id.nunique()),
                                    "depth_counts": f.n_mut.value_counts().to_dict()}
                                 for k, (f, _, _) in loaded.items()})
    if audit_dataset_dir:
        matches, sign = audit_external(frame, seqs, assignments, audit_dataset_dir)
        pd.DataFrame(matches).to_csv(out / "external_overlap.csv", index=False)
        note["external_sign_audit"] = sign
        note["external_dataset_dir"] = str(audit_dataset_dir)
    (out / "manifest.json").write_text(json.dumps(note, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--test-csv", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--audit-dataset-dir", type=Path)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ddg-sign", choices=("destabilizing_positive", "stabilizing_positive"),
                   default="destabilizing_positive")
    print(build(**vars(p.parse_args())))


if __name__ == "__main__":
    main()

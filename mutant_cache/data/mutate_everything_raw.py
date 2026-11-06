"""Mutate Everything release archive -> raw cDNA tables, external panels, holdouts.

Portable equivalents of three raw scripts kept as run in ``experiments/``
(they hardcode the original scratch root):

- ``fetch`` / ``extract``: ``experiments/mutate_everything_prepare.py``
- ``external``: ``experiments/mutate_everything_external_prepare.py``
- ``holdouts``: the split block of ``experiments/start_mutate_everything_accuracy_v2.py``
  (that script also launches GPU workers; only the split logic is here)

Each function reproduces its script's outputs byte for byte on the same inputs;
every path comes from ``$MUTANT_CACHE_SCRATCH`` (``paths.scratch()``) unless
passed explicitly. The canonical tables themselves are built by
``python -m mutant_cache.data.mutate_everything``.

    python -m mutant_cache.data.mutate_everything_raw fetch      # download zip if absent
    python -m mutant_cache.data.mutate_everything_raw extract    # sha256 check + cDNA tables
    python -m mutant_cache.data.mutate_everything_raw external   # S669 / ProTherm / cDNA1 labels
    python -m mutant_cache.data.mutate_everything_raw holdouts   # seed 1/2 protein holdouts
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .mutate_everything import DATA_ARCHIVE_ID, DATASET, SPLIT, normalize
from .mutations import format_mutant, mutate, parse_mutant
from .paths import scratch

ARCHIVE_NAME = "mutate_everything_data.zip"
# sha256 of the archive the processed tables were built from (2026-11-04).
ARCHIVE_SHA256 = "4e4a579902db8308196f4c1bf0fed25ee31b1573c28613525ff2f3def2350f9f"
# The request experiments/mutate_everything_prepare.py actually made.
DRIVE_URL = "https://drive.usercontent.google.com/download"
CDNA_MEMBERS = ("cdna_train.csv", "cdna2_test.csv")
EXTERNAL_PANELS = (("s669", "s669.csv"), ("ptmul", "protherm_multiple.csv"),
                   ("cdna1", "cdna1_test.csv"))
HOLDOUT_SEEDS = (1, 2)
HOLDOUT_VAL_FRACTION = 0.1


def me_root() -> Path:
    return scratch() / "mutate_everything"


def processed_root() -> Path:
    return scratch() / "processed" / DATASET


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root: Path | None = None) -> Path:
    """Download the release zip from Google Drive, only if it is absent."""
    root = Path(root or me_root())
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE_NAME
    if archive.exists():
        print("archive present", archive, archive.stat().st_size, flush=True)
        return archive
    import requests

    response = requests.get(DRIVE_URL, params={
        "id": DATA_ARCHIVE_ID, "export": "download", "confirm": "t"}, stream=True, timeout=120)
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        raise RuntimeError("Drive returned HTML instead of the release archive")
    partial = archive.with_suffix(".zip.partial")
    with partial.open("wb") as f:
        for chunk in response.iter_content(1 << 20):
            f.write(chunk)
    partial.replace(archive)
    print("archive", archive, archive.stat().st_size, flush=True)
    return archive


def extract(root: Path | None = None, *, expected_sha256: str | None = ARCHIVE_SHA256) -> dict:
    """Write ``archive.sha256`` and ``data/{cdna_train,cdna2_test}.csv``.

    Unlike the raw script, the archive hash is checked *before* extraction.
    """
    root = Path(root or me_root())
    archive = root / ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(f"{archive}: run `fetch` first")
    digest = _sha256(archive)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"{archive}: sha256 {digest} != expected {expected_sha256}")
    (root / "archive.sha256").write_text(f"{digest}  {archive.name}\n")
    out = {}
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if m.filename.endswith(
            tuple("/" + n for n in CDNA_MEMBERS)) and "__MACOSX" not in m.filename]
        if len(members) != 2:
            raise RuntimeError(f"expected two cDNA tables, got {[m.filename for m in members]}")
        dest = root / "data"
        dest.mkdir(exist_ok=True)
        for m in members:
            output = dest / Path(m.filename).name
            output.write_bytes(z.read(m))
            out[output.name] = output
            print("extracted", m.filename, "->", output, output.stat().st_size, flush=True)
    print("archive_sha256", digest, flush=True)
    return out


def external(root: Path | None = None, processed: Path | None = None) -> dict:
    """Released external panels -> ``external/{tag}_labels.csv`` etc. + ``inventory.json``.

    Needs the canonical tables (``targets.parquet``, the protein split) first.
    No filtering by results; the S669 reverse panel is the explicit
    mutant-reference / negated-label construction.
    """
    root = Path(root or me_root())
    processed = Path(processed or processed_root())
    out = root / "external"
    out.mkdir(exist_ok=True)
    targets = pd.read_parquet(processed / "targets.parquet")
    split = pd.read_parquet(processed / "splits" / f"{SPLIT}.parquet")
    training = targets.merge(split, on="target_id").query("split == 'train'")
    inventory = {}
    with zipfile.ZipFile(root / ARCHIVE_NAME) as z:
        for tag, base in EXTERNAL_PANELS:
            name = next(n for n in z.namelist() if n.endswith("/" + base) and "__MACOSX" not in n)
            raw_bytes = z.read(name)
            (out / base).write_bytes(raw_bytes)
            raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
            frame, seqs, audit = normalize(raw)
            frame["wt_seq"] = frame.target_id.map(seqs)
            frame["reference_id"] = frame.target_id
            frame.to_csv(out / f"{tag}_labels.csv", index=False)
            audit.to_csv(out / f"{tag}_exclusions.csv", index=False)
            overlaps = []
            for tid in frame.target_id.unique():
                for r in training.itertuples():
                    if seqs[tid] in r.sequence or r.sequence in seqs[tid]:
                        overlaps.append(dict(external_target=tid, train_target=r.target_id,
                                             exact=seqs[tid] == r.sequence))
            inventory[tag] = dict(raw_rows=len(raw), rows=len(frame), proteins=int(frame.target_id.nunique()),
                                  depths=frame.n_mut.value_counts().sort_index().to_dict(),
                                  raw_sha256=hashlib.sha256(raw_bytes).hexdigest(), excluded_rows=len(audit),
                                  exact_or_containment_overlaps=overlaps, homology_audit="not yet recomputed")
            if tag == "s669":
                reverse = frame.copy()
                reverse["forward_mutant"] = frame.mutant
                reverse["wt_seq"] = [mutate(s, m) for s, m in zip(frame.wt_seq, frame.mutant)]
                reverse["reference_id"] = frame.target_id + "::" + frame.mutant
                reverse["mutant"] = [format_mutant([(p, a, w) for p, w, a in parse_mutant(m)])
                                     for m in frame.mutant]
                reverse["score"] = -frame.score
                reverse.to_csv(out / "s669_reverse_labels.csv", index=False)
                for r in reverse.itertuples():
                    assert mutate(r.wt_seq, r.mutant) == seqs[r.target_id]
                inventory["s669_reverse"] = dict(
                    rows=len(reverse), proteins=int(reverse.target_id.nunique()),
                    unique_references=int(reverse.reference_id.nunique()),
                    convention="mutant reference, reverse substitution, negated label; "
                               "no result-based calibration")
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps(inventory, indent=2), flush=True)
    return inventory


def holdout_split(split: pd.DataFrame, sequences: dict, seed: int,
                  val_fraction: float = HOLDOUT_VAL_FRACTION) -> pd.DataFrame:
    """Re-split the development (non-test) proteins by unique sequence.

    Sorted unique dev sequences, shuffled with ``default_rng(seed)``; the
    first ``round(n * val_fraction)`` are val. Test rows are unchanged; row
    order follows ``split``.
    """
    dev = split.loc[split["split"] != "test", "target_id"]
    seqs = sorted({sequences[t] for t in dev})
    np.random.default_rng(seed).shuffle(seqs)
    val = set(seqs[:round(len(seqs) * val_fraction)])
    extra = split.copy()
    extra.loc[extra["split"] != "test", "split"] = ["val" if sequences[t] in val else "train" for t in dev]
    assert set(extra.loc[extra["split"] == "test", "target_id"]) == set(
        split.loc[split["split"] == "test", "target_id"])
    return extra


def holdouts(processed: Path | None = None, seeds=HOLDOUT_SEEDS, *, overwrite: bool = False) -> dict:
    """Write ``splits/mutate_everything_cdna_holdout{seed}.parquet``.

    An existing file must equal the recomputed split (``DataFrame.equals``,
    as the original script asserted); it is rewritten only with ``overwrite``.
    """
    data = Path(processed or processed_root())
    sp = pd.read_parquet(data / "splits" / f"{SPLIT}.parquet")
    sequences = pd.read_parquet(data / "targets.parquet").set_index("target_id").sequence.to_dict()
    status = {}
    for seed in seeds:
        extra = holdout_split(sp, sequences, seed)
        path = data / "splits" / f"{SPLIT}_holdout{seed}.parquet"
        if path.exists() and not overwrite:
            if not pd.read_parquet(path).equals(extra):
                raise RuntimeError(f"{path} differs from the recomputed seed-{seed} holdout; "
                                   "inspect it, then rerun with overwrite (FORCE=1)")
            status[path.name] = "exists, identical"
        else:
            extra.to_parquet(path, index=False)
            status[path.name] = "written"
        counts = extra.split.value_counts().to_dict()
        print(path, status[path.name], counts, flush=True)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("extract")
    sub.add_parser("external")
    h = sub.add_parser("holdouts")
    h.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "extract":
        extract()
    elif a.cmd == "external":
        external()
    else:
        holdouts(overwrite=a.overwrite)


if __name__ == "__main__":
    main()

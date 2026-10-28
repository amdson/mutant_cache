"""
mutant_cache/data/metl.py
-------------------------
METL (Gelman et al., Nat Methods 2025) loader: the Rosetta *synthetic* sets
(~12.5M variants for GB1, ~20M for avGFP) and the experimental DMS sets, with
METL's own splits preserved so numbers compare to the paper directly.

Two index conventions, verified against the wild type rather than assumed
(HANDOFF §5.1 is the same bug class):

* ``metl-pub/data/dms_data/<p>/<p>.tsv`` — **0-indexed** residue positions.
* ``rosetta/<p>/<p>.tsv``               — **1-indexed** residue positions.

`_offset` re-derives the convention from the data on every build and raises
unless one offset reproduces the wild-type residue at *every* position, so a
silent off-by-one cannot survive a build.

Mutation strings are comma-separated in both files (``T17P,T44L``) and are
rewritten to the canonical colon-separated 1-based form (`format_mutant`).

The Rosetta tsv is 5.8 GB / 69 columns for GB1, so `build_rosetta` streams it
once, keeping only the rows named by the splits it is asked for and only the
columns it needs (``mutations`` and the target attribute).

Splits written to ``processed/splits/``, one table per train size:
``metl_<p>_rosetta_ds<N>``  — train/val from METL's ``resampled_1/ds<N>``,
test from a fixed subsample of METL's shared supertest holdout (``stest.txt``,
627,819 rows for GB1; evaluating all of it every epoch is wasteful and the
subsample is the same rows for every size, so sizes stay comparable).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from .mutations import format_mutant
from .paths import raw_dir, splits_dir
from .schema import BuildReport, rank_normalize, write_dataset

METL_DIR = raw_dir("metl")
DMS_DIR = METL_DIR / "metl-pub" / "data" / "dms_data"
ROSETTA_DIR = METL_DIR / "rosetta"

# Rosetta attribute used as the supervised target. METL reports held-out
# Spearman on the total score; the individual terms are in the same tsv.
TARGET_ATTR = "total_score"
TEST_SUBSAMPLE = 50_000       # rows of stest used as the eval split
TEST_SEED = 0


@dataclass
class _Protein:
    name: str
    wt: str
    rosetta: Path
    dms: Path


def protein(name: str) -> _Protein:
    cfg = yaml.safe_load((DMS_DIR / "datasets.yml").read_text())
    if name not in cfg:
        raise ValueError(f"{name} not in datasets.yml ({sorted(cfg)})")
    e = cfg[name]
    if int(e.get("wt_ofs", 0)) != 0:
        raise ValueError(f"{name}: wt_ofs={e['wt_ofs']} not handled")
    return _Protein(name=name, wt=e["wt_aa"],
                    rosetta=ROSETTA_DIR / name, dms=DMS_DIR / name)


# --- index convention -------------------------------------------------------
def _parse(mut: str) -> list[tuple[int, str, str]]:
    """``T17P,T44L`` -> [(17, 'T', 'P'), (44, 'T', 'L')]; position as written."""
    return [(int(m[1:-1]), m[0], m[-1]) for m in mut.split(",") if m]


def _offset(wt: str, muts, label: str) -> int:
    """Residue-index offset (0 or 1) that reproduces the wild type everywhere.

    Returns the offset such that written position ``p`` is ``wt[p - offset]``.
    Raises unless exactly one offset matches every sampled substitution, so the
    convention is proved from the data, never assumed.
    """
    rates = {}
    for off in (0, 1):
        ok = n = 0
        for mut in muts:
            for p, ref, _ in _parse(mut):
                i = p - off
                ok += 0 <= i < len(wt) and wt[i] == ref
                n += 1
        rates[off] = ok / max(n, 1)
    good = [o for o, r in rates.items() if r == 1.0]
    if len(good) != 1:
        raise ValueError(
            f"{label}: no unique index convention (match rates {rates}); "
            "refusing to guess — the wild type must be reproduced exactly")
    return good[0]


def _canonical(mut: str, wt: str, off: int) -> str:
    """To the repo's 1-based colon-separated form, checking the WT residue."""
    subs = []
    for p, ref, alt in _parse(mut):
        i = p - off
        if not (0 <= i < len(wt)) or wt[i] != ref:
            raise ValueError(f"{mut}: expected {ref} at residue {p} (offset {off}), "
                             f"sequence has {wt[i] if 0 <= i < len(wt) else 'out of range'}")
        subs.append((i + 1, ref, alt))      # 1-based: residue i+1 == token i+1
    return format_mutant(subs)


# --- Rosetta synthetic ------------------------------------------------------
def _split_dir(p: _Protein, n: int) -> Path:
    pat = f"resampled_ds{n}_*"
    hits = sorted((p.rosetta / "splits" / "resampled_1").glob(pat))
    if not hits:
        raise FileNotFoundError(f"no {pat} under {p.rosetta}/splits/resampled_1")
    reps = sorted(hits[0].glob("*_rep_*"))
    return reps[0] if reps else hits[0]


def _indices(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def _stest(p: _Protein) -> np.ndarray:
    """The shared supertest holdout, subsampled deterministically."""
    hits = sorted((p.rosetta / "splits").glob("supertest_*.txt"))
    if not hits:
        raise FileNotFoundError(f"no supertest_*.txt under {p.rosetta}/splits")
    idx = _indices(hits[0])
    if len(idx) > TEST_SUBSAMPLE:
        rng = np.random.default_rng(TEST_SEED)
        idx = np.sort(rng.choice(idx, TEST_SUBSAMPLE, replace=False))
    return idx


def build_rosetta(name: str = "gb1", sizes=(1000, 4000, 16000, 64000),
                  attr: str = TARGET_ATTR) -> BuildReport:
    """Rosetta synthetic variants for ``sizes`` plus the supertest eval rows.

    One dataset (``metl_<name>_rosetta``) holding the union of every size's
    train/val rows and the test subsample, plus one split table per size.
    """
    p = protein(name)
    tsv = p.rosetta / f"{name}.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"{tsv} — extract processed-{name}.tar.gz first")

    per_size = {n: (_indices(_split_dir(p, n) / "train.txt"),
                    _indices(_split_dir(p, n) / "val.txt")) for n in sizes}
    test_idx = _stest(p)
    wanted = np.unique(np.concatenate(
        [test_idx, *[a for tv in per_size.values() for a in tv]]))
    want = set(wanted.tolist())

    # one streaming pass: row index -> (mutations, attr)
    with open(tsv) as f:
        header = next(f).rstrip("\n").split("\t")
        c_mut, c_attr = header.index("mutations"), header.index(attr)
        rows = {}
        for i, line in enumerate(f):
            if i in want:
                fields = line.rstrip("\n").split("\t")
                rows[i] = (fields[c_mut], fields[c_attr])
    missing = len(want) - len(rows)
    if missing:
        raise ValueError(f"{tsv}: {missing} split indices beyond the file")

    order = sorted(rows)
    raw_muts = [rows[i][0] for i in order]
    off = _offset(p.wt, raw_muts[:20000], f"rosetta/{name}")
    muts = [_canonical(m, p.wt, off) for m in raw_muts]
    scores = [float(rows[i][1]) for i in order]
    pos_of = {i: j for j, i in enumerate(order)}

    tid, ssid = f"metl_{name}", f"metl_{name}_rosetta"
    targets = pl.DataFrame({"target_id": [tid], "sequence": [p.wt],
                            "uniprot": [None], "source": ["metl_rosetta"]})
    variants = rank_normalize(pl.DataFrame({
        "score_set_id": ssid, "target_id": tid, "mutant": muts,
        "n_mut": [m.count(":") + 1 if m else 0 for m in muts],
        "score": scores, "score_se": None, "assay_type": "rosetta_energy",
    }))
    rep = write_dataset(ssid, targets, variants,
                        notes=[f"METL Rosetta synthetic, attribute {attr}; "
                               f"index offset {off} verified against the wild type",
                               f"sizes {sorted(sizes)}; test = {len(test_idx):,} of "
                               f"METL's supertest holdout (seed {TEST_SEED})"])

    for n, (tr, va) in per_size.items():
        lab = np.full(len(order), "unused", dtype=object)
        for ids, which in ((tr, "train"), (va, "val"), (test_idx, "test")):
            for i in ids:
                lab[pos_of[int(i)]] = which
        tbl = pl.DataFrame({"score_set_id": ssid, "mutant": muts,
                            "split": list(lab)}).filter(pl.col("split") != "unused")
        out = splits_dir() / f"{ssid}_ds{n}.parquet"
        tbl.write_parquet(out)
        print(f"  split {out.name}: "
              + " ".join(f"{k}={v:,}" for k, v in
                         sorted(tbl.group_by("split").len().rows())))
    return rep


# --- experimental DMS -------------------------------------------------------
def build_dms(name: str = "gb1") -> BuildReport:
    """METL's experimental DMS table for one protein (0-indexed positions)."""
    p = protein(name)
    d = pl.read_csv(p.dms / f"{name}.tsv", separator="\t")
    off = _offset(p.wt, d["variant"].to_list()[:20000], f"dms/{name}")
    muts = [_canonical(m, p.wt, off) for m in d["variant"].to_list()]

    tid, ssid = f"metl_{name}", f"metl_{name}_dms"
    targets = pl.DataFrame({"target_id": [tid], "sequence": [p.wt],
                            "uniprot": [None], "source": ["metl_dms"]})
    variants = rank_normalize(pl.DataFrame({
        "score_set_id": ssid, "target_id": tid, "mutant": muts,
        "n_mut": d["num_mutations"].cast(pl.Int32),
        "score": d["score"].cast(pl.Float64), "score_se": None,
        "assay_type": "dms_fitness",
    }))
    return write_dataset(ssid, targets, variants,
                         notes=[f"METL DMS; index offset {off} verified "
                                "against the wild type"])


DMS_SIZES = (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480)


def build_dms_resampled(name: str = "gb1", sizes=DMS_SIZES) -> list[Path]:
    """METL's DMS train-size grid as split tables, every replicate.

    ``splits/main/resampled/resampled_ds<N>_…/<…>_rep_<k>/{train,val,stest}.txt``
    are row indices into ``<p>.tsv``; written as
    ``metl_<p>_dms_ds<N>_r<k>.parquet`` against the ``metl_<p>_dms`` dataset
    (`build_dms`). Test is METL's supertest, all of it, identical for every
    size and replicate (checked), so numbers compare to the paper's figure.
    """
    p = protein(name)
    d = pl.read_csv(p.dms / f"{name}.tsv", separator="\t", columns=["variant"])
    raw = d["variant"].to_list()
    off = _offset(p.wt, raw[:20000], f"dms/{name}")
    muts = np.array([_canonical(m, p.wt, off) for m in raw], dtype=object)
    if len(set(muts.tolist())) != len(muts):
        raise ValueError(f"dms/{name}: duplicate mutants; splits would leak")
    ssid = f"metl_{name}_dms"

    root = p.dms / "splits" / "main"
    stest_ref = np.sort(_indices(next(root.parent.glob("supertest_*.txt"))))
    out = []
    for n in sizes:
        hits = sorted((root / "resampled").glob(f"resampled_ds{n}_*"))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected one resampled_ds{n}_* dir, got {hits}")
        reps = sorted(hits[0].glob("*_rep_*"), key=lambda r: int(r.name.rsplit("_", 1)[1]))
        for rd in reps:
            k = int(rd.name.rsplit("_", 1)[1])
            tr, va, te = (_indices(rd / f) for f in ("train.txt", "val.txt", "stest.txt"))
            if not np.array_equal(np.sort(te), stest_ref):
                raise ValueError(f"{rd.name}: stest differs from the shared supertest")
            if len(tr) + len(va) != n or np.intersect1d(np.concatenate([tr, va]), te).size:
                raise ValueError(f"{rd.name}: sizes {len(tr)}+{len(va)} != {n} or overlap with test")
            idx = np.concatenate([tr, va, te])
            lab = ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te)
            tbl = pl.DataFrame({"score_set_id": ssid, "mutant": muts[idx].tolist(),
                                "split": lab})
            f = splits_dir() / f"{ssid}_ds{n}_r{k}.parquet"
            tbl.write_parquet(f)
            out.append(f)
        print(f"  ds{n}: {len(reps)} reps, train {len(tr)} / val {len(va)} / test {len(te):,}")
    return out

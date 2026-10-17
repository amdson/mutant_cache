"""ProteinGym reference file -> the training exclusion list.

DATASETS.md: *exclude from training every protein that appears in ProteinGym
under any assay, then cluster-filter the remainder at 30% identity.* This
module builds ``$MUTANT_CACHE_SCRATCH/processed/exclusion/proteingym_overlap.parquet``.

Four passes, in increasing looseness, each recorded with its own
``overlap_type`` so a caller can tighten or loosen the exclusion without
rebuilding:

``exact``       identical sequence (whitespace- and case-normalized).
``substring``   one sequence contains the other. This is the pass that matters
                for Tsuboyama, whose targets are short natural *domains* that
                sit inside the full-length proteins ProteinGym scores.
``uniprot``     same UniProt accession, where both sides report one.
``mmseqs30``    same mmseqs2 cluster at 30% identity.

The default exclusion set is the union of all four; ``load_excluded_targets``
takes an ``overlap_types`` argument if you want a stricter definition.
"""

from __future__ import annotations

import polars as pl

from . import cluster as clust
from .paths import exclusion_dir, raw_dir
from .schema import load_targets

MIN_SEQ_ID = 0.3
COVERAGE = 0.5
OUT_NAME = "proteingym_overlap.parquet"


def load_reference() -> pl.DataFrame:
    p = raw_dir("proteingym") / "DMS_substitutions.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} — make -C datasets proteingym")
    return pl.read_csv(p, infer_schema_length=10_000)


def _norm(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "").replace("\n", "")


def build(datasets: list[str] | None = None, use_mmseqs: bool = True) -> pl.DataFrame:
    datasets = datasets or ["tsuboyama", "mavedb", "flip_gb1"]
    pg = load_reference()
    pg_rows = [
        (r["DMS_id"], r["UniProt_ID"], _norm(r["target_seq"]))
        for r in pg.iter_rows(named=True)
        if r["target_seq"]
    ]

    # ProteinGym has one row per assay; several assays share a protein.
    pg_by_seq: dict[str, list[str]] = {}
    for dms_id, _up, seq in pg_rows:
        pg_by_seq.setdefault(seq, []).append(dms_id)
    pg_by_uniprot: dict[str, list[str]] = {}
    for dms_id, up, _seq in pg_rows:
        if up:
            pg_by_uniprot.setdefault(up.split("_")[0].upper(), []).append(dms_id)
    pg_seqs = list(pg_by_seq)

    tgt_frames = []
    for ds in datasets:
        try:
            t = load_targets(ds).with_columns(pl.lit(ds).alias("dataset"))
        except FileNotFoundError:
            print(f"  {ds}: not built, skipping")
            continue
        tgt_frames.append(t)
    if not tgt_frames:
        raise RuntimeError("no built datasets to screen")
    targets = pl.concat(tgt_frames, how="vertical_relaxed")
    print(f"  screening {targets.height:,} targets against {len(pg_rows)} ProteinGym assays")

    hits: list[dict] = []

    # --- exact + substring (both directions) -------------------------------
    for r in targets.iter_rows(named=True):
        seq = _norm(r["sequence"])
        if not seq:
            continue
        if seq in pg_by_seq:
            for dms_id in pg_by_seq[seq]:
                hits.append(
                    dict(target_id=r["target_id"], dataset=r["dataset"],
                         overlap_type="exact", proteingym_dms_id=dms_id, identity=1.0)
                )
            continue
        for pseq in pg_seqs:
            if len(seq) <= len(pseq):
                contained = seq in pseq
            else:
                contained = pseq in seq
            if contained:
                for dms_id in pg_by_seq[pseq]:
                    hits.append(
                        dict(target_id=r["target_id"], dataset=r["dataset"],
                             overlap_type="substring", proteingym_dms_id=dms_id,
                             identity=1.0)
                    )
                break

    # --- uniprot -----------------------------------------------------------
    for r in targets.iter_rows(named=True):
        up = (r["uniprot"] or "").split("_")[0].split("-")[0].upper()
        if up and up in pg_by_uniprot:
            for dms_id in pg_by_uniprot[up]:
                hits.append(
                    dict(target_id=r["target_id"], dataset=r["dataset"],
                         overlap_type="uniprot", proteingym_dms_id=dms_id, identity=None)
                )

    # --- mmseqs2 30% -------------------------------------------------------
    if use_mmseqs and clust.available():
        records = [(f"PG|{d}", s) for d, _u, s in pg_rows]
        tid_of: dict[str, tuple[str, str]] = {}
        for i, r in enumerate(targets.iter_rows(named=True)):
            s = _norm(r["sequence"])
            if not s or len(s) < 10:
                continue
            name = f"T|{i}"
            tid_of[name] = (r["target_id"], r["dataset"])
            records.append((name, s))
        print(f"  mmseqs2 easy-cluster on {len(records):,} sequences "
              f"(min-seq-id {MIN_SEQ_ID}, c {COVERAGE}, cov-mode 1)")
        clusters = clust.cluster(records, min_seq_id=MIN_SEQ_ID, coverage=COVERAGE)
        if clusters:
            members: dict[str, list[str]] = {}
            for member, rep in clusters.items():
                members.setdefault(rep, []).append(member)
            for rep, mem in members.items():
                pg_ids = [m[3:] for m in mem if m.startswith("PG|")]
                tgts = [m for m in mem if m.startswith("T|")]
                if not pg_ids or not tgts:
                    continue
                for m in tgts:
                    tid, ds = tid_of[m]
                    for dms_id in pg_ids:
                        hits.append(
                            dict(target_id=tid, dataset=ds, overlap_type="mmseqs30",
                                 proteingym_dms_id=dms_id, identity=None)
                        )

    schema = {
        "target_id": pl.Utf8, "dataset": pl.Utf8, "overlap_type": pl.Utf8,
        "proteingym_dms_id": pl.Utf8, "identity": pl.Float64,
    }
    df = (
        pl.DataFrame(hits, schema=schema)
        if hits
        else pl.DataFrame({k: [] for k in schema}, schema=schema)
    ).unique()

    out = exclusion_dir() / OUT_NAME
    df.write_parquet(out)

    n_tot = targets.height
    excl = df["target_id"].n_unique()
    print(f"  -> {out}")
    print(f"  {excl:,} of {n_tot:,} targets overlap ProteinGym "
          f"({100 * excl / max(n_tot, 1):.1f}%)")
    if df.height:
        print(df.group_by(["dataset", "overlap_type"])
                .agg(pl.col("target_id").n_unique().alias("n_targets"))
                .sort(["dataset", "overlap_type"]))
    return df


def load_overlap() -> pl.DataFrame:
    p = exclusion_dir() / OUT_NAME
    if not p.exists():
        raise FileNotFoundError(f"{p} — python -m mutant_cache.data.proteingym")
    return pl.read_parquet(p)


def load_excluded_targets(overlap_types: tuple[str, ...] | None = None) -> set[str]:
    df = load_overlap()
    if overlap_types is not None:
        df = df.filter(pl.col("overlap_type").is_in(list(overlap_types)))
    return set(df["target_id"].to_list())


if __name__ == "__main__":
    build()

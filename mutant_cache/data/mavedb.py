"""MaveDB bulk dump -> canonical tables.

Source: Zenodo 14172004 ``mavedb-dump.20241114101443.zip`` (1.4 GB, v2,
2024-11-14). The zip is ``main.json`` (all score-set metadata) plus one
``csv/<urn>.scores.csv`` per score set. Observed: 2,567 score sets,
7,134,905 variants, every one CC0.

What this loader keeps and what it drops, with counts reported at build time:

* **One target gene.** 2,564 of 2,567 score sets have exactly one; the rest
  have zero. Multi-target score sets do not exist in this dump.
* **A protein sequence.** 631 targets ship a protein sequence directly; 1,933
  ship DNA, which is translated here (standard code, stopping at the first
  stop codon). Targets given only as an accession have no sequence in the dump
  and are skipped.
* **``hgvs_pro`` substitutions.** The canonical ``mutant`` column can only
  express substitutions, so deletions, insertions, duplications, frameshifts
  and extensions are dropped with a count. Rows whose only coordinate is
  ``hgvs_nt`` cannot be placed on a protein and are dropped too. Synonymous
  variants (``p.=``, or ``Xnn=``) become the wild type, ``mutant == ""``.
* **A finite score.**

``assay_type`` is guessed from the experiment's free text. MaveDB's structured
keyword fields are effectively unpopulated in this dump — 13 keyword entries
across 1,196 experiment sets — so there is nothing better to key on. The guess
is reported as a distribution and is null where no pattern matches; do not
treat it as curated.

``reliability`` is left null. DATASETS.md notes that a reproducibility
estimate exists for most score sets but that only about a tenth are findable
by a column-name search; finding the rest is its own task, not this loader's.
"""

from __future__ import annotations

import ast
import json
import zipfile
from collections import Counter

import polars as pl

from .mutations import (
    ConflictingSubstitutions,
    UnsupportedHGVS,
    format_mutant,
    parse_hgvs_pro,
    suggest_offsets,
    verify_wt,
)
from .paths import interim_dir, raw_dir
from .schema import BuildReport, rank_normalize, write_dataset

DATASET = "mavedb"
ZIP_NAME = "mavedb-dump.20241114101443.zip"

_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    f"{a}{b}{c}": _AAS[i]
    for i, (a, b, c) in enumerate(
        (x, y, z) for x in _BASES for y in _BASES for z in _BASES
    )
}

# Free-text patterns -> assay_type. First match wins, so order matters.
ASSAY_PATTERNS = [
    ("abundance", ("abundance", "vamp-seq", "vampseq", "stability", "protein level",
                   "steady-state level", "degron")),
    ("binding",   ("binding", "affinity", "interaction", "two-hybrid", "y2h",
                   "display", "pull-down", "antibody escape", "neutraliz")),
    ("fluorescence", ("fluorescen", "gfp", "brightness")),
    ("growth",    ("growth", "proliferation", "survival", "viability", "competition",
                   "complementation", "drug resistance", "selection coefficient")),
    ("enzymatic", ("enzym", "catalytic", "activity assay", "kinase activity",
                   "protease activity")),
    ("expression", ("expression", "transcription", "reporter")),
    ("splicing",  ("splic",)),
]


def _translate(dna: str) -> str:
    dna = (dna or "").upper().replace("U", "T")
    out = []
    for i in range(0, len(dna) - len(dna) % 3, 3):
        aa = CODON_TABLE.get(dna[i : i + 3], "X")
        if aa == "*":
            break
        out.append(aa)
    return "".join(out)


def _maybe_literal(v):
    if isinstance(v, str):
        if v == "None":
            return None
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _ensure_extracted():
    work = interim_dir(DATASET)
    if (work / "main.json").exists() and (work / "csv").is_dir():
        return work
    zp = raw_dir(DATASET) / ZIP_NAME
    if not zp.exists():
        raise FileNotFoundError(f"{zp} — make -C datasets mavedb")
    print(f"  extracting {zp.name}")
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(work)
    return work


def _guess_assay(*texts: str) -> str | None:
    blob = " ".join(t for t in texts if t).lower()
    for label, pats in ASSAY_PATTERNS:
        if any(p in blob for p in pats):
            return label
    return None


def _score_set_meta(work) -> list[dict]:
    meta = json.load(open(work / "main.json"))
    out = []
    for es in meta["experimentSets"]:
        for ex in es["experiments"]:
            assay = _guess_assay(
                ex.get("title"), ex.get("shortDescription"),
                ex.get("abstractText"), ex.get("methodText"),
            )
            for ss in ex["scoreSets"]:
                tgs = ss.get("targetGenes") or []
                if len(tgs) != 1:
                    out.append(dict(urn=ss["urn"], skip="n_target_genes != 1"))
                    continue
                tg = tgs[0]
                tsq = _maybe_literal(tg.get("targetSequence"))
                if not isinstance(tsq, dict) or not tsq.get("sequence"):
                    out.append(dict(urn=ss["urn"], skip="no target sequence (accession-only)"))
                    continue
                seq_type = (tsq.get("sequenceType") or "").lower()
                raw_seq = tsq["sequence"].strip().upper()
                if seq_type == "dna":
                    prot = _translate(raw_seq)
                elif seq_type == "protein":
                    prot = raw_seq
                else:
                    out.append(dict(urn=ss["urn"], skip=f"unknown sequenceType {seq_type!r}"))
                    continue
                if not prot:
                    out.append(dict(urn=ss["urn"], skip="empty translation"))
                    continue
                lic = (ss.get("license") or {}).get("shortName")
                out.append(
                    dict(
                        urn=ss["urn"],
                        skip=None,
                        target_name=tg.get("name") or ss["urn"],
                        sequence=prot,
                        seq_type=seq_type,
                        license=lic,
                        assay_type=_guess_assay(ss.get("title"), ss.get("shortDescription"),
                                                ss.get("abstractText")) or assay,
                        num_variants=ss.get("numVariants"),
                    )
                )
    return out


_HGVS_CACHE: dict[str, str | None] = {}


def _canon_hgvs(h: str) -> str | None:
    """Canonical mutant string, or None if not representable."""
    got = _HGVS_CACHE.get(h, 0)
    if got != 0:
        return got
    try:
        subs = parse_hgvs_pro(h)
        # No token exists for a stop codon on either side: a read-through
        # (WT is Ter) is not a substitution at all, and a nonsense variant
        # changes the sequence length. Both are dropped, not coerced.
        if any(w == "*" or a == "*" for _p, w, a in subs):
            val = None
        else:
            val = format_mutant(subs)
    except (UnsupportedHGVS, ConflictingSubstitutions, ValueError):
        val = None
    _HGVS_CACHE[h] = val
    return val


def build(limit: int | None = None) -> BuildReport:
    work = _ensure_extracted()
    metas = _score_set_meta(work)
    usable = [m for m in metas if not m["skip"]]
    skipped = Counter(m["skip"] for m in metas if m["skip"])
    print(f"  {len(metas):,} score sets in main.json, {len(usable):,} with a usable target")
    if limit:
        usable = usable[:limit]

    csv_dir = work / "csv"
    target_rows: dict[str, dict] = {}
    frames: list[pl.DataFrame] = []
    ss_rows: list[dict] = []
    stat = Counter()
    n_done = 0

    for m in usable:
        urn = m["urn"]
        path = csv_dir / f"{urn.replace(':', '-')}.scores.csv"
        if not path.exists():
            stat["no_scores_csv"] += 1
            continue
        try:
            d = pl.read_csv(path, infer_schema_length=0)  # all Utf8, then cast
        except Exception as e:  # noqa: BLE001
            stat["unreadable_csv"] += 1
            print(f"  {urn}: unreadable ({type(e).__name__})")
            continue
        if "score" not in d.columns:
            stat["no_score_column"] += 1
            continue
        stat["rows_read"] += d.height

        if "hgvs_pro" not in d.columns:
            stat["no_hgvs_pro_column"] += 1
            stat["rows_dropped_no_hgvs_pro"] += d.height
            continue

        d = d.with_columns(
            pl.col("score").cast(pl.Float64, strict=False),
            pl.col("hgvs_pro").cast(pl.Utf8),
        )
        se_col = next((c for c in ("se", "sd", "SE", "stderr") if c in d.columns), None)
        d = d.with_columns(
            (pl.col(se_col).cast(pl.Float64, strict=False) if se_col else pl.lit(None, pl.Float64)).alias("_se")
        )
        before = d.height
        d = d.filter(pl.col("score").is_not_null() & pl.col("score").is_finite())
        stat["rows_dropped_bad_score"] += before - d.height
        if d.height == 0:
            stat["empty_after_score_filter"] += 1
            continue

        h = d["hgvs_pro"].to_list()
        canon = [None if (x is None or x == "NA") else _canon_hgvs(x) for x in h]
        keep = [c is not None for c in canon]
        n_unrep = len(keep) - sum(keep)
        stat["rows_dropped_unrepresentable_hgvs"] += n_unrep
        if not any(keep):
            stat["empty_after_hgvs_filter"] += 1
            continue

        target_id = f"{urn}|{m['target_name']}"
        d = (
            d.with_columns(pl.Series("mutant", canon, dtype=pl.Utf8))
            .filter(pl.col("mutant").is_not_null())
            .select(
                pl.lit(urn).alias("score_set_id"),
                pl.lit(target_id).alias("target_id"),
                pl.col("mutant"),
                pl.when(pl.col("mutant") == "")
                .then(0)
                .otherwise(pl.col("mutant").str.count_matches(":") + 1)
                .cast(pl.Int32)
                .alias("n_mut"),
                pl.col("score"),
                pl.col("_se").alias("score_se"),
                pl.lit(m["assay_type"], dtype=pl.Utf8).alias("assay_type"),
            )
        )
        frames.append(d)
        target_rows[target_id] = dict(
            target_id=target_id, sequence=m["sequence"], uniprot=None, source="mavedb"
        )
        ss_rows.append(
            dict(
                score_set_id=urn, target_id=target_id, n_variants=d.height,
                max_n_mut=int(d["n_mut"].max()), assay_type=m["assay_type"],
                reliability=None, license=m["license"],
            )
        )
        stat["rows_kept"] += d.height
        n_done += 1
        if n_done % 250 == 0:
            print(f"  {n_done:,}/{len(usable):,} score sets, {stat['rows_kept']:,} rows kept")

    if not frames:
        raise RuntimeError("no MaveDB score sets produced rows")

    variants = rank_normalize(pl.concat(frames, how="vertical_relaxed"))
    targets = pl.DataFrame(list(target_rows.values()))
    score_sets = pl.DataFrame(ss_rows)

    seqs = dict(zip(targets["target_id"], targets["sequence"]))
    chk = verify_wt(variants["mutant"].to_list(), variants["target_id"].to_list(), seqs)

    # Per-target mismatch is the number that matters: a handful of targets with
    # a wrong offset can dominate a global rate.
    bad_targets = len(chk.per_target)
    notes = [
        f"score sets: {len(metas):,} total, {len(usable):,} usable, {n_done:,} produced rows",
        "skipped at metadata stage: "
        + (", ".join(f"{k} ({v})" for k, v in skipped.items()) or "none"),
        "row accounting: "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(stat.items())),
        f"WT check: {chk}",
        f"  targets with >=1 WT mismatch: {bad_targets:,} of {targets.height:,} "
        f"({100 * bad_targets / max(targets.height, 1):.1f}%)",
    ]
    if chk.examples:
        notes.append("  e.g. " + "; ".join(chk.examples[:3]))
    offs = suggest_offsets(
        variants["mutant"].to_list(), variants["target_id"].to_list(), seqs
    )
    if offs:
        shown = sorted(offs.items(), key=lambda kv: -(kv[1][1] - kv[1][2]))[:5]
        notes.append(
            f"  {len(offs)} target(s) badly mismatched (>10%) are explained by a constant "
            f"numbering offset; reported, NOT applied. e.g. "
            + "; ".join(
                f"{t} off {o:+d} ({100 * r0:.0f}%->{100 * rb:.0f}%)"
                for t, (o, r0, rb) in shown
            )
        )
    at = variants.group_by("assay_type").len().sort("len", descending=True)
    notes.append(
        "assay_type (guessed from free text, not curated): "
        + ", ".join(f"{a or 'null'} {n:,}" for a, n in at.iter_rows())
    )

    return write_dataset(DATASET, targets, variants, score_sets, notes)


if __name__ == "__main__":
    print(build())

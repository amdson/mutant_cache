"""Tsuboyama et al. 2023 Megascale -> canonical tables.

**Which file.** DATASETS.md names ``Tsuboyama2023_Dataset1_*.csv`` as the
curated per-variant ΔG table. That is wrong: Dataset1 (1.28 GB) is keyed by
*DNA* sequence and carries no ``aa_seq``, ``mut_type`` or ``WT_name`` column,
so there is no way to tell which variant a row is. The curated table is
``Tsuboyama2023_Dataset2_Dataset3_20230416.csv`` (718 MB), which has exactly
the 776,298 rows over 479 domains that DATASETS.md attributes to Dataset1, plus
``aa_seq`` / ``mut_type`` / ``WT_name`` / ``ddG_ML``. This loader reads that
file.

**Score.** ``ddG_ML``, kcal/mol, equal to ``dG_ML(mut) - dG_ML(wt)``, so
*negative is destabilizing*. 168,459 rows report ``"-"`` instead of a number
(stability outside the assay's dynamic range); they are dropped and counted,
not coerced to zero.

**Uncertainty.** There is no ΔΔG confidence interval in the file, only
``deltaG_95CI``, the full width of the 95% interval on the variant's ΔG.
``score_se`` is ``deltaG_95CI / (2 * 1.96)`` combined in quadrature with the
same quantity for that domain's wild type, since ΔΔG is a difference of two
fitted ΔG values.

**Indels.** 50,643 insertion and 25,275 deletion rows cannot be written as
substitutions and are dropped with a count. The canonical ``mutant`` column has
no representation for them.

**Source typo**, left as found: the ``match_aaseq`` column spells True as
``"Ture"`` on 566,178 rows.

Score sets are one per domain (``tsuboyama_<WT_name>``). The schema gives each
score set a single ``target_id``, so one score set spanning all 479 domains
would not be expressible — and per-domain rank normalization is what the
epsilon construction wants anyway.
"""

from __future__ import annotations

import re
import zipfile

import polars as pl

from .mutations import format_mutant, verify_wt
from .paths import interim_dir, raw_dir
from .schema import BuildReport, rank_normalize, write_dataset

DATASET = "tsuboyama"
CSV_NAME = "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
CI95_TO_SE = 2 * 1.959964  # full 95% CI width -> standard error

_SUB = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _ensure_extracted() -> "Path":  # noqa: F821
    work = interim_dir(DATASET)
    csv = work / "Processed_K50_dG_datasets" / CSV_NAME
    if csv.exists():
        return csv
    zp = raw_dir(DATASET) / "Processed_K50_dG_datasets.zip"
    if not zp.exists():
        raise FileNotFoundError(f"{zp} — make -C datasets tsuboyama")
    print(f"  extracting {CSV_NAME} from {zp.name}")
    with zipfile.ZipFile(zp) as zf:
        member = f"Processed_K50_dG_datasets/{CSV_NAME}"
        zf.extract(member, work)
        for extra in ("Single_DMS_list.csv", "Double_DMS_list.csv", "Triple_DMS_list.csv"):
            try:
                zf.extract(f"Processed_K50_dG_datasets/{extra}", work)
            except KeyError:
                pass
    return csv


def _mut_kind(m: str) -> str:
    if m == "wt":
        return "wt"
    for tok in m.split(":"):
        if tok.startswith("ins"):
            return "ins"
        if tok.startswith("del"):
            return "del"
        if not _SUB.fullmatch(tok):
            return "other"
    return "sub"


def load_raw() -> pl.DataFrame:
    csv = _ensure_extracted()
    return pl.read_csv(
        csv,
        infer_schema_length=100_000,
        columns=[
            "WT_name", "aa_seq", "mut_type", "ddG_ML", "dG_ML",
            "deltaG", "deltaG_95CI", "WT_cluster", "name",
        ],
    ).with_columns(
        pl.col("ddG_ML").cast(pl.Float64, strict=False),
        pl.col("dG_ML").cast(pl.Float64, strict=False),
    )


def build() -> BuildReport:
    d = load_raw()
    n_raw = d.height
    notes = [f"source file {CSV_NAME}: {n_raw:,} rows, {d['WT_name'].n_unique()} domains"]

    d = d.with_columns(
        pl.col("mut_type").map_elements(_mut_kind, return_dtype=pl.Utf8).alias("kind")
    )
    kinds = dict(d.group_by("kind").len().iter_rows())
    notes.append(
        "mutation kinds: "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(kinds.items(), key=lambda x: -x[1]))
        + " (ins/del have no canonical representation and are dropped)"
    )

    # --- reference sequences ------------------------------------------------
    wt_rows = (
        d.filter(pl.col("kind") == "wt")
        .group_by("WT_name")
        .agg(
            pl.col("aa_seq").first().alias("sequence"),
            pl.col("dG_ML").mean().alias("wt_dG_ML"),
            pl.col("deltaG_95CI").mean().alias("wt_95CI"),
            pl.col("WT_cluster").first(),
        )
    )
    seqs = dict(zip(wt_rows["WT_name"], wt_rows["sequence"]))

    # One domain has no explicit wt row; recover its sequence by reverting a
    # single substitution in one of its variants.
    missing = set(d["WT_name"].unique()) - set(seqs)
    recovered = []
    for name in sorted(missing):
        sub = d.filter((pl.col("WT_name") == name) & (pl.col("kind") == "sub"))
        got = False
        for row in sub.head(50).iter_rows(named=True):
            toks = row["mut_type"].split(":")
            s = list(row["aa_seq"])  # the variant sequence
            ok = True
            for t in toks:
                m = _SUB.fullmatch(t)
                # mut_type is <wt><pos><alt>; revert alt back to wt.
                if not m or int(m.group(2)) > len(s) or s[int(m.group(2)) - 1] != m.group(3):
                    ok = False
                    break
                s[int(m.group(2)) - 1] = m.group(1)
            if ok:
                seqs[name] = "".join(s)
                recovered.append(name)
                got = True
                break
        if not got:
            notes.append(f"domain {name}: no wt row and could not reconstruct; dropped")
    if recovered:
        notes.append(
            f"{len(recovered)} domain(s) had no wt row; sequence reconstructed by "
            f"reverting a single mutant: {', '.join(recovered)}"
        )

    targets = pl.DataFrame(
        {
            "target_id": list(seqs),
            "sequence": [seqs[k] for k in seqs],
            "uniprot": [None] * len(seqs),
            "source": ["tsuboyama"] * len(seqs),
        }
    )

    # --- variants -----------------------------------------------------------
    subs = d.filter(pl.col("kind").is_in(["sub", "wt"]))
    n_no_ddg = subs.filter(pl.col("ddG_ML").is_null()).height
    notes.append(
        f'ddG_ML == "-" (outside dynamic range) on {n_no_ddg:,} of {subs.height:,} '
        f"substitution/wt rows; dropped"
    )
    subs = subs.filter(pl.col("ddG_ML").is_not_null())
    subs = subs.filter(pl.col("WT_name").is_in(list(seqs)))

    wt_se = {
        r["WT_name"]: (r["wt_95CI"] / CI95_TO_SE if r["wt_95CI"] is not None else None)
        for r in wt_rows.iter_rows(named=True)
    }

    # mut_type is already <wt><pos><alt>, 1-based on aa_seq -- checked on a
    # 20,000-row sample: the first letter matches the reference sequence 100%
    # of the time and the third matches the variant's own aa_seq 100% of the
    # time. So canonicalization is just a re-sort by position.
    def canon(mut: str) -> str:
        if mut == "wt":
            return ""
        return format_mutant(
            [
                (int(m.group(2)), m.group(1), m.group(3))
                for m in (_SUB.fullmatch(t) for t in mut.split(":"))
            ]
        )
    variants = (
        subs.with_columns(
            pl.col("mut_type").map_elements(canon, return_dtype=pl.Utf8).alias("mutant")
        )
        .with_columns(
            pl.when(pl.col("mutant") == "")
            .then(0)
            .otherwise(pl.col("mutant").str.count_matches(":") + 1)
            .cast(pl.Int32)
            .alias("n_mut"),
            (pl.col("deltaG_95CI") / CI95_TO_SE).alias("_se_var"),
            pl.col("WT_name")
            .replace_strict(wt_se, default=None, return_dtype=pl.Float64)
            .alias("_se_wt"),
        )
        .with_columns(
            (pl.col("_se_var") ** 2 + pl.col("_se_wt").fill_null(0.0) ** 2)
            .sqrt()
            .alias("score_se")
        )
        .select(
            ("tsuboyama_" + pl.col("WT_name")).alias("score_set_id"),
            pl.col("WT_name").alias("target_id"),
            pl.col("mutant"),
            pl.col("n_mut"),
            pl.col("ddG_ML").alias("score"),
            pl.col("score_se"),
            pl.lit("dG").alias("assay_type"),
        )
    )
    variants = rank_normalize(variants)

    chk = verify_wt(
        variants["mutant"].to_list(), variants["target_id"].to_list(), seqs
    )
    notes.append(f"WT check: {chk}")
    if chk.examples:
        notes.append("  e.g. " + "; ".join(chk.examples[:3]))

    score_sets = (
        variants.group_by("score_set_id")
        .agg(
            pl.col("target_id").first(),
            pl.len().alias("n_variants"),
            pl.col("n_mut").max().alias("max_n_mut"),
            pl.col("assay_type").first(),
        )
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("reliability"),
            pl.lit("CC-BY-4.0 (Zenodo 7844779)").alias("license"),
        )
        .sort("score_set_id")
    )

    return write_dataset(DATASET, targets, variants, score_sets, notes)


if __name__ == "__main__":
    print(build())

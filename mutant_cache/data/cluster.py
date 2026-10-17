"""mmseqs2 sequence clustering, shared by the exclusion list and the splits.

``--cov-mode 1`` (coverage of the target) is deliberate: a 60-residue domain
has to be able to join the cluster of the 600-residue protein it is part of.
Under the default coverage mode it never would, and short-domain datasets like
Tsuboyama would look spuriously unrelated to everything.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

MMSEQS = "/home/andrew/MMseqs2/build/bin/mmseqs"


def available() -> bool:
    return Path(MMSEQS).exists()


def write_fasta(path: Path, records) -> None:
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")


def cluster(
    records,
    min_seq_id: float = 0.3,
    coverage: float = 0.5,
    cov_mode: int = 1,
    sensitivity: float = 7.5,
    timeout: int = 3600,
) -> dict[str, str] | None:
    """``[(name, seq), ...]`` -> ``{name: cluster_representative}``.

    Returns ``None`` if mmseqs2 is missing or fails, so callers can fall back.
    """
    records = list(records)
    if not records or not available():
        return None
    with tempfile.TemporaryDirectory(prefix="mc_clust_") as td:
        tmp = Path(td)
        fasta = tmp / "in.fasta"
        write_fasta(fasta, records)
        try:
            subprocess.run(
                [
                    MMSEQS, "easy-cluster", str(fasta), str(tmp / "clu"), str(tmp / "work"),
                    "--min-seq-id", str(min_seq_id),
                    "-c", str(coverage),
                    "--cov-mode", str(cov_mode),
                    "-s", str(sensitivity),
                    "--cluster-mode", "0",
                    "-v", "1",
                ],
                check=True, capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            msg = getattr(e, "stderr", "") or ""
            print(f"  mmseqs2 failed ({type(e).__name__}): {str(msg)[:300]}")
            return None
        out = tmp / "clu_cluster.tsv"
        if not out.exists():
            return None
        mapping: dict[str, str] = {}
        with open(out) as fh:
            for line in fh:
                rep, member = line.rstrip("\n").split("\t")
                mapping[member] = rep
        return mapping

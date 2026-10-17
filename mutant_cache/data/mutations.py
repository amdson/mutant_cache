"""Mutation-string parsing and WT verification.

Canonical form (INTERFACES.md section 2): colon-joined, 1-based on the target
sequence, ``A23G:K45R``; the empty string for wild type.

The point of ``verify_wt`` is that it reports mismatch rates instead of
silently dropping or "fixing" rows: an offset between a dataset's numbering
and its own reference sequence is a real finding about the dataset, not noise
to be cleaned away.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_ALLOWED = set(AA20) | {"X"}

_SUB_RE = re.compile(r"^([A-Z])(\d+)([A-Z*])$")


def parse_mutant(mutant: str) -> list[tuple[int, str, str]]:
    """``"A23G:K45R"`` -> ``[(23, "A", "G"), (45, "K", "R")]``, 1-based.

    Returns ``[]`` for the wild type (empty string).
    """
    if mutant is None:
        raise ValueError("mutant is None")
    mutant = mutant.strip()
    if mutant == "" or mutant.upper() in {"WT", "WILDTYPE"}:
        return []
    out = []
    for tok in mutant.split(":"):
        tok = tok.strip()
        m = _SUB_RE.match(tok)
        if not m:
            raise ValueError(f"cannot parse substitution {tok!r} in {mutant!r}")
        wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
        if pos < 1:
            raise ValueError(f"non-positive position in {mutant!r}")
        out.append((pos, wt, alt))
    return out


class ConflictingSubstitutions(ValueError):
    """Two different substitutions at the same position in one variant."""


def format_mutant(subs) -> str:
    """Inverse of :func:`parse_mutant`; sorts by position for a canonical key.

    Exact duplicates are collapsed. MaveDB really does emit things like
    ``p.[Pro5Lys;Pro5Lys]`` -- 25,611 variants in the bulk dump repeat a
    substitution -- and leaving them in would make ``n_mut`` disagree with the
    number of distinct substitutions the variant actually has.

    Two *different* alternatives at one position raise
    :class:`ConflictingSubstitutions`: that variant cannot be expressed as a
    set of independent substitutions and must be dropped, not guessed at.
    """
    by_pos: dict[int, tuple[str, str]] = {}
    for pos, wt, alt in subs:
        prev = by_pos.get(pos)
        if prev is None:
            by_pos[pos] = (wt, alt)
        elif prev != (wt, alt):
            raise ConflictingSubstitutions(
                f"position {pos}: {prev[0]}{pos}{prev[1]} vs {wt}{pos}{alt}"
            )
    return ":".join(f"{wt}{pos}{alt}" for pos, (wt, alt) in sorted(by_pos.items()))


def n_mut(mutant: str) -> int:
    return len(parse_mutant(mutant))


def mutate(sequence: str, mutant: str, check: bool = True) -> str:
    """Apply a canonical mutant string to a sequence (1-based)."""
    s = list(sequence)
    for pos, wt, alt in parse_mutant(mutant):
        if pos > len(s):
            raise IndexError(f"position {pos} beyond sequence of length {len(s)}")
        if check and s[pos - 1] != wt:
            raise ValueError(
                f"WT mismatch at {pos}: sequence has {s[pos - 1]}, mutant says {wt}"
            )
        s[pos - 1] = alt
    return "".join(s)


def diff_sequences(wt_seq: str, var_seq: str) -> str:
    """Canonical mutant string from an aligned pair of equal-length sequences."""
    if len(wt_seq) != len(var_seq):
        raise ValueError(f"length mismatch {len(wt_seq)} vs {len(var_seq)}")
    return format_mutant(
        [(i + 1, a, b) for i, (a, b) in enumerate(zip(wt_seq, var_seq)) if a != b]
    )


@dataclass
class WTCheck:
    """Outcome of checking mutant strings against their target sequence."""

    n_variants: int = 0
    n_subs: int = 0
    n_sub_mismatch: int = 0
    n_variant_mismatch: int = 0
    n_out_of_range: int = 0
    n_unparseable: int = 0
    examples: list[str] = field(default_factory=list)
    per_target: Counter = field(default_factory=Counter)

    @property
    def sub_mismatch_rate(self) -> float:
        return self.n_sub_mismatch / self.n_subs if self.n_subs else 0.0

    @property
    def variant_mismatch_rate(self) -> float:
        return self.n_variant_mismatch / self.n_variants if self.n_variants else 0.0

    def __str__(self) -> str:
        return (
            f"WTCheck(variants={self.n_variants:,}, subs={self.n_subs:,}, "
            f"sub_mismatch={self.n_sub_mismatch:,} ({100 * self.sub_mismatch_rate:.4f}%), "
            f"variant_mismatch={self.n_variant_mismatch:,} "
            f"({100 * self.variant_mismatch_rate:.4f}%), "
            f"out_of_range={self.n_out_of_range:,}, unparseable={self.n_unparseable:,})"
        )


def suggest_offsets(
    mutants,
    target_ids,
    sequences: dict[str, str],
    search: range = range(-5, 6),
    min_rate: float = 0.10,
) -> dict[str, tuple[int, float, float]]:
    """For badly-mismatching targets, find the constant offset that fixes them.

    A dataset whose numbering is shifted relative to its own deposited
    sequence produces a high mismatch rate that one integer explains. This
    *reports* that integer; it does not apply it, because an offset that fixes
    99% of a target's rows is evidence about the deposit, and silently
    shifting coordinates is how a whole assay ends up mislabelled.

    Returns ``{target_id: (best_offset, rate_at_0, rate_at_best)}`` for targets
    whose mismatch rate at offset 0 exceeds ``min_rate``.
    """
    by_target: dict[str, list[tuple[int, str]]] = {}
    for mut, tid in zip(mutants, target_ids):
        try:
            subs = parse_mutant(mut)
        except ValueError:
            continue
        by_target.setdefault(tid, []).extend((p, w) for p, w, _ in subs)

    out = {}
    for tid, subs in by_target.items():
        seq = sequences.get(tid)
        if not seq or not subs:
            continue

        def rate(off: int) -> float:
            good = sum(
                1
                for p, w in subs
                if 1 <= p + off <= len(seq) and seq[p + off - 1] == w
            )
            return 1.0 - good / len(subs)

        r0 = rate(0)
        if r0 <= min_rate:
            continue
        best_off, best_r = 0, r0
        for off in search:
            if off == 0:
                continue
            r = rate(off)
            if r < best_r:
                best_off, best_r = off, r
        if best_off != 0:
            out[tid] = (best_off, r0, best_r)
    return out


def verify_wt(
    mutants,
    target_ids,
    sequences: dict[str, str],
    max_examples: int = 10,
) -> WTCheck:
    """Check every substitution's WT letter against the target sequence.

    Reports; never mutates. ``sequences`` maps target_id -> reference sequence.
    """
    chk = WTCheck()
    for mut, tid in zip(mutants, target_ids):
        chk.n_variants += 1
        seq = sequences.get(tid)
        if seq is None:
            chk.n_variant_mismatch += 1
            chk.per_target[tid] += 1
            if len(chk.examples) < max_examples:
                chk.examples.append(f"{tid}: no reference sequence")
            continue
        try:
            subs = parse_mutant(mut)
        except ValueError:
            chk.n_unparseable += 1
            chk.n_variant_mismatch += 1
            chk.per_target[tid] += 1
            if len(chk.examples) < max_examples:
                chk.examples.append(f"{tid}: unparseable {mut!r}")
            continue
        bad = False
        for pos, wt, _alt in subs:
            chk.n_subs += 1
            if pos > len(seq):
                chk.n_out_of_range += 1
                chk.n_sub_mismatch += 1
                bad = True
                if len(chk.examples) < max_examples:
                    chk.examples.append(
                        f"{tid}: {mut} position {pos} > len {len(seq)}"
                    )
            elif seq[pos - 1] != wt:
                chk.n_sub_mismatch += 1
                bad = True
                if len(chk.examples) < max_examples:
                    chk.examples.append(
                        f"{tid}: {mut} says {wt}{pos} but sequence has {seq[pos - 1]}"
                    )
        if bad:
            chk.n_variant_mismatch += 1
            chk.per_target[tid] += 1
    return chk


# --- MaveDB HGVS protein strings -------------------------------------------

_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*", "Sec": "U", "Xaa": "X",
}

_HGVS_P_SUB = re.compile(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|\*)$")


class UnsupportedHGVS(ValueError):
    """Raised for HGVS the canonical schema has no room for (indels, fs, ...)."""


def parse_hgvs_pro(hgvs: str) -> list[tuple[int, str, str]]:
    """Parse a MaveDB ``p.`` HGVS string into 1-based substitutions.

    Only pure substitutions are representable in the canonical schema; anything
    else (deletion, insertion, frameshift, duplication) raises
    :class:`UnsupportedHGVS` so the caller can count and drop it explicitly.
    """
    if hgvs is None:
        raise UnsupportedHGVS("null")
    s = hgvs.strip()
    if s in {"", "_wt", "_sy", "p.=", "p.(=)"}:
        return []
    if s.startswith("p."):
        s = s[2:]
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    if s in {"=", ""}:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    for bad in ("del", "ins", "dup", "fs", "ext", "?"):
        if bad in s:
            raise UnsupportedHGVS(hgvs)
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if tok in {"=", ""}:
            continue
        m = _HGVS_P_SUB.match(tok)
        if not m:
            raise UnsupportedHGVS(hgvs)
        wt3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)
        wt = _THREE_TO_ONE.get(wt3)
        if wt is None:
            raise UnsupportedHGVS(hgvs)
        if alt3 == "=":
            continue  # silent at this position
        alt = "*" if alt3 == "*" else _THREE_TO_ONE.get(alt3)
        if alt is None:
            raise UnsupportedHGVS(hgvs)
        out.append((pos, wt, alt))
    return out

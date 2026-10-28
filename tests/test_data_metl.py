"""METL loader: the two index conventions are derived from the wild type (not
assumed), canonicalisation round-trips, and a wrong offset is refused.

The build functions themselves need the 19 GB raw download, so they are only
exercised when it is present (`-m slow` style guard via skipif)."""
from __future__ import annotations

import pytest

from mutant_cache.data import metl

WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"   # METL GB1


def test_offset_detects_one_indexed():
    # T17P: 1-based residue 17 is WT[16] == 'T'
    assert WT[16] == "T"
    assert metl._offset(WT, ["T17P,T44L", "T17G,Y45G"], "x") == 1


def test_offset_detects_zero_indexed():
    # Q1A: 0-based index 1 is WT[1] == 'Q'
    assert WT[1] == "Q"
    assert metl._offset(WT, ["Q1A", "Q1C", "Y2D"], "x") == 0


def test_offset_refuses_when_ambiguous_or_wrong():
    with pytest.raises(ValueError, match="no unique index convention"):
        metl._offset(WT, ["Z9Z"], "x")          # matches neither
    with pytest.raises(ValueError, match="no unique index convention"):
        # 'M' at written 0 (offset 0) and 'Q' nowhere -> mixed, so neither is 1.0
        metl._offset(WT, ["M0A", "T44L"], "x")


def test_canonical_is_one_based_and_checks_wt():
    # 1-indexed input: residue 17 stays 17 (token 17 under BOS=0)
    assert metl._canonical("T17P,T44L", WT, 1) == "T17P:T44L"
    # 0-indexed input: index 1 becomes residue 2
    assert metl._canonical("Q1A", WT, 0) == "Q2A"
    with pytest.raises(ValueError, match="expected"):
        metl._canonical("A17P", WT, 1)          # WT has T at 17, not A


def test_canonical_agrees_with_the_repo_parser():
    from mutant_cache.data.mutations import mutate, parse_mutant
    m = metl._canonical("T17P,T44L", WT, 1)
    assert [p for p, _, _ in parse_mutant(m)] == [17, 44]
    out = mutate(WT, m)                          # validates the WT residues too
    assert out[16] == "P" and out[43] == "L" and len(out) == len(WT)


@pytest.mark.skipif(not (metl.ROSETTA_DIR / "gb1" / "gb1.tsv").exists(),
                    reason="METL raw download absent")
def test_real_files_use_the_conventions_the_module_documents():
    import itertools
    p = metl.protein("gb1")
    assert p.wt == WT
    ros = [l.split("\t")[1] for l in
           itertools.islice(open(p.rosetta / "gb1.tsv"), 1, 5001)]
    dms = [l.split("\t")[0] for l in
           itertools.islice(open(p.dms / "gb1.tsv"), 1, 5001)]
    assert metl._offset(WT, ros, "rosetta") == 1
    assert metl._offset(WT, dms, "dms") == 0

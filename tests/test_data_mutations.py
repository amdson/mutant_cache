"""Unit tests for mutation-string handling. No built files needed."""

import pytest

from mutant_cache.data.mutations import (
    UnsupportedHGVS,
    diff_sequences,
    format_mutant,
    mutate,
    n_mut,
    parse_hgvs_pro,
    parse_mutant,
    suggest_offsets,
    verify_wt,
)


def test_parse_roundtrip():
    assert parse_mutant("A23G:K45R") == [(23, "A", "G"), (45, "K", "R")]
    assert parse_mutant("") == []
    assert parse_mutant("WT") == []
    assert format_mutant(parse_mutant("K45R:A23G")) == "A23G:K45R"
    assert n_mut("A23G:K45R") == 2
    assert n_mut("") == 0


@pytest.mark.parametrize("bad", ["A23", "23G", "AG", "A-1G", "p.Ala23Gly", "A23g"])
def test_parse_rejects_junk(bad):
    with pytest.raises(ValueError):
        parse_mutant(bad)


def test_format_sorts_by_position():
    assert format_mutant([(45, "K", "R"), (23, "A", "G")]) == "A23G:K45R"


def test_mutate_and_diff():
    seq = "MAKTG"
    assert mutate(seq, "A2G") == "MGKTG"
    assert mutate(seq, "A2G:T4S") == "MGKSG"
    assert diff_sequences(seq, "MGKSG") == "A2G:T4S"
    assert diff_sequences(seq, seq) == ""
    with pytest.raises(ValueError):
        mutate(seq, "G2A")  # WT letter is A, not G
    with pytest.raises(IndexError):
        mutate(seq, "M9A")


def test_verify_wt_counts_rather_than_raises():
    seqs = {"t": "MAKTG"}
    chk = verify_wt(["A2G", "G2A", "M9A"], ["t", "t", "t"], seqs)
    assert chk.n_variants == 3
    assert chk.n_subs == 3
    assert chk.n_sub_mismatch == 2       # wrong letter + out of range
    assert chk.n_variant_mismatch == 2
    assert chk.n_out_of_range == 1
    assert 0.6 < chk.sub_mismatch_rate < 0.7


def test_suggest_offsets_finds_a_shift():
    seq = "MAKTGWY"
    # every mutant numbered one too high
    muts = ["A3G", "K4R", "T5S", "G6W"]
    out = suggest_offsets(muts, ["t"] * 4, {"t": seq})
    assert out["t"][0] == -1
    assert out["t"][2] == 0.0  # perfect after the shift


def test_suggest_offsets_silent_when_clean():
    assert suggest_offsets(["A2G"], ["t"], {"t": "MAKTG"}) == {}


def test_hgvs_substitutions():
    assert parse_hgvs_pro("p.Pro73Gln") == [(73, "P", "Q")]
    assert parse_hgvs_pro("p.[Ala1Gly;Lys4Arg]") == [(1, "A", "G"), (4, "K", "R")]
    assert parse_hgvs_pro("p.=") == []
    assert parse_hgvs_pro("_wt") == []
    assert parse_hgvs_pro("p.Trp10Ter") == [(10, "W", "*")]


@pytest.mark.parametrize(
    "bad",
    ["p.Ala3del", "p.Ala3_Gly4insLys", "p.Ala3fs", "p.Ala3dup", "p.Met1ext-5", "p.?"],
)
def test_hgvs_rejects_non_substitutions(bad):
    with pytest.raises(UnsupportedHGVS):
        parse_hgvs_pro(bad)

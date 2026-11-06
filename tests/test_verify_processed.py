import json

import pandas as pd

from mutant_cache.data.mutate_everything_raw import holdout_split
from mutant_cache.data.verify_processed import describe


def _write(df, path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def test_table_hash_ignores_layout_but_not_content(tmp_path):
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1.5, 2.0, None], "n": [1, 2, 3]})
    base = describe(_p(tmp_path, "base.parquet", df), tmp_path, raw=False)
    shuffled = df.iloc[[2, 0, 1]][["n", "b", "a"]]
    same = describe(_p(tmp_path, "shuf.parquet", shuffled), tmp_path, raw=False)
    assert same["content_sha256"] == base["content_sha256"]
    assert same["ordered_sha256"] != base["ordered_sha256"]
    changed = df.assign(b=[1.5, 2.0000000001, None])
    diff = describe(_p(tmp_path, "chg.parquet", changed), tmp_path, raw=False)
    assert diff["content_sha256"] != base["content_sha256"]
    assert diff["rounded_sha256"] == base["rounded_sha256"]
    as_int = df.assign(b=[1, 2, 3])
    assert describe(_p(tmp_path, "int.parquet", as_int), tmp_path, raw=False)["content_sha256"] \
        != base["content_sha256"]


def test_scratch_paths_normalized(tmp_path):
    a, b = tmp_path / "s1", tmp_path / "s2"
    for root in (a, b):
        root.mkdir()
        (root / "m.json").write_text(json.dumps({"path": str(root / "x.csv"), "k": [1, 2.5]}))
        _write(pd.DataFrame({"f": [str(root / "y")]}), root / "t.csv")
    for name in ("m.json", "t.csv"):
        assert describe(a / name, a, raw=False)["content_sha256"] == \
            describe(b / name, b, raw=False)["content_sha256"]


def test_holdout_split_keeps_test_and_identical_sequences_together():
    split = pd.DataFrame({"target_id": list("abcdefghijk") + ["t1", "t2"],
                          "split": ["train"] * 10 + ["val"] + ["test"] * 2})
    seqs = {t: f"S{i}" for i, t in enumerate(split.target_id)} | {"b": "S0"}
    out1 = holdout_split(split, seqs, 1)
    assert out1.equals(holdout_split(split, seqs, 1))
    assert out1.loc[out1.split == "test", "target_id"].tolist() == ["t1", "t2"]
    assert out1.set_index("target_id").split["a"] == out1.set_index("target_id").split["b"]
    val_seqs = {seqs[t] for t in out1.loc[out1.split == "val", "target_id"]}
    assert len(val_seqs) == round(10 * 0.1)  # 10 unique dev sequences


def _p(root, name, df):
    path = root / name
    _write(df, path)
    return path

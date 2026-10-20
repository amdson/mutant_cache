"""End-to-end smoke test for ``mutant_cache.harness``: the shared loop every real
experiment goes through, run on the synthetic §2 tables with a toy-size real
ESM-C (random weights, real cache code path, real tokenizer).

It checks plumbing, not accuracy: the run finishes, writes its record, and
every prediction lands on the row it belongs to. Each of these caught a real
bug once — batches on the GPU handing CUDA indices to pandas, bf16 backbone
states reaching the fp32 head, and splits looked up in the wrong directory.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from fixtures import make_synthetic_dataset
from mutant_cache import harness

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d, _ = make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_targets=2,
                                  seq_len=20, n_singles=40, n_doubles=40, n_triples=10)
    return d


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_harness_runs_end_to_end(data, tmp_path, precision):
    out = tmp_path / "run"
    rc = harness.main([
        "--data-dir", str(data), "--dataset", "synth", "--split", "protein",
        "--backbone", "random:64:4:2", "--precision", precision, "--device", DEV,
        "--epochs", "2", "--batch-size", "16", "--baselines", "zero_shot",
        "--control-backend", "recompute",
        "--output", str(out), "--out-root", str(tmp_path)])
    assert rc == 0

    rec = json.loads((out / "run.json").read_text())
    assert rec["hyperparams"]["precision"] == precision
    assert (tmp_path / "index.jsonl").exists()          # run record stays out of scratch

    hist = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert hist[0]["tag"] == "init" and len(hist) >= 3
    assert all(np.isfinite(h["train_loss"]) for h in hist if "train_loss" in h)

    # every prediction is attached to its own row: the score carried next to it
    # must be that (score set, mutant)'s score in the source table
    pred = pd.read_csv(out / "predictions.csv", keep_default_na=False)
    assert len(pred) and pred["pred"].astype(float).notna().all()
    src = pd.read_parquet(data / "variants.parquet")
    m = pred.merge(src[["score_set_id", "mutant", "score"]],
                   on=["score_set_id", "mutant"], suffixes=("", "_src"))
    assert len(m) == len(pred)
    assert np.allclose(m["score"].astype(float), m["score_src"])


@pytest.mark.parametrize("lora", [["--lora-r", "4"], ["--lora-r", "4", "--shared-lora-r", "4"]])
def test_harness_trains_lora_adapters(data, tmp_path, lora):
    """Adapters live inside the backbone, so a frozen-base loop that only
    optimizes head_parameters() would silently never touch them. lora_B starts
    at zero: nonzero after training means the optimizer updated it."""
    args = harness.build_parser().parse_args([
        "--data-dir", str(data), "--dataset", "synth", "--split", "protein",
        "--backbone", "random:64:4:2", "--device", DEV, "--epochs", "1",
        "--batch-size", "16", "--baselines", "none", *lora,
        "--output", str(tmp_path / "run"), "--out-root", str(tmp_path)])
    res = harness.run(args)
    assert res["status"] == "ok"
    bb = res["model"].backbone
    sets = {"variant": bb.variant_adapters, "shared": bb.shared_adapters}
    for name, ad in sets.items():
        if name == "shared" and "--shared-lora-r" not in lora:
            assert ad is None
            continue
        Bs = [p for n, p in ad.named_parameters() if n.endswith("lora_B")]
        assert Bs and all(p.dtype == torch.float32 for p in Bs)
        assert sum(p.abs().sum().item() for p in Bs) > 0, f"{name} adapters never updated"
    assert all(not p.requires_grad for _, p in bb.base_parameters())


def test_load_split_falls_back_to_shared_splits_dir(tmp_path, monkeypatch):
    """The data layer writes splits to $MUTANT_CACHE_SCRATCH/processed/splits/,
    not under each dataset; the harness must find them there."""
    from mutant_cache.models.data_compat import load_split
    monkeypatch.setenv("MUTANT_CACHE_SCRATCH", str(tmp_path))
    shared = tmp_path / "processed" / "splits"
    shared.mkdir(parents=True)
    pd.DataFrame({"target_id": ["A"], "split": ["train"]}).to_parquet(shared / "s.parquet")
    ds = tmp_path / "processed" / "ds"
    ds.mkdir()
    assert load_split(ds, "s").split.tolist() == ["train"]
    (ds / "splits").mkdir()                              # a dataset-local copy wins
    pd.DataFrame({"target_id": ["A"], "split": ["test"]}).to_parquet(ds / "splits" / "s.parquet")
    assert load_split(ds, "s").split.tolist() == ["test"]

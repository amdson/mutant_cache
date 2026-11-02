"""Harness protocol hygiene (TODO.md §4): a missing validation partition is an
error, never a silent fall-back to test; the train-only rank transform uses
training labels alone."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fixtures import make_synthetic_dataset
from mutant_cache import harness


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d, _ = make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_targets=2,
                                  seq_len=20, n_singles=30, n_doubles=30, n_triples=5)
    return d


def _args(data, *extra):
    return harness.build_parser().parse_args(
        ["--data-dir", str(data), "--dataset", "synth", *extra])


def test_missing_val_is_an_error(data):
    # the 2-target synthetic "protein" split has train + test, no val
    with pytest.raises(SystemExit, match="no 'val' rows"):
        harness.prepare_frames(_args(data, "--split", "protein"))
    with pytest.raises(SystemExit, match="no val rows"):
        harness.prepare_frames(_args(data, "--split", "protein", "--eval-split", "test",
                                     "--select-on-val"))
    f = harness.prepare_frames(_args(data, "--split", "protein", "--eval-split", "test"))
    assert len(f["eval"]) and len(f["train"])


def test_train_rank_transform_uses_train_labels_only():
    tr = pd.DataFrame({"score_set_id": "S", "score": [3.0, 1.0, 2.0, 4.0]})
    ev = pd.DataFrame({"score_set_id": "S", "score": [0.0, 2.5, 9.0, 2.0]})
    tr2, ev2 = harness.train_rank_transform(tr, ev)
    assert np.allclose(tr2["target_train_rank"], [0.625, 0.125, 0.375, 0.875])
    # eval rows land on the training ECDF, clipped into the training range
    assert np.allclose(ev2["target_train_rank"], [0.125, 0.5, 0.875, 0.375])
    # changing eval labels never changes training targets
    _, ev3 = harness.train_rank_transform(tr, ev.assign(score=ev["score"] * 100))
    tr3, _ = harness.train_rank_transform(tr, ev.assign(score=-ev["score"]))
    assert np.allclose(tr3["target_train_rank"], tr2["target_train_rank"])


def test_target_transform_flag(data):
    a = _args(data, "--split", "depth", "--target-transform", "train_rank",
              "--select-on-val", "--eval-split", "test")
    f = harness.prepare_frames(a)
    assert a.target_col == "target_train_rank"
    for k in ("train", "eval", "select"):
        assert f[k]["target_train_rank"].between(0, 1).all()

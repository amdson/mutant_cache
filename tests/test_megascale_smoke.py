"""End-to-end plumbing for ``mutant_cache.megascale`` (TODO_ESM_MSR §3 smoke):
synthetic protein-level split, toy-size real ESM-C (random weights), variant
LoRA. Checks the run finishes for both arms, adapters and head move while the
frozen base does not, selection follows the fixed rule, S-add predictions are
saved for the joint arm only, and test is untouched without --final-test."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from fixtures import make_synthetic_dataset
from mutant_cache import megascale

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d, _ = make_synthetic_dataset(tmp_path_factory.mktemp("mega"), n_targets=5, seq_len=24,
                                  n_singles=40, n_doubles=40, n_triples=0)
    return d


def _args(data, tmp_path, *extra):
    return megascale.build_parser().parse_args([
        "--data-dir", str(data), "--dataset", "synth", "--split", "protein",
        "--backbone", "random:64:4:2", "--precision", "fp32", "--device", DEV,
        "--lora-r", "4", "--lora-alpha", "8", "--exposures", "2", "--micro-batch", "16",
        "--proteins-per-step", "2", "--eval-min-n", "5",
        "--output", str(tmp_path / "run"), "--out-root", str(tmp_path), *extra])


@pytest.mark.parametrize("comp", ["joint", "additive"])
def test_megascale_runs(data, tmp_path, comp):
    args = _args(data, tmp_path, "--composition", comp)
    out = megascale.run(args)
    assert out["status"] == "ok"
    model, run_dir = out["model"], tmp_path / "run"
    assert out["extra"]["arm"] == {"joint": "S", "additive": "A"}[comp]
    pv = pd.read_csv(run_dir / "predictions_val.csv")
    assert pv["pred"].notna().all()
    assert ("pred_add" in pv.columns) == (comp == "joint")
    wt = pv[pv["n_mut"] == 0]
    assert (wt["pred"] == 0).all()
    assert not (run_dir / "predictions_test.csv").exists()
    hist = pd.read_csv(run_dir / "history.csv")
    sel = out["extra"]["selection"]
    trained = hist[hist["step"] > 0]
    keys = list(zip(trained["val_double_rmse"].fillna(float("inf")),
                    trained["val_single_rmse"].fillna(float("inf"))))
    assert sel["step"] == trained["step"].iloc[keys.index(min(keys))] > 0
    lora = [p for n, p in model.backbone.named_parameters() if "lora_B" in n]
    assert lora and any(float(p.detach().abs().sum()) > 0 for p in lora)
    assert all(not p.requires_grad for n, p in model.backbone.named_parameters()
               if "lora_" not in n)
    rec = json.loads((run_dir / "results.json").read_text())
    assert set(rec["extra"]["exposure_by_n_mut"]) <= {"0", "1", "2"}
    assert rec["extra"]["timing"]["total_steps"] == hist["step"].max() == 16


def test_megascale_final_test(data, tmp_path):
    out = megascale.run(_args(data, tmp_path, "--final-test"))
    pt = pd.read_csv(tmp_path / "run" / "predictions_test.csv")
    assert pt["pred"].notna().all() and set(pt["split"]) == {"test"}
    assert out["extra"]["test_summary"]


def test_sampler_is_protein_uniform():
    f = pd.DataFrame({"target_id": ["a"] * 1000 + ["b"] * 10, "x": range(1010)})
    s = megascale.ProteinUniformSampler(f, micro_batch=8, seed=0)
    draws = [s.draw()["target_id"].iloc[0] for _ in range(2000)]
    assert 0.45 < draws.count("b") / len(draws) < 0.55
    mb = s.draw()
    assert mb.index.is_unique


def test_full_training_refit_uses_all_development_proteins_and_final_weights(data, tmp_path):
    a = _args(data, tmp_path, "--full-train-refit", "--final-eval-only", "--save-model")
    before = megascale.prepare(_args(data, tmp_path))
    after = megascale.prepare(a)
    assert len(after['train']) == len(before['train']) + len(before['val'])
    assert set(after['train'].target_id) == set(before['train'].target_id) | set(before['val'].target_id)
    assert after['test'].empty
    out = megascale.run(a)
    assert len(out['history']) == 1
    assert out['extra']['selection']['step'] == out['extra']['timing']['total_steps']
    assert out['extra']['selection']['rule'] == 'fixed_final'
    assert out['extra']['validation_role'] == 'in_sample_former_validation'
    assert not (tmp_path/'run/predictions_test.csv').exists()
    ck = torch.load(tmp_path/'run/model.pt', weights_only=False)
    assert all(torch.equal(v, ck['trainable_state'][k]) for k, v in megascale.trainable_state(out['model']).items())


def test_full_training_refit_rejects_validation_selection(data, tmp_path):
    with pytest.raises(ValueError, match='final-eval-only'):
        megascale.prepare(_args(data, tmp_path, '--full-train-refit'))


@pytest.mark.parametrize('extra', [
    ['--stability-rank-weight', '0.3'],
    ['--interaction-residual', '--shared-lora-r', '4'],
    ['--train-backbone', '--backbone-train-last', '1'],
    ['--shared-lora-r', '4', '--shared-lora-targets', 'q', 'k', 'v', 'o', 'gate', 'up', 'down'],
])
def test_next_round_training_paths(data, tmp_path, extra):
    args = _args(data, tmp_path, '--minimal', '--head', 'wt_relative', '--zero-anchor', *extra)
    out = megascale.run(args)
    assert out['status'] == 'ok'
    assert out['extra']['test_summary'] is None
    if args.backbone_train_last:
        bb = out['model'].backbone
        assert not any(p.requires_grad for p in bb.blocks[0].parameters())
        assert all(p.requires_grad for p in bb.blocks[-1].parameters())
    if args.interaction_residual:
        assert out['model'].residual_mlp[-1].weight.abs().sum() > 0


def test_saved_checkpoint_rescored_exactly(data, tmp_path):
    a = _args(data, tmp_path, "--save-model")
    megascale.run(a)
    first = pd.read_csv(tmp_path / "run" / "predictions_val.csv")
    b = _args(data, tmp_path / "again", "--exposures", "0", "--load-trainable",
              str(tmp_path / "run" / "model.pt"), "--final-test")
    out = megascale.run(b)
    again = pd.read_csv(tmp_path / "again" / "run" / "predictions_val.csv")
    assert (first["pred"] - again["pred"]).abs().max() < 1e-5
    assert (tmp_path / "again" / "run" / "predictions_test.csv").exists()
    assert out["extra"]["timing"]["total_steps"] == 0

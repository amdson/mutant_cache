import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mutant_cache.diagnostics import (ActivationProfile, ParameterSample, TrainingDiagnostics,
    frame_diagnostics, training_probe)
from mutant_cache.megascale import build_parser, build_loss, run
from mutant_cache.eval import megascale as M


def test_probe_is_bounded_stable_and_contains_measured_triples():
    records = []
    for target in ("a", "b", "c"):
        for i in range(1, 10):
            records += [dict(target_id=target, score_set_id=target, mutant=f"A{i}C", n_mut=1, score=float(i)),
                dict(target_id=target, score_set_id=target, mutant=f"A{i}C:A{i+10}D", n_mut=2, score=float(-i)),
                dict(target_id=target, score_set_id=target, mutant=f"A{i+10}D", n_mut=1, score=float(-i))]
    frame = pd.DataFrame(records)
    np.random.seed(123)
    state = np.random.get_state()
    a = training_probe(frame, proteins=2, rows=12, seed=3)
    pd.testing.assert_frame_equal(a, training_probe(frame.sample(frac=1, random_state=2), proteins=2, rows=12, seed=3))
    assert all(np.array_equal(x, y) for x, y in zip(state, np.random.get_state()))
    assert len(a) <= 24 and a.target_id.nunique() == 2
    assert (a.groupby("target_id").size() <= 12).all()
    for _, group in a.groupby("target_id"):
        mutants = set(group.mutant)
        assert any(all(m in mutants for m in r.mutant.split(":")) for r in group[group.n_mut == 2].itertuples())
    assert set(zip(a.target_id, a.mutant)) <= set(zip(frame.target_id, frame.mutant))


def test_optimizer_diagnostics_measure_preclip_norm_and_actual_step(tmp_path):
    args = build_parser().parse_args(["--diagnostics", "--diagnostic-log-every", "1"])
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3., 4.]]))
    optimizer = torch.optim.SGD([dict(params=list(model.parameters()), name="head")], lr=.5)
    d = TrainingDiagnostics(args, tmp_path, SimpleNamespace(log=lambda *a: None, summary=lambda *a: None))
    d.model = model
    d.samples = {"head": ParameterSample(model.parameters(), 8192)}
    d.begin_step(1, 1)
    (model.weight*torch.tensor([[3., 4.]])).sum().backward()
    d.before_clip(optimizer.param_groups)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    d.observe_clip(norm)
    optimizer.step()
    d.after_step(optimizer.param_groups, 1)
    row = json.loads((tmp_path/"diagnostics/history.jsonl").read_text())
    assert row["optimization/head/grad_norm_pre_clip"] == pytest.approx(5.)
    assert row["optimization/head/update_rms_sampled"] == pytest.approx(np.sqrt(.125), rel=1e-5)
    assert row["optimization/head/relative_update_sampled"] == pytest.approx(.1, rel=1e-5)
    assert row["optimization/head/distance_from_initial_rms_sampled"] == pytest.approx(np.sqrt(.125), rel=1e-5)
    assert row["optimization/clipped_fraction"] == 1.
    assert row["optimization/head/nonfinite_gradient_elements"] == 0


def test_activation_rms_excludes_dense_padding_and_ragged_dummy_tokens():
    profile = ActivationProfile(None)
    profile.batch = SimpleNamespace(var_mask=torch.tensor([[True, False], [True, True]]), n=2,
        positions=torch.tensor([0, 1]))
    profile.tokens = 3
    profile.ragged = False
    dense = torch.tensor([[[1., 1.], [999., 999.]], [[2., 2.], [3., 3.]]])
    profile.record("hidden", profile.real_rows(dense))
    assert profile.values()["hidden/rms"] == pytest.approx(np.sqrt(14/3))
    profile.ragged = True
    packed = torch.tensor([[[1., 1.], [2., 2.], [3., 3.], [999., 999.]]])
    torch.testing.assert_close(profile.real_rows(packed), torch.tensor([[1., 1.], [2., 2.], [3., 3.]]))


def test_calibration_interaction_and_clean_objective_have_numeric_oracles():
    args = build_parser().parse_args([])
    frame = pd.DataFrame(dict(target_id=["p"]*3, mutant=["A1C", "A2D", "A1C:A2D"],
        n_mut=[1, 1, 2], score=[-1., 1., 3.], pred=[-.5, .5, 1.5], pred_add=[-.5, .5, .5]))
    ev = M.evaluate_frame(frame, "pred", min_n=1)
    values = frame_diagnostics(frame, ev, args, build_loss(args))
    assert values["loss"] == pytest.approx(2.75/3)
    assert values["calibration/all/std_ratio"] == pytest.approx(.5)
    assert values["calibration/all/slope"] == pytest.approx(2)
    assert values["calibration/all/bias"] == pytest.approx(-.5)
    assert values["interaction/double_rmse_gain_over_additive_macro"] == pytest.approx(1.)
    assert values["interaction/complete_groups"] == 1
    assert values["interaction/observed_epsilon_rms"] == pytest.approx(3.)
    assert values["interaction/predicted_epsilon_rms"] == pytest.approx(1.5)


@pytest.mark.parametrize("mode", ["joint_dropout", "additive", "shared_background"])
def test_diagnostics_preserve_training_and_emit_complete_metrics(tmp_path, mode):
    from fixtures import make_synthetic_dataset
    data, _ = make_synthetic_dataset(tmp_path/"data", n_targets=5, seq_len=24,
        n_singles=40, n_doubles=40, n_triples=0)
    extra = {"joint_dropout": ["--dropout", ".2"], "additive": ["--composition", "additive"],
        "shared_background": ["--shared-lora-r", "2", "--head", "wt_relative", "--reference-augmentation", "background",
            "--augmentation-fraction", "1", "--single-only-exposures", ".5", "--loss", "huber_dg",
            "--single-weight", ".1", "--task-reduction", "depth", "--stabilizing-double-weight", "4"]}[mode]
    def args(directory, diagnostics):
        return build_parser().parse_args(["--data-dir", str(data), "--dataset", "synthetic", "--split", "protein",
            "--backbone", "random:64:4:2", "--device", "cpu", "--precision", "fp32", "--lora-r", "2",
            "--exposures", "1", "--micro-batch", "16", "--proteins-per-step", "2", "--eval-min-n", "3",
            "--selection", "spearman", "--evals-per-exposure", "2", "--output", str(directory), *extra,
            *(["--diagnostics", "--diagnostic-log-every", "2", "--diagnostic-profile-every", "3",
               "--diagnostic-probe-proteins", "2", "--diagnostic-probe-rows", "32",
               "--diagnostic-activation-proteins", "1", "--diagnostic-activation-rows", "8",
               "--diagnostic-plot-rows", "8"] if diagnostics else [])])
    instrumented = run(args(tmp_path/"on", True))
    state = torch.random.get_rng_state().clone()
    baseline = run(args(tmp_path/"off", False))
    assert torch.equal(state, torch.random.get_rng_state())
    for key, tensor in instrumented["model"].state_dict().items():
        torch.testing.assert_close(tensor, baseline["model"].state_dict()[key], rtol=0, atol=0)
    pd.testing.assert_series_equal(pd.read_csv(tmp_path/"on/predictions_val.csv").pred,
        pd.read_csv(tmp_path/"off/predictions_val.csv").pred)
    assert instrumented["extra"]["selection"] == baseline["extra"]["selection"]
    d = tmp_path/"on/diagnostics"
    records = [json.loads(line) for line in (d/"history.jsonl").read_text().splitlines()]
    assert {"optimization", "train_probe", "val", "activation", "timing"} <= {r["kind"] for r in records}
    assert [r["step"] for r in records if r["kind"] == "activation"] == [0, 8]
    assert [r["step"] for r in records if r["kind"] == "optimization"] == [2, 4, 6, 8]
    train_rows = [r for r in records if r["kind"] == "train_probe"]
    val_rows = [r for r in records if r["kind"] == "val"]
    assert all("train_probe/loss" in r and "train_probe/all_spearman_pooled" in r for r in train_rows)
    assert all("val/loss" in r and "val/all_spearman_pooled" in r for r in val_rows)
    selected = next(r for r in records if r["kind"] == "selected_checkpoint")
    assert "selected/train_probe/loss" in selected and "selected/val/loss" in selected
    assert len(pd.read_csv(d/"train_probe.csv")) <= 64
    assert len(pd.read_csv(d/"val_scatter_step_0.csv")) <= 8
    exposure = pd.read_csv(d/"reference_exposure.csv")
    assert exposure.rows_seen.sum() == instrumented["extra"]["timing"]["train_rows_seen"]
    assert not (tmp_path/"on/predictions_test.csv").exists()
    assert not (tmp_path/"off/diagnostics").exists()
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in instrumented["model"].modules())

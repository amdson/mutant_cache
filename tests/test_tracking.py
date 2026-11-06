"""Check that tracking mirrors local results and never changes the training recipe."""
import builtins
import json
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from mutant_cache.tracking import ExperimentTracker, load_tracking_config


class FakeRun:
    id = "run123"
    name = "fake"
    project = "mutant-cache"
    entity = "test-team"
    url = "https://wandb.ai/test-team/mutant-cache/runs/run123"
    dir = "/tmp/test-wandb"

    def __init__(self):
        self.summary = {}
        self.logs = []
        self.exit_codes = []
        self.definitions = []

    def log(self, values):
        self.logs.append(values)

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def finish(self, exit_code):
        self.exit_codes.append(exit_code)


@pytest.fixture
def sdk(monkeypatch):
    run = FakeRun()
    calls = []
    def init(**kwargs):
        calls.append(kwargs)
        return run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, __version__="test"))
    return run, calls


def test_disabled_does_not_import_sdk_or_write_tracking_files(tmp_path, monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "wandb"
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with ExperimentTracker(SimpleNamespace(), tmp_path) as tracker:
        tracker.validation(dict(step=0, exposure=0, train_loss=float("nan"), lr_scale=1))
    assert not (tmp_path / "tracking.json").exists()


def test_missing_sdk_is_actionable_before_training(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(RuntimeError, match="optional SDK"):
        with ExperimentTracker(SimpleNamespace(wandb_mode="offline"), tmp_path):
            pytest.fail("training must not start")


@pytest.mark.parametrize("config", [dict(wandb_api_key="not-a-credential"),
    dict(wandb_mode="invalid"), dict(wandb_log_every=0), dict(wandb_tags="bad-tags"),
    dict(diagnostics="yes"), dict(diagnostic_log_every=0), dict(diagnostic_probe_rows=2.5)])
def test_invalid_tracking_presets_are_rejected(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_tracking_config(path)


def test_run_metadata_and_failed_training_are_recorded(tmp_path, sdk):
    run, calls = sdk
    args = SimpleNamespace(wandb_mode="online", seed=2, experiment_context=dict(
        sweep="tuning-v2", trial_id="B01", stage="B", attempt=2))
    with pytest.raises(ValueError, match="training failed"):
        with ExperimentTracker(args, tmp_path):
            raise ValueError("training failed")
    assert calls[0]["group"] == "tuning-v2"
    assert calls[0]["name"] == "B01-s2-attempt2"
    assert calls[0]["tags"] == ["stage-B"]
    assert run.exit_codes == [1]
    record = json.loads((tmp_path / "tracking.json").read_text())
    assert record["url"] == run.url and record["status"] == "failed"


def test_logging_failure_does_not_abort_training_and_still_finishes(tmp_path, sdk):
    run, _ = sdk
    def fail(*args, **kwargs):
        raise OSError("logger disconnected")
    run.log = fail
    with ExperimentTracker(SimpleNamespace(wandb_mode="offline", seed=0), tmp_path) as tracker:
        with pytest.warns(RuntimeWarning, match="training continues"):
            tracker.log({"train/loss": .5}, 1)
        tracker.log({"train/loss": .4}, 2)
    assert run.exit_codes == [0]
    record = json.loads((tmp_path / "tracking.json").read_text())
    assert record["status"] == "finished" and "logging_error" in record


def test_training_metrics_match_local_history_and_predictions(tmp_path, sdk):
    from fixtures import make_synthetic_dataset
    from mutant_cache.megascale import build_parser, run
    fake, calls = sdk
    data, _ = make_synthetic_dataset(tmp_path / "data", n_targets=5, seq_len=24,
        n_singles=16, n_doubles=16, n_triples=0)
    def args(directory, *extra):
        return build_parser().parse_args(["--data-dir", str(data), "--dataset", "synthetic",
            "--split", "protein", "--backbone", "random:64:4:2", "--device", "cpu",
            "--precision", "fp32", "--lora-r", "2", "--exposures", "1", "--micro-batch", "16",
            "--proteins-per-step", "2", "--eval-min-n", "3", "--selection", "spearman",
            "--output", str(directory), *extra])
    tracked = run(args(tmp_path / "tracked", "--wandb-mode", "offline", "--wandb-log-every", "1"))
    baseline = run(args(tmp_path / "baseline"))
    history = pd.read_csv(tmp_path / "tracked/history.csv")
    logged = [v for v in fake.logs if "val/all_spearman" in v]
    assert len(logged) == len(history)
    for entry, row in zip(logged, history.to_dict("records")):
        assert entry["trainer/step"] == row["step"]
        assert entry["val/all_spearman"] == pytest.approx(row["val_all_spearman"])
        assert entry["val/double_rmse"] == pytest.approx(row["val_double_rmse"])
    assert "train/loss" not in logged[0] # untrained loss is undefined
    progress = [v for v in fake.logs if "train/loss_running" in v]
    assert progress and progress[-1]["progress/fraction"] == 1
    assert "lr/head" in progress[-1] and "lr/variant_lora" in progress[-1]
    assert fake.summary["selected/step"] == tracked["extra"]["selection"]["step"]
    assert fake.summary["selected/val/all_spearman_macro"] == pytest.approx(
        history.loc[history.step == fake.summary["selected/step"], "val_all_spearman"].iloc[0])
    assert fake.exit_codes == [0] and calls[0]["mode"] == "offline"
    assert not (tmp_path / "tracked/predictions_test.csv").exists()
    a = pd.read_csv(tmp_path / "tracked/predictions_val.csv").pred
    b = pd.read_csv(tmp_path / "baseline/predictions_val.csv").pred
    pd.testing.assert_series_equal(a, b)
    assert tracked["extra"]["selection"] == baseline["extra"]["selection"]

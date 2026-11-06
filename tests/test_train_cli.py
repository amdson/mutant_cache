import json
from pathlib import Path

import pytest
from mutant_cache import train as cli

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "experiments" / "recipes"


def write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("MUTANT_CACHE_SCRATCH", str(tmp_path / "scratch"))
    ready = write(tmp_path / "ready.json", {"id": "r", "status": "ready",
                  "training": {"dataset": "toy", "split": "toy_split"}})
    planned = write(tmp_path / "planned.json", {"id": "p", "status": "planned",
                    "training": {"dataset": "toy", "split": "toy_split"}})
    recipe = write(tmp_path / "recipe.json", {"lr": 0.002, "lora_r": 4, "exposures": 5})
    (tmp_path / "exp").mkdir()
    experiment = write(tmp_path / "exp" / "experiment.json",
                       {"id": "e", "protocol": "../ready.json", "recipe": "../recipe.json",
                        "seeds": [0, 1], "training": {"exposures": 2}})
    return tmp_path, ready, planned, recipe, experiment


@pytest.mark.parametrize("name", ["esm2_650m_shared.json", "esmc_600m_shared.json"])
def test_recipes_use_megascale_names_without_protocol_identity(name):
    recipe = json.loads((RECIPES / name).read_text())
    assert not {"dataset", "split", "data_dir"} & set(recipe)
    protocol = {"status": "ready", "training": {"dataset": "d", "split": "s"}}
    config = cli.resolve_config(recipe, protocol)
    assert config["shared_lora_r"] == 8 and config["head"] == "wt_relative"


def test_precedence_recipe_experiment_cli(files):
    _, ready, _, recipe, experiment = files
    args, own = cli.resolve(["--experiment", str(experiment), "--seed", "1", "--lora-r", "8",
                             "--dry-run"])
    assert own["dry_run"]
    assert (args.lr, args.exposures, args.lora_r, args.seed) == (0.002, 2, 8, 1)
    assert (args.dataset, args.split) == ("toy", "toy_split")
    assert args.protocol_path == str(ready.resolve()) and args.recipe_path == str(recipe.resolve())
    args, _ = cli.resolve(["--protocol", str(ready), "--recipe", str(recipe), "--lr", "0.001"])
    assert (args.lr, args.exposures) == (0.001, 5)


def test_explicit_default_valued_flag_still_overrides(files):
    _, ready, _, recipe, _ = files
    default_r = cli.megascale.build_parser().parse_args([]).lora_r
    args, _ = cli.resolve(["--protocol", str(ready), "--recipe", str(recipe),
                           "--lora-r", str(default_r)])
    assert args.lora_r == default_r


def test_planned_protocol_rejected_in_dry_run(files, capsys):
    _, _, planned, recipe, _ = files
    assert cli.main(["--protocol", str(planned), "--recipe", str(recipe), "--seed", "0",
                     "--dry-run"]) == 2
    assert "planned" in capsys.readouterr().err


def test_errors(files, capsys):
    _, ready, _, recipe, experiment = files
    base = ["--protocol", str(ready), "--recipe", str(recipe), "--dry-run"]
    for argv in (base + ["--dataset", "other"], base + ["--data-dir", "/x"],
                 ["--experiment", str(experiment), "--dry-run"],
                 ["--experiment", str(experiment), "--seed", "9", "--dry-run"],
                 ["--recipe", str(recipe), "--dry-run"]):
        assert cli.main(argv) == 2
        assert "error:" in capsys.readouterr().err


def test_dry_run_prints_and_writes_nothing(files, capsys):
    tmp_path, ready, _, recipe, _ = files
    before = sorted(tmp_path.rglob("*"))
    assert cli.main(["--protocol", str(ready), "--recipe", str(recipe), "--seed", "3",
                     "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["config"]["seed"] == 3 and out["output"].startswith(str(tmp_path / "scratch"))
    assert not out["data_dir_exists"]
    assert sorted(tmp_path.rglob("*")) == before

"""CPU-only tests for the parallel queue runner; a stub stands in for mutant_cache.train."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mutant_cache import queue_runner as qr

REPO = Path(__file__).resolve().parents[1]

STUB = r'''
import json, os, sys, time
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--output") + 1])
tag = argv[argv.index("--tag") + 1] if "--tag" in argv else "ok"
out.mkdir(parents=True, exist_ok=True)
(out / "resolved_config.json").write_text(json.dumps({"save_model": True}))
start = time.time()
time.sleep(float(os.environ.get("STUB_SLEEP", "0.4")))
(out / "seen.json").write_text(json.dumps({"gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                                          "start": start, "end": time.time()}))
if tag == "fail" or (tag == "flaky" and not (out.parent.parent / "flaky_ok").exists()):
    sys.exit(3)
(out / "results.json").write_text("{}")
(out / "model.pt").write_text("")
'''


def write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MUTANT_CACHE_SCRATCH", str(tmp_path / "scratch"))
    split = tmp_path / "scratch" / "processed" / "toy" / "splits" / "toy_split.parquet"
    split.parent.mkdir(parents=True)
    split.touch()
    write(tmp_path / "ready.json", {"id": "r", "status": "ready",
                                    "training": {"dataset": "toy", "split": "toy_split"}})
    write(tmp_path / "planned.json", {"id": "p", "status": "planned",
                                      "training": {"dataset": "toy", "split": "toy_split"}})
    write(tmp_path / "recipe.json", {"lr": 0.002})
    (tmp_path / "exp").mkdir()
    write(tmp_path / "exp" / "experiment.json", {"id": "toy_exp", "protocol": "../ready.json",
                                                 "recipe": "../recipe.json", "seeds": [0, 1]})
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    return tmp_path, [sys.executable, str(stub)]


def jobs_file(tmp_path, jobs=(), experiments=()):
    return write(tmp_path / "jobs.json", {"jobs": list(jobs), "experiments": list(experiments)})


def job(name, tag="ok", protocol="ready.json"):
    return {"name": name, "protocol": protocol, "recipe": "recipe.json", "seed": 0,
            "args": ["--tag", tag]}


def run(path, root, cmd, *extra):
    return qr.main(["run", str(path), "--root", str(root), "--poll", "0.05",
                    "--train-cmd", " ".join(cmd), *extra])


def seen(root, name):
    return json.loads((root / "runs" / name / "seen.json").read_text())


def test_experiment_seeds_expand_and_example_parses():
    jobs = qr.load_jobs(REPO / "experiments" / "stability_paper" / "jobs_example.json")
    assert [j["name"] for j in jobs] == [f"{e}_s{s}" for e in ("me_esm2_v1", "msr_esm2_v1")
                                         for s in range(3)]
    root = Path("/q")
    argv = qr.train_argv(jobs[4], root)
    assert argv[argv.index("--seed") + 1] == "1" and argv[-2:] == ["--output", "/q/runs/msr_esm2_v1_s1"]


def test_bad_specs(env, capsys):
    tmp_path, _ = env
    for jobs in ([job("a"), job("a")], [job("bad/name")],
                 [dict(job("a"), args=["--output", "/x"])], [{"name": "a", "recipe": "recipe.json"}]):
        assert qr.main(["status", str(jobs_file(tmp_path, jobs)), "--root", str(tmp_path / "q")]) == 2
        assert "error:" in capsys.readouterr().err


def test_gpu_discovery(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert qr.discover_gpus() == ["3", "5"] and qr.discover_gpus("0,2") == ["0", "2"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert qr.discover_gpus() == []


def test_dry_run_prints_and_writes_nothing(env, capsys):
    tmp_path, cmd = env
    path = jobs_file(tmp_path, [job("a"), job("b"), job("c")])
    before = sorted(tmp_path.rglob("*"))
    assert run(path, tmp_path / "q", cmd, "--gpus", "0,1", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "# a: gpu 0" in out and "# b: gpu 1" in out and "# c: queued" in out
    assert f"--output {tmp_path / 'q' / 'runs' / 'a'}" in out
    assert sorted(tmp_path.rglob("*")) == before


def test_no_gpus_is_an_error(env, capsys):
    tmp_path, cmd = env
    assert run(jobs_file(tmp_path, [job("a")]), tmp_path / "q", cmd) == 2
    assert "no GPUs" in capsys.readouterr().err


def test_invalid_job_blocks_launch(env, capsys):
    tmp_path, cmd = env
    path = jobs_file(tmp_path, [job("a"), job("b", protocol="planned.json")])
    assert run(path, tmp_path / "q", cmd, "--gpus", "0") == 2
    assert "planned" in capsys.readouterr().err and not (tmp_path / "q").exists()


def test_parallel_one_job_per_gpu_and_skip_completed(env, capsys):
    tmp_path, cmd = env
    path = jobs_file(tmp_path, experiments=[{"experiment": "exp/experiment.json"}],
                     jobs=[job("c"), job("d")])
    root = tmp_path / "q"
    assert run(path, root, cmd, "--gpus", "0,1") == 0
    names = ["toy_exp_s0", "toy_exp_s1", "c", "d"]
    rows = {n: seen(root, n) for n in names}
    assert {r["gpu"] for r in rows.values()} == {"0", "1"}
    for gpu in ("0", "1"):  # serial within a GPU
        spans = sorted((r["start"], r["end"]) for r in rows.values() if r["gpu"] == gpu)
        assert len(spans) == 2 and spans[0][1] <= spans[1][0]
    first = sorted(rows.values(), key=lambda r: r["start"])
    assert first[1]["start"] < first[0]["end"]  # parallel across GPUs
    state = json.loads((root / "queue_state.json").read_text())
    assert all(state["jobs"][n]["exit_code"] == 0 and state["jobs"][n]["status"] == "done"
               for n in names)
    assert (root / "logs" / "c.log").read_text().startswith("\n==== ")
    mtimes = {n: (root / "runs" / n / "seen.json").stat().st_mtime for n in names}
    capsys.readouterr()
    assert run(path, root, cmd, "--gpus", "0,1") == 0
    assert "started" not in capsys.readouterr().out
    assert mtimes == {n: (root / "runs" / n / "seen.json").stat().st_mtime for n in names}


def test_failure_isolated_then_retry_moves_aside(env, capsys):
    tmp_path, cmd = env
    path = jobs_file(tmp_path, [job("bad", "flaky"), job("x"), job("y")])
    root = tmp_path / "q"
    assert run(path, root, cmd, "--gpus", "0,1") == 1
    assert "failed: bad" in capsys.readouterr().err
    assert qr.complete(root / "runs" / "x") and qr.complete(root / "runs" / "y")
    assert qr.main(["status", str(path), "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "done=2 running=0 failed=1 pending=0" in out
    assert run(path, root, cmd, "--gpus", "0,1") == 1  # not silently overwritten
    assert "use --retry-failed" in capsys.readouterr().err
    assert not (root / "runs" / "bad" / "results.json").exists()
    (root / "flaky_ok").touch()
    assert run(path, root, cmd, "--gpus", "0,1", "--retry-failed") == 0
    aside = list((root / "runs").glob("bad.failed-*"))
    assert len(aside) == 1 and (aside[0] / "seen.json").exists()
    assert qr.complete(root / "runs" / "bad")
    assert (root / "logs" / "bad.log").read_text().count("==== ") == 2


def test_adopts_job_left_running_by_earlier_runner(env, capsys):
    tmp_path, cmd = env
    path = jobs_file(tmp_path, [job("a"), job("b")])
    root = tmp_path / "q"
    (root / "runs").mkdir(parents=True)
    out = root / "runs" / "a"
    orphan = subprocess.Popen(cmd + ["--output", str(out)], start_new_session=True,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", STUB_SLEEP="1"))
    qr.write_json(root / "queue_state.json", {"jobs": {"a": {"gpu": "0", "pid": orphan.pid,
                                                            "exit_code": None}}})
    time.sleep(0.2)
    assert qr.status_table(qr.load_jobs(path), root)[0][1] == "running"
    assert run(path, root, cmd, "--gpus", "0") == 0
    orphan.wait()
    assert seen(root, "b")["start"] >= seen(root, "a")["end"]  # waited for the orphan's GPU
    assert "a started" not in capsys.readouterr().out

"""Run `mutant_cache.train` jobs in parallel, one job per GPU.

    python -m mutant_cache.queue_runner run JOBS.json [--root DIR] [--gpus 0,2] [--dry-run]
    python -m mutant_cache.queue_runner status JOBS.json [--root DIR]

JOBS.json: {"jobs": [{"name", "experiment" | "protocol"+"recipe", "seed", "args": [...]}],
            "experiments": [{"experiment", "seeds"?, "args"?, "prefix"?}]}
Paths are relative to JOBS.json. Each `experiments` entry expands to one job per seed
named `<prefix or experiment id>_s<seed>`. Job `name` writes to `<root>/runs/<name>`
(passed as --output) and logs to `<root>/logs/<name>.log`; state is `<root>/queue_state.json`.
A job is done when its run directory has results.json (and model.pt if save_model).
A populated incomplete directory is failed and is never overwritten; `--retry-failed`
moves it to `<name>.failed-<timestamp>` first. Workers run in their own session and
survive the runner; a later `run` waits for them instead of relaunching.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = [sys.executable, "-m", "mutant_cache.train"]
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OWNED = ("--output", "--out-root", "--dry-run", "--experiment", "--protocol", "--recipe", "--seed")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, value):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def load_jobs(path) -> list[dict]:
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    rel = lambda p: str((path.parent / p).resolve())
    jobs = []
    for job in spec.get("jobs", []):
        job = dict(job, args=list(job.get("args", [])))
        for key in ("experiment", "protocol", "recipe"):
            if job.get(key):
                job[key] = rel(job[key])
        jobs.append(job)
    for entry in spec.get("experiments", []):
        experiment = rel(entry["experiment"])
        info = json.loads(Path(experiment).read_text())
        seeds = entry.get("seeds", info.get("seeds"))
        if not seeds:
            raise ValueError(f"no seeds for {experiment}")
        prefix = entry.get("prefix") or info["id"]
        jobs += [{"name": f"{prefix}_s{s}", "experiment": experiment, "seed": s,
                  "args": list(entry.get("args", []))} for s in seeds]
    names = [j.get("name") for j in jobs]
    for job in jobs:
        if not isinstance(job.get("name"), str) or not NAME.match(job["name"]):
            raise ValueError(f"bad job name {job.get('name')!r}")
        if names.count(job["name"]) > 1:
            raise ValueError(f"duplicate job name {job['name']!r}")
        pair = [bool(job.get(k)) for k in ("protocol", "recipe")]
        if [bool(job.get("experiment"))] * 2 != [not v for v in pair]:
            raise ValueError(f"{job['name']}: give experiment or protocol+recipe")
        if any(str(a).split("=")[0] in OWNED for a in job["args"]):
            raise ValueError(f"{job['name']}: args may not set {'/'.join(OWNED)}")
    return jobs


def train_argv(job: dict, root: Path) -> list[str]:
    argv = ["--experiment", job["experiment"]] if job.get("experiment") else \
        ["--protocol", job["protocol"], "--recipe", job["recipe"]]
    if job.get("seed") is not None:
        argv += ["--seed", str(job["seed"])]
    return argv + [str(a) for a in job["args"]] + ["--output", str(root / "runs" / job["name"])]


def validate(jobs, root: Path) -> list[str]:
    """Resolve every job through the train CLI; return error lines."""
    import io
    from mutant_cache import train
    from mutant_cache.harness import resolve_data_dir
    errors = []
    for job in jobs:
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                args, _ = train.resolve(train_argv(job, root))
            split = resolve_data_dir(args) / "splits" / f"{args.split}.parquet"
            if not split.is_file():
                errors.append(f"{job['name']}: split not found: {split}")
        except SystemExit:
            lines = stderr.getvalue().strip().splitlines() or ["invalid arguments"]
            errors.append(f"{job['name']}: {lines[-1]}")
        except (ValueError, OSError) as e:
            errors.append(f"{job['name']}: {e}")
    return errors


def complete(run_dir: Path) -> bool:
    if not (run_dir / "results.json").is_file():
        return False
    try:
        save = json.loads((run_dir / "resolved_config.json").read_text()).get("save_model")
    except (OSError, ValueError):
        save = True
    return not save or (run_dir / "model.pt").is_file()


def alive(pid, run_dir) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except (OSError, TypeError):
        return False
    return str(run_dir) in cmdline.split("\0")  # guards against pid reuse


def classify(job, root: Path, entry: dict) -> str:
    run_dir = root / "runs" / job["name"]
    if complete(run_dir):
        return "done"
    if entry.get("pid") and entry.get("exit_code") is None and alive(entry["pid"], run_dir):
        return "running"
    if (run_dir.exists() and any(run_dir.iterdir())) or entry.get("exit_code") not in (None, 0):
        return "failed"
    return "pending"


def discover_gpus(override=None) -> list[str]:
    if override is not None:
        return [g for g in override.split(",") if g.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [g.strip() for g in visible.split(",") if g.strip()]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [str(i) for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]


def read_state(root: Path) -> dict:
    try:
        return json.loads((root / "queue_state.json").read_text())
    except (OSError, ValueError):
        return {"jobs": {}}


def status_table(jobs, root: Path) -> list[tuple]:
    entries = read_state(root).get("jobs", {})
    rows = []
    for job in jobs:
        e = entries.get(job["name"], {})
        rows.append((job["name"], classify(job, root, e), e.get("gpu", ""), e.get("pid", ""),
                     e.get("exit_code", ""), e.get("start", ""), e.get("end", "")))
    return rows


def print_table(rows):
    head = ("job", "status", "gpu", "pid", "exit", "start", "end")
    rows = [head] + [tuple("" if v is None else str(v) for v in r) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())


def run(jobs, root: Path, gpus, *, cmd=TRAIN, retry_failed=False, dry_run=False, poll=10.0) -> int:
    if not gpus:
        print("error: no GPUs visible (set --gpus or CUDA_VISIBLE_DEVICES)", file=sys.stderr)
        return 2
    errors = validate(jobs, root)
    if errors:
        print("error: job validation failed:\n  " + "\n  ".join(errors), file=sys.stderr)
        if not dry_run:
            return 2
    state = read_state(root)
    entries = state.setdefault("jobs", {})
    status = {j["name"]: classify(j, root, entries.get(j["name"], {})) for j in jobs}
    failed = [n for n, s in status.items() if s == "failed"]
    pending = [j for j in jobs if status[j["name"]] == "pending" or
               (retry_failed and status[j["name"]] == "failed")]
    adopted = {entries[n]["gpu"]: n for n, s in status.items() if s == "running"}
    if dry_run:
        free = [g for g in gpus if g not in adopted]
        for n, s in status.items():
            if s in ("done", "running") or (s == "failed" and not retry_failed):
                print(f"# {n}: {s}" + (" (use --retry-failed)" if s == "failed" else ""))
        for i, job in enumerate(pending):
            where = f"gpu {free[i]}" if i < len(free) else "queued (next free gpu)"
            move = " [moves partial dir aside]" if status[job["name"]] == "failed" else ""
            print(f"# {job['name']}: {where}{move}")
            print(f"CUDA_VISIBLE_DEVICES={free[i] if i < len(free) else '<gpu>'} "
                  + shlex.join(list(cmd) + train_argv(job, root)))
        return 2 if errors else 0
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    lock = (root / "queue.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"error: another queue runner holds {root / 'queue.lock'}", file=sys.stderr)
        return 2
    state["gpus"] = gpus
    save = lambda: write_json(root / "queue_state.json", dict(state, updated=now()))
    busy = {g: (n, None) for g, n in adopted.items() if g in gpus}
    for g, n in adopted.items():
        if g not in gpus:
            print(f"warning: {n} still running on gpu {g}, outside --gpus", file=sys.stderr)
    for name in failed:
        if not retry_failed:
            print(f"skip {name}: failed/partial (use --retry-failed)", file=sys.stderr)
    bad = []
    while pending or busy:
        for gpu, (name, proc) in list(busy.items()):
            e = entries[name]
            if proc is None:  # adopted from an earlier runner; no exit code available
                if alive(e["pid"], root / "runs" / name):
                    continue
                e["exit_code"] = 0 if complete(root / "runs" / name) else -1
            else:
                if proc.poll() is None:
                    continue
                e["exit_code"] = proc.returncode
            e["end"] = now()
            ok = e["exit_code"] == 0 and complete(root / "runs" / name)
            e["status"] = "done" if ok else "failed"
            print(f"[{e['end']}] {name} {e['status']} (exit {e['exit_code']}, gpu {gpu})", flush=True)
            if not ok:
                bad.append(name)
            del busy[gpu]
            save()
        for gpu in [g for g in gpus if g not in busy]:
            if not pending:
                break
            job = pending.pop(0)
            name, run_dir = job["name"], root / "runs" / job["name"]
            if run_dir.exists() and any(run_dir.iterdir()):
                aside = run_dir.with_name(f"{name}.failed-{datetime.datetime.now():%Y%m%d-%H%M%S}")
                run_dir.rename(aside)
                print(f"moved partial {run_dir} -> {aside}", flush=True)
            argv = list(cmd) + train_argv(job, root)
            log = (root / "logs" / f"{name}.log").open("a")
            log.write(f"\n==== {now()} gpu {gpu}: {shlex.join(argv)}\n")
            log.flush()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            entries[name] = {"status": "running", "gpu": gpu, "pid": proc.pid, "start": now(),
                             "end": None, "exit_code": None, "output": str(run_dir),
                             "log": log.name, "command": argv}
            busy[gpu] = (name, proc)
            print(f"[{entries[name]['start']}] {name} started on gpu {gpu} (pid {proc.pid})", flush=True)
            save()
        if pending or busy:
            time.sleep(poll)
    save()
    skipped = [] if retry_failed else failed
    if bad or skipped:
        print(f"failed: {', '.join(bad + skipped)}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        s = sub.add_parser(name)
        s.add_argument("jobs", help="jobs JSON")
        s.add_argument("--root", default=None,
                       help="queue root (default $MUTANT_CACHE_SCRATCH/queues/<jobs file stem>)")
        if name == "run":
            s.add_argument("--gpus", default=None,
                           help="comma list (default CUDA_VISIBLE_DEVICES, else nvidia-smi -L)")
            s.add_argument("--dry-run", action="store_true",
                           help="validate jobs and print commands and GPU assignment; launch nothing")
            s.add_argument("--retry-failed", action="store_true",
                           help="move failed/partial run dirs aside and rerun them")
            s.add_argument("--poll", type=float, default=10.0, help="seconds between checks")
            s.add_argument("--train-cmd", default=None,
                           help="replace `python -m mutant_cache.train` (testing)")
    a = p.parse_args(argv)
    try:
        jobs = load_jobs(a.jobs)
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.root:
        root = Path(a.root).resolve()
    else:
        from mutant_cache.harness import scratch_root
        root = scratch_root() / "queues" / Path(a.jobs).stem
    if a.command == "status":
        rows = status_table(jobs, root)
        print_table(rows)
        counts = {s: sum(r[1] == s for r in rows) for s in ("done", "running", "failed", "pending")}
        print(" ".join(f"{k}={v}" for k, v in counts.items()))
        return 0
    cmd = shlex.split(a.train_cmd) if a.train_cmd else TRAIN
    return run(jobs, root, discover_gpus(a.gpus), cmd=cmd, retry_failed=a.retry_failed,
               dry_run=a.dry_run, poll=a.poll)


if __name__ == "__main__":
    raise SystemExit(main())

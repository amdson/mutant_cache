"""Single-run megascale training entry point and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from mutant_cache import megascale
from mutant_cache.harness import resolve_data_dir, scratch_root


def _read(path: str | Path) -> dict:
    with Path(path).open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _config_keys() -> set[str]:
    return set(vars(megascale.build_parser().parse_args([])))


def _normalize(values: dict, *, source: str) -> dict:
    keys = _config_keys()
    result = {}
    for key, value in values.items():
        dest = key.replace("-", "_")
        if dest not in keys:
            raise ValueError(f"unknown {source} key {key!r}")
        result[dest] = value
    return result


def resolve_config(recipe: dict, protocol: dict, overrides: dict | None = None) -> dict:
    """Resolve defaults, protocol identity, recipe, then experiment overrides."""
    if protocol.get("status") != "ready":
        raise ValueError(f"protocol status is {protocol.get('status')!r}; training requires 'ready'")
    training = protocol.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("dataset"), str) \
            or not isinstance(training.get("split"), str):
        raise ValueError("ready protocol requires training.dataset and training.split")
    defaults = vars(megascale.build_parser().parse_args([]))
    identity = {"dataset": training["dataset"], "split": training["split"]}
    for source, mapping in (("recipe", recipe), ("training override", overrides or {})):
        normalized = _normalize(mapping, source=source)
        for key in identity:
            if key in normalized and normalized[key] != identity[key]:
                raise ValueError(f"{source} cannot override protocol {key}={identity[key]!r}")
        defaults.update(normalized)
    defaults.update(identity)
    return defaults


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source_files() -> list[Path]:
    package = Path(__file__).resolve().parent
    return sorted(package.rglob("*.py"))


def _provenance(args, run_dir: Path, recipe_path=None, protocol_path=None) -> dict:
    repo = Path(__file__).resolve().parents[1]
    snapshot = run_dir / "code_snapshot" / "mutant_cache"
    sources = _source_files()
    code_hashes = {}
    for source in sources:
        rel = source.relative_to(Path(__file__).resolve().parent)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        code_hashes[str(rel)] = _sha256(source)
    data_dir = resolve_data_dir(args)
    data_hashes = {}
    if data_dir.exists():
        for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
            data_hashes[str(path.relative_to(data_dir))] = _sha256(path)
    try:
        git = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = None
    return {"git_commit": git, "recipe": str(recipe_path) if recipe_path else None,
            "protocol": str(protocol_path) if protocol_path else None,
            "dataset_dir": str(data_dir), "dataset_sha256": data_hashes,
            "code_sha256": code_hashes}


def train(config):
    """Train one megascale run from an argparse Namespace or resolved mapping."""
    if isinstance(config, dict):
        config = dict(config)
        config.pop("dry_run", None)
        args = megascale.build_parser().parse_args([])
        for key, value in config.items():
            if key not in vars(args):
                raise ValueError(f"unknown config key {key!r}")
            setattr(args, key, value)
    else:
        args = config
    output = getattr(args, "output", None)
    if output:
        run_dir = Path(output)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"output directory is populated: {run_dir}")
    elif getattr(args, "tag", None):
        run_dir = (Path(args.out_root) if args.out_root else scratch_root() / "runs") / \
            f"{args.dataset}_{args.tag}_{uuid.uuid4().hex[:10]}"
        args.output = str(run_dir)
    else:
        run_dir = None
    if run_dir is None:
        # Preserve megascale's established deterministic output naming.
        return megascale.run(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    provenance = _provenance(args, run_dir, getattr(args, "recipe_path", None),
                             getattr(args, "protocol_path", None))
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return megascale.run(args)


"""Train one seed from a protocol and a recipe.

    python -m mutant_cache.train --protocol P.json --recipe R.json --seed 0 [megascale flags]
    python -m mutant_cache.train --experiment E.json --seed 0 [megascale flags] [--dry-run]

Precedence: megascale defaults < recipe < experiment `training` < explicit CLI
flags. The protocol's training dataset/split cannot be overridden.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from mutant_cache import megascale
from mutant_cache.harness import resolve_data_dir, scratch_root
from mutant_cache.training import _read, resolve_config, train

OWN = ("experiment", "protocol", "recipe", "dry_run")


def build_parser() -> argparse.ArgumentParser:
    p = megascale.build_parser()
    p.description = __doc__
    g = p.add_argument_group("experiment")
    g.add_argument("--experiment", default=None,
                   help="experiment.json; its protocol/recipe paths are relative to it")
    g.add_argument("--protocol", default=None, help="protocol JSON (status must be 'ready')")
    g.add_argument("--recipe", default=None, help="recipe JSON of megascale argument names")
    g.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and output directory; load and write nothing")
    return p


def explicit_args(argv) -> dict:
    """Only the flags actually present on the command line."""
    p = build_parser()
    for action in p._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    p._defaults.clear()
    return vars(p.parse_args(argv))


def _seed(experiment: dict, cli: dict):
    seeds = experiment.get("seeds")
    if not seeds:
        return cli.get("seed")
    if "seed" in cli:
        if cli["seed"] not in seeds:
            raise ValueError(f"--seed {cli['seed']} is not in the experiment's seeds {seeds}")
        return cli["seed"]
    if len(seeds) == 1:
        return seeds[0]
    raise ValueError(f"experiment lists seeds {seeds}; one launch runs one seed, pass --seed")


def resolve(argv) -> tuple[argparse.Namespace, dict]:
    build_parser().parse_args(argv)  # full validation and --help
    cli = explicit_args(argv)
    own = {k: cli.pop(k) for k in OWN if k in cli}
    experiment, overrides = {}, {}
    if own.get("experiment"):
        if own.get("protocol") or own.get("recipe"):
            raise ValueError("pass --experiment or --protocol/--recipe, not both")
        experiment_path = Path(own["experiment"]).resolve()
        experiment = _read(experiment_path)
        for key in ("protocol", "recipe"):
            if not isinstance(experiment.get(key), str):
                raise ValueError(f"experiment requires a {key!r} path")
        protocol_path = (experiment_path.parent / experiment["protocol"]).resolve()
        recipe_path = (experiment_path.parent / experiment["recipe"]).resolve()
        overrides.update(experiment.get("training") or {})
    else:
        if not (own.get("protocol") and own.get("recipe")):
            raise ValueError("pass --experiment, or both --protocol and --recipe")
        experiment_path = None
        protocol_path = Path(own["protocol"]).resolve()
        recipe_path = Path(own["recipe"]).resolve()
    seed = _seed(experiment, cli)
    overrides.update(cli)
    if seed is not None:
        overrides["seed"] = seed
    recipe, protocol = _read(recipe_path), _read(protocol_path)
    config = resolve_config(recipe, protocol, overrides)
    if config.get("data_dir"):
        raise ValueError("--data-dir would bypass the protocol dataset; use the processed dataset")
    args = megascale.build_parser().parse_args([])
    for key, value in config.items():
        setattr(args, key, value)
    if not args.output:
        root = Path(args.out_root) if args.out_root else scratch_root() / "runs"
        label = args.tag or experiment.get("id") or recipe_path.stem
        args.output = str(root / f"{args.dataset}_{label}_s{args.seed}_{uuid.uuid4().hex[:10]}")
    args.recipe_path = str(recipe_path)
    args.protocol_path = str(protocol_path)
    args.experiment_path = str(experiment_path) if experiment_path else None
    return args, own


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        args, own = resolve(argv)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if own.get("dry_run"):
        data_dir = resolve_data_dir(args)
        print(json.dumps({"output": args.output, "data_dir": str(data_dir),
                          "data_dir_exists": data_dir.exists(),
                          "split_exists": (data_dir / "splits" / f"{args.split}.parquet").exists(),
                          "config": vars(args)}, indent=2, default=str))
        return 0
    return 0 if train(args).get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

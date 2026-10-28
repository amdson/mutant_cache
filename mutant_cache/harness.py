"""
mutant_cache/harness.py
-----------------------
The one shared run loop, in the shape ``~/svar/variant_ll/harness.py`` uses: a
thin ``run.py`` parses args and hands them here, this does everything — load
the canonical tables, fit the baselines, build the model, train, evaluate at
every depth, run the full-recompute control, and write a run record.

Why one loop rather than a script per experiment.  The experiments PLAN.md
calls for differ only in *which rows train*, *which ablation is on*, and
*which backend runs*:

    the gate (Phase 6)   --split depth --ablation none
    the additive control --split depth --ablation additive
    feature ablation     --ablation no_cross_attn | random_ref | position_only
    approximation control --control-backend exact      (same weights, full forward)
    numerics check       --control-backend recompute   (same weights, cache oracle)

Everything else — the split, the batching, the loss, the depth buckets, the
reliability ceilings — has to stay identical or the numbers stop being
comparable, and in the predecessor repo two of these started as throwaway
scripts whose numbers were comparable only by luck.

Every knob that changes a number is a CLI argument and lands in ``run.json``.

The unit is **Spearman within score set, stratified by mutation depth**, always
printed beside the assay's reliability ceiling (DATASETS.md: "0.6 against a
test-retest of 0.7 reads nothing like 0.6 against 0.95").
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mutant_cache.baselines import (
    EmbeddingMLP, EmbeddingRidge, MaskedMarginalZeroShot, OneHotRidgePerAssay,
    build_llr_matrices, mean_pooled_embeddings, score_variants,
)
from mutant_cache.eval import (
    ResultsTable, depth_spearman, depth_summary, epsilon_correlation,
)
from mutant_cache.models import (
    ABLATIONS, BACKENDS, LOSSES, LossConfig, VariantEffectModel,
    VariantModelConfig, VariantLoss, build_variant_batches,
)
from mutant_cache.models.data_compat import apply_split, load_split, load_tables

BASELINE_CHOICES = ("zero_shot", "onehot_ridge", "embedding_ridge", "embedding_mlp")


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_common_args(p: argparse.ArgumentParser) -> None:
    """Every knob that changes a number."""
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="megascale",
                   help="processed dataset name under "
                        "$MUTANT_CACHE_SCRATCH/processed/")
    g.add_argument("--data-dir", default=None,
                   help="use this processed directory directly instead of "
                        "resolving --dataset against scratch")
    g.add_argument("--split", default=None,
                   help="name of a table under splits/. Omit to train on "
                        "everything and evaluate on it too, which is a "
                        "capacity probe and nothing else.")
    g.add_argument("--eval-split", choices=["val", "test"], default="val",
                   help="tune on val, touch test once")
    g.add_argument("--score-sets", nargs="*", default=None,
                   help="restrict to these score_set_ids")
    g.add_argument("--max-n-mut", type=int, default=0,
                   help="drop variants deeper than this (0 = keep all)")
    g.add_argument("--min-variants-per-set", type=int, default=8,
                   help="drop score sets smaller than this. Spearman on fewer "
                        "than ~8 points is close to pure noise, and a ranking "
                        "loss on them is fitting the noise.")
    g.add_argument("--max-train-variants", type=int, default=0,
                   help="subsample the training rows (0 = all)")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("backbone")
    g.add_argument("--backbone", default="esmc_300m",
                   help="passed to mutant_cache.backbone.load_esmc. Two "
                        "special values: 'stub' loads tests/fixtures.StubESMC, "
                        "and 'random:D:H:L' builds the real ESM-C architecture "
                        "at a toy size with random weights — the real cache "
                        "code path, no checkpoint, meaningless numbers.")
    g.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    g.add_argument("--device", default=None)
    g.add_argument("--train-backbone", action="store_true",
                   help="unfreeze the backbone. Off by default: PLAN.md 17 "
                        "wants variant-only LoRA over a frozen bf16 base, and "
                        "a frozen reference is also what makes the reference "
                        "cache (and therefore the recompute control) honest.")
    g.add_argument("--lora-r", type=int, default=0,
                   help="variant-only LoRA rank (0 = off): adapters on the "
                        "variant stream alone; the reference stays a constant")
    g.add_argument("--shared-lora-r", type=int, default=0,
                   help="shared LoRA rank (0 = off): a low-rank change to the "
                        "base, applied to the reference AND variant streams, so "
                        "the wild-type context the variants read can move too. "
                        "Costs the reference cache during training (the "
                        "reference is recomputed with grad every batch)")
    g.add_argument("--lora-alpha", type=float, default=None,
                   help="LoRA alpha for both sets (default 2*r)")
    g.add_argument("--lora-targets", nargs="+", default=None,
                   help="variant LoRA projections, from q k v o gate up down; "
                        "default q o gate up down (the ones applied only to the "
                        "variant stream; adapting k/v makes variant and "
                        "reference keys use different projections)")
    g.add_argument("--shared-lora-targets", nargs="+", default=None,
                   help="shared LoRA projections (default q o gate up down)")
    g.add_argument("--lora-lr", type=float, default=None,
                   help="learning rate for both adapter sets (default --lr)")
    g.add_argument("--stub-dim", type=int, default=32, help="--backbone stub only")
    g.add_argument("--stub-layers", type=int, default=2, help="--backbone stub only")

    g = p.add_argument_group("model")
    g.add_argument("--ablation", choices=list(ABLATIONS), default="none",
                   help="DATASETS.md's four: additive (variant-variant "
                        "attention masked and the interaction head dropped), "
                        "no_cross_attn, random_ref, position_only")
    g.add_argument("--backend", choices=list(BACKENDS), default="cache",
                   help="the path the model trains and is scored through")
    g.add_argument("--control-backend", choices=["none"] + list(BACKENDS),
                   default="none",
                   help="additionally score the SAME weights through this "
                        "backend, for the throughput ratio and the accuracy "
                        "delta by depth. 'exact' is the full forward of every "
                        "mutant (PLAN.md 22: what the cache costs in accuracy); "
                        "'recompute' is the cache's bruteforce oracle — the same "
                        "approximation, so its delta only checks numerics")
    g.add_argument("--no-ragged", action="store_true",
                   help="cache backend: use the padded shared-position path "
                        "(cost N x union-K) instead of ragged per-variant "
                        "positions (cost = real substitutions)")
    g.add_argument("--ref-summary", choices=["mean", "attn"], default="mean",
                   help="PLAN.md's open question: mean-pool or learned query")
    g.add_argument("--pool", choices=["mean", "sum", "max"], default="mean")
    g.add_argument("--head-hidden", type=int, default=128)
    g.add_argument("--head-layers", type=int, default=2)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--no-ref-residue", action="store_true",
                   help="drop the wild-type residue identity from the variant token")
    g.add_argument("--no-position-embedding", action="store_true")
    g.add_argument("--assay-in-token", action="store_true",
                   help="also add the assay embedding to the variant tokens, "
                        "not just the head")
    g.add_argument("--no-marginals", action="store_true",
                   help="drop the ESM-C masked-marginal additive prior")
    g.add_argument("--marginal-weight", type=float, default=1.0,
                   help="initial weight on the summed masked marginals; with "
                        "zero-init the untrained model IS the zero-shot baseline")
    g.add_argument("--freeze-marginal-weight", action="store_true")
    g.add_argument("--no-zero-init-additive", action="store_true")
    g.add_argument("--position-only-token", choices=["mask", "ref"], default="mask")
    g.add_argument("--random-ref-seed", type=int, default=0)
    g.add_argument("--marginal-mode", choices=["masked", "wt"], default="masked",
                   help="'masked' is one forward per mutated position and is "
                        "the standard; 'wt' is a single forward and is cheaper")

    g = p.add_argument_group("loss")
    g.add_argument("--loss", default="rank",
                   help=f"one of {LOSSES}, or a '+'-joined mix, e.g. rank+mse_rank")
    g.add_argument("--target-col", default="score_rank",
                   help="score_rank for cross-assay training (the link is "
                        "absorbed); score for a single ΔG assay")
    g.add_argument("--se-col", default="score_se")
    g.add_argument("--rank-weight", type=float, default=1.0)
    g.add_argument("--mse-weight", type=float, default=1.0)
    g.add_argument("--rank-sigma", type=float, default=1.0,
                   help="RankNet's σ; 1.0 is ESMRank's (and XGBoost's fixed) value")
    g.add_argument("--rank-margin", type=float, default=0.0)
    g.add_argument("--rank-min-gap", type=float, default=0.0,
                   help="drop pairs whose labels differ by less than this — "
                        "the cheap way to stop fitting assay noise")
    g.add_argument("--pair-weight",
                   choices=["uniform", "gap", "inv_var", "gap_inv_var"],
                   default="uniform")
    g.add_argument("--num-pair-per-sample", type=int, default=0,
                   help="0 = all within-batch pairs; ESMRank samples 20")
    g.add_argument("--listnet-temperature", type=float, default=1.0)
    g.add_argument("--soft-rank-tau", type=float, default=0.05)
    g.add_argument("--inverse-variance", action="store_true",
                   help="inverse-variance weight the ΔG MSE by score_se")

    g = p.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=0,
                   help="0 = evaluate the (zero-shot-initialised) model only")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="only used with --train-backbone")
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    g.add_argument("--warmup", type=int, default=50)
    g.add_argument("--batch-size", type=int, default=32,
                   help="variants per forward. One batch is one score set and "
                        "one reference, so this is also the list length the "
                        "ranking loss sees — too small and the within-set "
                        "ranking signal is mostly ties.")
    g.add_argument("--eval-batch-size", type=int, default=0,
                   help="variants per forward at evaluation (0 = --batch-size). "
                        "Predictions do not depend on batch-mates, so this only "
                        "changes speed; with the ragged cache, large is cheap")
    g.add_argument("--accum", type=int, default=1,
                   help="batches accumulated into one optimizer step")
    g.add_argument("--grad-clip", type=float, default=1.0, help="0 = off")
    g.add_argument("--max-union-positions", type=int, default=0,
                   help="cap K, the union of mutated positions in one forward "
                        "(0 = uncapped). Cost is O(N·K), so on a sparse single-"
                        "mutant score set a large batch pays for positions "
                        "almost every row pads.")

    g = p.add_argument_group("evaluation")
    g.add_argument("--no-select-best", action="store_true",
                   help="keep the final weights instead of restoring the epoch "
                        "that scored best on the eval split (which selects on "
                        "it — fine for val, not for test)")
    g.add_argument("--select-on-val", action="store_true",
                   help="with --eval-split test: score VAL every epoch (logging "
                        "val loss + Spearman), restore the best-on-val epoch, "
                        "and score test once at the end — METL's protocol")
    g.add_argument("--eval-every", type=int, default=0,
                   help="also evaluate every N optimizer steps (0 = epoch ends)")
    g.add_argument("--baselines", nargs="*", default=["zero_shot"],
                   choices=list(BASELINE_CHOICES) + ["all", "none"])
    g.add_argument("--ridge-alpha", type=float, default=1.0)
    g.add_argument("--epsilon-depths", type=int, nargs="*", default=[2, 3],
                   help="Megascale ε is defined for doubles and triples")
    g.add_argument("--allow-nonadditive-epsilon", action="store_true",
                   help="compute ε on fitness-scale assays anyway. Off by "
                        "default because without a monotone link a sigmoid "
                        "masquerades as learned epistasis (DATASETS.md).")
    g.add_argument("--throughput-variants", type=int, default=0,
                   help="time this many variants through each backend "
                        "(0 = skip). Requires --control-backend.")

    g = p.add_argument_group("output")
    g.add_argument("--output", default=None, help="run directory")
    g.add_argument("--out-root", default=None)
    g.add_argument("--tag", default=None, help="extra label in the run id")
    g.add_argument("--save-model", action="store_true")
    g.add_argument("--strict", action="store_true",
                   help="content-hash input artifacts into the run record")


# ── run record (self-contained; mirrors training/common/run_record.py) ────────

def _git_sha() -> tuple[str | None, bool]:
    try:
        repo = Path(__file__).resolve().parents[1]
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse",
                                       "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status",
                                              "--porcelain"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return sha, dirty
    except Exception:
        return None, False


def _artifact_tag(path, *, strict: bool):
    if not path:
        return None
    p = Path(path)
    tag = {"path": str(path)}
    if p.exists():
        st = p.stat()
        tag.update(size=st.st_size, mtime=int(st.st_mtime))
        if strict and p.is_file():
            h = hashlib.sha1()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            tag["sha1"] = h.hexdigest()[:16]
    return tag


def scratch_root() -> Path:
    return Path(os.environ.get("MUTANT_CACHE_SCRATCH",
                               Path.home() / "mutant_cache_scratch"))


def resolve_data_dir(args) -> Path:
    if args.data_dir:
        return Path(args.data_dir)
    return scratch_root() / "processed" / args.dataset


def resolve_output(args) -> Path:
    if args.output:
        d = Path(args.output)
    else:
        root = Path(args.out_root) if args.out_root else scratch_root() / "runs"
        rid = (f"{args.dataset}_{args.ablation}_{args.backend}_{args.loss}"
               f"_e{args.epochs}_lr{args.lr:g}_s{args.seed}")
        if args.tag:
            rid += f"_{args.tag}"
        d = root / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_record(args, run_dir: Path, metrics: dict, extra: dict) -> dict:
    ident = {"dataset": args.dataset, "ablation": args.ablation,
             "backend": args.backend, "backbone": args.backbone,
             "loss": args.loss, "split": args.split, "seed": args.seed,
             "hyperparams": {k: v for k, v in sorted(vars(args).items())
                             if k not in ("output", "out_root", "device",
                                          "save_model", "strict", "tag")}}
    sha, dirty = _git_sha()
    rec = {
        "run_id": hashlib.sha1(json.dumps(ident, sort_keys=True,
                                          default=str).encode()).hexdigest()[:10],
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": sha, "git_dirty": dirty,
        "dataset": args.dataset, "model": f"variant_cache_{args.ablation}",
        "backbone": args.backbone, "backend": args.backend,
        "loss": args.loss, "seed": args.seed,
        "hyperparams": ident["hyperparams"],
        "data": _artifact_tag(resolve_data_dir(args), strict=args.strict),
        "metrics": metrics, "extra": extra,
    }
    (run_dir / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    index = (Path(args.out_root) if args.out_root else scratch_root() / "runs") / "index.jsonl"
    try:
        index.parent.mkdir(parents=True, exist_ok=True)
        with open(index, "a") as f:
            f.write(json.dumps({k: v for k, v in rec.items()
                                if k != "hyperparams"}, default=str) + "\n")
    except Exception:
        pass
    return rec


# ── loading ───────────────────────────────────────────────────────────────────

def load_backbone(args, device, dtype):
    """§1's ``load_esmc``, or the test stub."""
    if args.backbone == "stub":
        import importlib.util
        fx = Path(__file__).resolve().parents[1] / "tests" / "fixtures.py"
        import sys
        spec = importlib.util.spec_from_file_location("mc_test_fixtures", fx)
        mod = importlib.util.module_from_spec(spec)
        # register before exec: @dataclass resolves annotations through
        # sys.modules[cls.__module__] and blows up if the module is absent
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        m = mod.make_stub_backbone(d_model=args.stub_dim, n_layers=args.stub_layers,
                                   seed=args.seed)
        return m.to(device)
    if args.backbone.startswith("random:"):
        # the real ESM-C architecture with random weights: exercises the real
        # cache code path end to end with no checkpoint download. Useful for a
        # plumbing run; the numbers mean nothing.
        from mutant_cache.backbone.esmc import ESMC
        d, h, n = (int(x) for x in args.backbone.split(":")[1:4])
        torch.manual_seed(args.seed)
        return ESMC(d, h, n).to(device=device, dtype=dtype).eval()
    from mutant_cache.backbone import load_esmc
    return load_esmc(args.backbone, device=device, dtype=dtype)


def attach_lora(args, backbone) -> None:
    """Attach the requested adapter sets. Called after the base is frozen and
    the masked-marginal prior is computed (fresh adapters are the identity, so
    the prior is the base model's either way). Adapters are fp32 and trainable;
    they live inside the backbone, so the optimizer adds them explicitly."""
    if not (args.lora_r or args.shared_lora_r):
        return
    from mutant_cache.backbone.esmc import ESMC
    from mutant_cache.cache.lora import (DEFAULT_TARGETS, LoRAConfig,
                                         attach_shared_lora, attach_variant_lora)
    if not isinstance(backbone, ESMC):
        raise ValueError("LoRA needs the ESM-C backbone (not --backbone stub)")

    def cfg(r, targets):
        return LoRAConfig(r=r, alpha=args.lora_alpha or 2.0 * r,
                          targets=tuple(targets or DEFAULT_TARGETS))
    freeze = not args.train_backbone
    if args.lora_r:
        attach_variant_lora(backbone, cfg(args.lora_r, args.lora_targets), freeze_base=freeze)
    if args.shared_lora_r:
        attach_shared_lora(backbone, cfg(args.shared_lora_r, args.shared_lora_targets),
                           freeze_base=freeze)
    for which in ("variant", "shared"):
        ps = _adapter_parameters(backbone, which)
        if ps:
            print(f"  {which} LoRA: {sum(p.numel() for p in ps):,} params (fp32)")


def _adapter_parameters(backbone, which: str = "all") -> list:
    if getattr(backbone, "variant_adapters", None) is None and \
            getattr(backbone, "shared_adapters", None) is None:
        return []
    from mutant_cache.cache.lora import lora_parameters
    return lora_parameters(backbone, which)


def prepare_frames(args) -> dict:
    """Canonical §2 tables, filtered and split."""
    d = resolve_data_dir(args)
    t = load_tables(d)
    v, targets = t["variants"], t.get("targets")
    ss = t.get("score_sets")

    if args.score_sets:
        v = v[v["score_set_id"].isin(args.score_sets)]
    if args.max_n_mut:
        v = v[v["n_mut"] <= args.max_n_mut]
    counts = v.groupby("score_set_id", observed=True)["mutant"].transform("size")
    v = v[counts >= args.min_variants_per_set].reset_index(drop=True)
    if ss is not None and "reliability" in ss.columns:
        v = v.merge(ss[["score_set_id", "reliability"]], on="score_set_id",
                    how="left")

    if args.split:
        sp = load_split(d, args.split)
        train = apply_split(v, sp, "train")
        evl = apply_split(v, sp, args.eval_split)
        if not len(evl):                      # e.g. a split with no val rows
            evl = apply_split(v, sp, "test")
        sel = apply_split(v, sp, "val") if args.select_on_val else None
    else:
        sel = None
        train = v.copy()
        evl = v.copy()

    if args.max_train_variants and len(train) > args.max_train_variants:
        train = train.sample(args.max_train_variants,
                             random_state=args.seed).reset_index(drop=True)

    sequences = {}
    if targets is not None:
        sequences = dict(zip(targets["target_id"], targets["sequence"]))
    return {"all": v, "train": train.reset_index(drop=True),
            "eval": evl.reset_index(drop=True), "sequences": sequences,
            "select": None if sel is None else sel.reset_index(drop=True),
            "score_sets": ss, "dir": d}


def build_config(args, backbone, n_assays: int) -> VariantModelConfig:
    return VariantModelConfig(
        d_model=int(backbone.d_model), n_assays=n_assays,
        vocab_size=int(getattr(backbone, "vocab_size", 0)) or 0,
        ablation=args.ablation, backend=args.backend, ragged=not args.no_ragged,
        use_position_embedding=not args.no_position_embedding,
        use_ref_residue=not args.no_ref_residue,
        assay_in_token=args.assay_in_token,
        pool=args.pool, ref_summary=args.ref_summary,
        head_hidden=args.head_hidden, head_layers=args.head_layers,
        dropout=args.dropout,
        use_marginals=not args.no_marginals,
        marginal_weight_init=args.marginal_weight,
        learn_marginal_weight=not args.freeze_marginal_weight,
        zero_init_additive=not args.no_zero_init_additive,
        position_only_token=args.position_only_token,
        random_ref_seed=args.random_ref_seed)


def build_loss(args) -> VariantLoss:
    return VariantLoss(LossConfig(
        kind=args.loss, rank_weight=args.rank_weight, mse_weight=args.mse_weight,
        sigma=args.rank_sigma, margin=args.rank_margin, min_gap=args.rank_min_gap,
        pair_weight=args.pair_weight, num_pair_per_sample=args.num_pair_per_sample,
        temperature=args.listnet_temperature, tau=args.soft_rank_tau,
        inverse_variance=args.inverse_variance))


def lr_scale(step: int, total: int, schedule: str, warmup: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "cosine":
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(max(prog, 0.0), 1.0)))
    return 1.0


# ── evaluation ────────────────────────────────────────────────────────────────

def score_frame(model, batches, frame: pd.DataFrame, backend: str | None = None,
                col: str = "pred") -> pd.DataFrame:
    idx, pred = model.predict(batches, backend=backend)
    out = frame.copy()
    out[col] = np.nan
    if len(idx):
        out.loc[idx.numpy(), col] = pred.numpy()
    return out


def evaluate(model, batches, frame, args, backend=None) -> dict:
    scored = score_frame(model, batches, frame, backend=backend)
    scored = scored[scored["pred"].notna()]
    table = depth_spearman(scored, "pred", "score")
    return {"depth_table": table, "scored": scored,
            "summary": depth_summary(table), "epsilon": pd.DataFrame()}


def epsilon_eval(model, result: dict, frames, args, seqs, aa_to_id, bkw) -> dict:
    """ε on the evaluation multi-mutants, pulling in their singles.

    The singles usually are *not* in the eval frame: the whole point of the
    depth split is to train on them and test on depth 2+.  But ε is defined
    against them, so they have to be scored too — with the trained model, and
    with their *measured* scores, which is legitimate because ε asks "given
    the singles, is the interaction predicted", not "are the singles held
    out".  They are scored and then dropped; only the multi-mutant rows enter
    the correlation.
    """
    scored = result["scored"]
    multis = scored[scored["n_mut"].isin(args.epsilon_depths)]
    if not len(multis):
        return {"epsilon": pd.DataFrame(), "note": "no multi-mutants in eval"}
    have = set(scored[scored["n_mut"] == 1].index)
    need = frames["all"]
    need = need[(need["n_mut"] == 1)
                & need["score_set_id"].isin(multis["score_set_id"].unique())]
    need = need[~need.index.isin(have)]
    frame = scored
    if len(need):
        b, _ = build_variant_batches(need, seqs, aa_to_id, **bkw)
        dev = next(model.parameters()).device
        singles = score_frame(model, [x.to(dev) for x in b], need)
        frame = pd.concat([scored, singles[singles["pred"].notna()]],
                          ignore_index=True)
        print(f"  ε: pulled in {len(need):,} singles to resolve the "
              f"{len(multis):,} multi-mutants")
    try:
        eps = epsilon_correlation(
            frame, pred_col="pred", depths=tuple(args.epsilon_depths),
            allow_nonadditive=args.allow_nonadditive_epsilon)
    except ValueError as e:
        print(f"  ε skipped: {e}")
        return {"epsilon": pd.DataFrame(), "note": str(e)}
    if len(eps):
        a = eps[eps["score_set_id"] == "ALL"]
        result["summary"].update(
            {f"epsilon_spearman_depth_{r.depth}": float(r.spearman)
             for r in a.itertuples()})
        print("  ε (predicted vs observed interaction): "
              + ", ".join(f"depth {r.depth} ρ={r.spearman:.4f} "
                          f"(≤{r.eps_ceiling:.3f}, n={int(r.n)})"
                          for r in a.itertuples()))
    return {"epsilon": eps}


def throughput(model, batches, backend: str, n_variants: int, repeats: int = 3) -> float:
    """Variants per second through one backend, warm cache, no gradients."""
    model.eval()
    sel, seen = [], 0
    for b in batches:
        sel.append(b)
        seen += b.n
        if seen >= n_variants:
            break
    if not sel:
        return float("nan")
    with torch.no_grad():
        for b in sel[:1]:
            model(b, backend=backend)           # warm the reference cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            for b in sel:
                model(b, backend=backend)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (sum(b.n for b in sel) * repeats) / max(dt, 1e-9)


# ── baselines ─────────────────────────────────────────────────────────────────

def run_baselines(args, frames, backbone, llr, table: ResultsTable) -> dict:
    wanted = set(args.baselines or [])
    if "none" in wanted:
        return {}
    if "all" in wanted:
        wanted = set(BASELINE_CHOICES)
    tr, ev, seqs = frames["train"], frames["eval"], frames["sequences"]
    out: dict = {}

    if "zero_shot" in wanted and llr:
        ev2 = ev.copy()
        ev2["pred"] = score_variants(ev2, llr, aa_to_id=dict(backbone.aa_to_id))
        t = depth_spearman(ev2[ev2["pred"].notna()], "pred", "score")
        out["zero_shot"] = t
        table.add_depth_table("ESM-C masked marginals", t, kind="zero_shot",
                              note="no labels seen — context row, not a comparator")

    if "onehot_ridge" in wanted and seqs:
        m = OneHotRidgePerAssay(seqs, alpha=args.ridge_alpha).fit(
            tr, target_col=args.target_col)
        ev2 = ev.copy()
        ev2["pred"] = m.predict(ev2)
        ev2 = ev2[ev2["pred"].notna()]
        if len(ev2):
            t = depth_spearman(ev2, "pred", "score")
            out["onehot_ridge"] = t
            table.add_depth_table("Ridge on one-hot, per assay", t, kind="baseline",
                                  note="purely additive by construction")

    if wanted & {"embedding_ridge", "embedding_mlp"} and seqs:
        # Frozen mean-pooled embeddings of the *mutant* sequences.  These
        # estimators take a matrix, so swapping in a cached embedding file
        # instead of computing one here is a one-line change.
        def embed(df):
            muts = df["mutant"].fillna("").tolist()
            seqs_ = [_apply_mutant(seqs[t], m)
                     for t, m in zip(df["target_id"], muts)]
            return mean_pooled_embeddings(backbone, seqs_, muts)

        Xtr, Xev = embed(tr), embed(ev)
        if "embedding_ridge" in wanted:
            r = EmbeddingRidge(alpha=args.ridge_alpha).fit(
                Xtr, tr, target_col=args.target_col)
            ev2 = ev.copy()
            ev2["pred"] = r.predict(Xev, ev2)
            ev2 = ev2[ev2["pred"].notna()]
            t = depth_spearman(ev2, "pred", "score")
            out["embedding_ridge"] = t
            table.add_depth_table("Ridge on frozen embeddings", t, kind="baseline")
        if "embedding_mlp" in wanted:
            r = EmbeddingMLP(seed=args.seed).fit(Xtr, tr, target_col=args.target_col)
            ev2 = ev.copy()
            ev2["pred"] = r.predict(Xev, ev2)
            ev2 = ev2[ev2["pred"].notna()]
            t = depth_spearman(ev2, "pred", "score")
            out["embedding_mlp"] = t
            table.add_depth_table("MLP on frozen embeddings", t, kind="baseline")
    return out


def _apply_mutant(seq: str, mutant: str) -> str:
    from mutant_cache.models.data_compat import parse_mutant
    s = list(seq)
    for (p, _ref, alt) in parse_mutant(mutant):
        s[p - 1] = alt
    return "".join(s)


# ── the run ───────────────────────────────────────────────────────────────────

def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    frames = prepare_frames(args)
    tr, ev = frames["train"], frames["eval"]
    if not len(ev):
        print("No evaluation rows — nothing to do.")
        return {"status": "empty"}
    print(f"{frames['dir']}")
    print(f"  train {len(tr):,} variants / "
          f"{tr['score_set_id'].nunique()} score sets  |  "
          f"eval {len(ev):,} / {ev['score_set_id'].nunique()}")
    print(f"  eval depth histogram: "
          f"{dict(ev['n_mut'].value_counts().sort_index())}")

    print(f"\nLoading backbone {args.backbone} on {device} ({dtype}) …")
    backbone = load_backbone(args, device, dtype)
    if not args.train_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)

    used_targets = sorted(set(tr["target_id"]) | set(ev["target_id"]))
    seqs = {t: frames["sequences"][t] for t in used_targets
            if t in frames["sequences"]}
    llr = {}
    if not args.no_marginals or "zero_shot" in (args.baselines or []):
        t0 = time.time()
        llr = build_llr_matrices(backbone, seqs, mode=args.marginal_mode)
        print(f"  masked marginals for {len(llr)} target(s) "
              f"({time.time() - t0:.1f}s)")

    attach_lora(args, backbone)

    assay_index = {s: i for i, s in
                   enumerate(sorted(set(tr["score_set_id"]) | set(ev["score_set_id"])))}
    bkw = dict(assay_index=assay_index, batch_size=args.batch_size,
               tokenizer=getattr(backbone, "tokenizer", None),
               marginals=llr or None, target_col=args.target_col,
               se_col=args.se_col,
               max_union_positions=args.max_union_positions or None)
    tr_batches, _ = build_variant_batches(tr, seqs, dict(backbone.aa_to_id),
                                          shuffle_seed=args.seed, **bkw)
    ev_batches, _ = build_variant_batches(
        ev, seqs, dict(backbone.aa_to_id),
        **{**bkw, "batch_size": args.eval_batch_size or args.batch_size})
    tr_batches = [b.to(device) for b in tr_batches]
    ev_batches = [b.to(device) for b in ev_batches]
    sel = frames["select"]
    if args.select_on_val:
        if args.eval_split != "test" or sel is None or not len(sel):
            raise SystemExit("--select-on-val needs --split with val rows "
                             "and --eval-split test")
        sel_batches, _ = build_variant_batches(
            sel, seqs, dict(backbone.aa_to_id),
            **{**bkw, "batch_size": args.eval_batch_size or args.batch_size})
        sel_batches = [b.to(device) for b in sel_batches]
        print(f"  selecting on val: {len(sel):,} variants")
    print(f"  {len(tr_batches)} train batches, {len(ev_batches)} eval batches; "
          f"mean K = {np.mean([b.k for b in ev_batches]):.1f} positions/forward")

    cfg = build_config(args, backbone, len(assay_index))
    model = VariantEffectModel(backbone, cfg).to(device)
    lossfn = build_loss(args)
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"  ablation={cfg.ablation} backend={cfg.backend} "
          f"interaction={cfg.interaction} cross_attn={cfg.cross_attention} "
          f"| head params {n_head:,}")

    run_dir = resolve_output(args)
    table = ResultsTable(f"{args.dataset} — {args.ablation} / {args.backend}")
    history: list[dict] = []

    base = run_baselines(args, frames, backbone, llr, table)

    zs = evaluate(model, ev_batches, ev, args)
    print(f"\n  at init (= masked-marginal additive prior): "
          f"{_fmt(zs['summary'])}")
    history.append({"tag": "init", "step": 0, **zs["summary"]})

    best = {"step": 0, "spearman_depth_all": zs["summary"].get("spearman_depth_all",
                                                               float("nan")),
            "summary": zs["summary"]}
    final = zs
    if args.epochs:
        sel_arg = (sel_batches, sel) if args.select_on_val else None
        best, final = train(model, lossfn, tr_batches, ev_batches, ev, args,
                            history, run_dir, select=sel_arg)

    if args.epsilon_depths:
        final.update(epsilon_eval(model, final, frames, args, seqs,
                                  dict(backbone.aa_to_id), bkw))

    table.add_depth_table(f"variant cache ({args.ablation})", final["depth_table"],
                          kind="ablation" if args.ablation != "none" else "model",
                          note=f"backend={args.backend}, loss={args.loss}")
    if len(final.get("epsilon", [])):
        a = final["epsilon"][final["epsilon"]["score_set_id"] == "ALL"]
        table.add(f"  ε, variant cache ({args.ablation})",
                  {f"eps_spearman_{r.depth}": float(r.spearman) for r in a.itertuples()},
                  kind="ablation" if args.ablation != "none" else "model",
                  ceiling={f"eps_spearman_{r.depth}": float(r.eps_ceiling)
                           for r in a.itertuples()},
                  note="interaction not explained by the singles")

    control = {}
    if args.control_backend != "none" and args.control_backend != args.backend:
        ctl = evaluate(model, ev_batches, ev, args, backend=args.control_backend)
        delta = _depth_delta(final["depth_table"], ctl["depth_table"])
        table.add_depth_table(f"same weights, {args.control_backend}",
                              ctl["depth_table"], kind="control",
                              note={"exact": "full forward of every mutant",
                                    "recompute": "cache oracle: same approximation",
                                    "cache": "sparse variant cache"}[args.control_backend])
        control = {"summary": ctl["summary"], "accuracy_delta_by_depth": delta}
        print(f"\n  control ({args.control_backend}): {_fmt(ctl['summary'])}")
        print(f"  accuracy delta by depth ({args.backend} - {args.control_backend}): "
              + ", ".join(f"{k}:{v:+.4f}" for k, v in delta.items()))
        if args.throughput_variants:
            a = throughput(model, ev_batches, args.backend, args.throughput_variants)
            b = throughput(model, ev_batches, args.control_backend,
                           args.throughput_variants)
            control.update(variants_per_s_cache=a, variants_per_s_full=b,
                           throughput_ratio=a / b if b else float("nan"))
            table.add_throughput("throughput", variants_per_s_cache=a,
                                 variants_per_s_full=b,
                                 accuracy_delta=delta.get("all"),
                                 note=f"{args.backend} vs {args.control_backend}")
            print(f"  throughput {a:,.0f} vs {b:,.0f} variants/s "
                  f"({a / b if b else float('nan'):.2f}x)")

    # ── output ───────────────────────────────────────────────────────────────
    final["depth_table"].to_csv(run_dir / "depth_spearman.csv", index=False)
    if len(final.get("epsilon", [])):
        final["epsilon"].to_csv(run_dir / "epsilon.csv", index=False)
    final["scored"][["score_set_id", "target_id", "mutant", "n_mut", "score",
                     "pred"]].to_csv(run_dir / "predictions.csv", index=False)
    table.write(run_dir / "results")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(h, default=float) for h in history) + "\n")

    metrics = {args.eval_split: {"mean": {**final["summary"],
                                          **{f"control_{k}": v
                                             for k, v in control.get("summary", {}).items()}}}}
    extra = {"n_train": len(tr), "n_eval": len(ev),
             "n_score_sets": int(ev["score_set_id"].nunique()),
             "n_targets": len(used_targets),
             "mean_union_positions": float(np.mean([b.k for b in ev_batches])),
             "model_config": asdict(cfg), "loss_config": lossfn.cfg.to_dict(),
             "baselines": {k: depth_summary(v) for k, v in base.items()},
             "control": {k: v for k, v in control.items() if k != "summary"},
             "best": {k: v for k, v in best.items() if k != "summary"}}
    rec = write_run_record(args, run_dir, metrics, extra)
    (run_dir / "results.json").write_text(json.dumps(
        {"metrics": metrics, "extra": extra, "history": history},
        indent=2, default=float))
    if args.save_model:
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                    "config": asdict(cfg)}, run_dir / "model.pt")

    print("\n" + table.to_markdown())
    print(f"[{args.dataset}/{args.ablation}] run_id={rec['run_id']}  -> {run_dir}")
    return {"status": "ok", "run_dir": str(run_dir), "metrics": metrics,
            "extra": extra, "table": table, "history": history,
            "final": final, "baselines": base, "control": control, "model": model}


def _fmt(summary: dict) -> str:
    return "  ".join(f"{k.replace('spearman_depth_', 'ρ@')}={v:.4f}"
                     for k, v in summary.items() if v == v)


def _depth_delta(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    aa = a[a["score_set_id"] == "ALL"].set_index("depth")["spearman"]
    bb = b[b["score_set_id"] == "ALL"].set_index("depth")["spearman"]
    return {d: float(aa[d] - bb[d]) for d in aa.index if d in bb.index}


def train(model, lossfn, tr_batches, ev_batches, ev, args, history, run_dir,
          select=None):
    """``select=(batches, frame)``: per-epoch scoring and best-epoch selection
    use that partition (val) instead of the reported one, which is then
    scored once at the end."""
    mon_batches, mon = select if select is not None else (ev_batches, ev)
    # val Spearman is NaN on METL's tiny val sets (2 rows at ds10), so
    # selection on val uses the loss, which is always defined
    crit = "val_loss" if select is not None else "spearman_depth_all"
    head = model.head_parameters()
    groups = [{"params": head, "lr": args.lr}]
    adapters = _adapter_parameters(model.backbone)
    if adapters:
        groups.append({"params": adapters,
                       "lr": args.lora_lr if args.lora_lr is not None else args.lr})
    if args.train_backbone:
        ids = {id(p) for p in adapters}
        groups.append({"params": [p for p in model.backbone.parameters()
                                  if id(p) not in ids],
                       "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(args.seed)
    accum = max(1, args.accum)
    steps_per_epoch = math.ceil(len(tr_batches) / accum)
    total = max(1, args.epochs * steps_per_epoch)
    warmup = min(args.warmup, max(0, total // 10))
    print(f"\n  {len(tr_batches)} batches x {args.epochs} epochs -> {total} "
          f"optimizer steps (accum {accum}, warmup {warmup})")

    best = {crit: float("inf") if crit == "val_loss" else -float("inf"),
            "step": 0, "summary": {}}
    best_state = None
    gstep = 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_batches))
        run_loss, run_n = 0.0, 0
        for si, s in enumerate(range(0, len(order), accum)):
            scale = lr_scale(gstep, total, args.schedule, warmup)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * scale
            opt.zero_grad(set_to_none=True)
            group = [tr_batches[i] for i in order[s:s + accum]]
            for b in group:
                out = model(b)
                # group=None is correct and load-bearing: every batch is one
                # score set by construction (build_variant_batches never
                # crosses one), so the whole batch is one comparable list.
                l, parts = lossfn(out.pred, b.target, group=None, se=b.se,
                                  weight=b.weight)
                (l / len(group)).backward()
                run_loss += float(l.detach())
                run_n += 1
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]],
                    args.grad_clip)
            opt.step()
            gstep += 1
            if args.eval_every and gstep % args.eval_every == 0:
                ev_out = evaluate(model, mon_batches, mon, args)
                row = {"tag": f"step {gstep}", "step": gstep,
                       "train_loss": run_loss / max(run_n, 1), **ev_out["summary"]}
                if select is not None:
                    row["val_loss"] = _eval_loss(model, lossfn, mon_batches)
                history.append(row)
                best, best_state = _maybe_best(model, best, best_state, row, crit)
                print(f"    step {gstep}: loss {row['train_loss']:.4f}  "
                      f"{_fmt(ev_out['summary'])}", flush=True)
                model.train()
        ev_out = evaluate(model, mon_batches, mon, args)
        row = {"tag": f"epoch {epoch}", "step": gstep,
               "train_loss": run_loss / max(run_n, 1), **ev_out["summary"]}
        if select is not None:
            row["val_loss"] = _eval_loss(model, lossfn, mon_batches)
        history.append(row)
        best, best_state = _maybe_best(model, best, best_state, row, crit)
        print(f"  epoch {epoch}: loss {row['train_loss']:.4f}  "
              f"{_fmt(ev_out['summary'])}", flush=True)

    if best_state is not None and not args.no_select_best:
        model.load_state_dict(best_state)
    final = evaluate(model, ev_batches, ev, args)
    print(f"\n  best {crit}={best[crit]:.4f} at step {best['step']}")
    return best, final


@torch.no_grad()
def _eval_loss(model, lossfn, batches) -> float:
    model.eval()
    tot = 0.0
    for b in batches:
        l, _ = lossfn(model(b).pred, b.target, group=None, se=b.se, weight=b.weight)
        tot += float(l)
    return tot / max(len(batches), 1)


def _maybe_best(model, best, best_state, row, crit="spearman_depth_all"):
    v = row.get(crit, float("nan"))
    better = v < best[crit] if crit == "val_loss" else v > best[crit]
    if v == v and better:
        best = {crit: v, "step": row["step"],
                "summary": {k: x for k, x in row.items() if k != "tag"}}
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    return best, best_state


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if run(args).get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

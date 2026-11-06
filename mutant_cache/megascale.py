"""
mutant_cache/megascale.py
-------------------------
Supervised Megascale training with held-out-protein evaluation, for the
ESM-MSR benchmark experiment (TODO_ESM_MSR_EXPERIMENT.md §2-§4).

Arms (same backbone, adapters, head, loss, sampler and schedule):

  S      ``--composition joint``     each variant in one sparse forward
  A      ``--composition additive``  each substitution scored alone, summed;
                                     double labels train it through the sum
  S-add  not trained: every evaluation of a joint model also scores it
         additively (same parameters), saved as ``pred_add``

Protocol pieces that differ from ``harness.py`` and why:

* **Sampling**: proteins uniformly, then variants uniformly within the chosen
  protein (``--micro-batch`` rows per microbatch, ``--proteins-per-step``
  microbatches per optimizer step). One microbatch is one protein, as the
  cache needs one reference per forward. Effective singles/doubles exposure is
  logged.
* **Budget**: ``--exposures`` training-set equivalents, one exposure =
  N_train sampled rows. Cosine schedule with ``--warmup-frac`` warmup; actual
  optimizer steps are recorded.
* **Selection**: full validation once per exposure; the checkpoint with the
  lowest macro-per-protein validation double RMSE, tie-broken by singles macro
  RMSE, then the earliest (step 0, the untrained model, is logged only).
  Fixed in code, not a flag.
* **Test**: scored only with ``--final-test`` (all frozen seeds as one batch
  of final analyses), never during training or selection.
* **Target**: raw ΔΔG (``score``, kcal/mol, WT-relative), MSE; the head is
  zero-anchored so d(WT) = 0 in both arms.

Everything else (backbone, LoRA, optimizer groups, run record) is the
harness's. Run ``python -m mutant_cache.megascale --help``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mutant_cache.eval import megascale as M
from mutant_cache.harness import (
    _sync, add_common_args, attach_lora, build_config, build_loss, build_param_groups,
    describe_param_groups, file_sha1, load_backbone, lora_record, lr_scale,
    resolve_data_dir, resolve_output, scratch_root, write_run_record,
)
from mutant_cache.models import VariantEffectModel, build_variant_batches
from mutant_cache.models.data_compat import apply_split, load_split, load_tables
from mutant_cache.tracking import ExperimentTracker, add_tracking_args
from mutant_cache.diagnostics import TrainingDiagnostics, add_diagnostic_args


# ── data ──────────────────────────────────────────────────────────────────────

def prepare(args) -> dict:
    """Protein-level train / val (/ test with --final-test) frames. Fails
    rather than falling back when a partition is missing or overlaps."""
    d = resolve_data_dir(args)
    t = load_tables(d)
    v = t["variants"]
    if args.max_n_mut:
        v = v[v["n_mut"] <= args.max_n_mut]
    if not args.split:
        raise SystemExit("--split is required (a protein-level train/val/test split)")
    sp = load_split(d, args.split)
    if "target_id" not in sp.columns:
        raise SystemExit(f"split {args.split!r} is not protein-level (no target_id)")
    parts = {k: apply_split(v, sp, k) for k in ("train", "val", "test")}
    if not args.final_test:          # test rows stay unread until configurations are frozen
        parts["test"] = parts["test"].iloc[:0]
    for k in ("train", "val"):
        if not len(parts[k]):
            raise SystemExit(f"split {args.split!r} has no {k!r} rows")
    if args.final_test and not len(parts["test"]):
        raise SystemExit(f"--final-test: split {args.split!r} has no test rows")
    prots = {k: set(f["target_id"]) for k, f in parts.items()}
    prots["test"] = set(sp.loc[sp["split"] == "test", "target_id"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if prots[a] & prots[b]:
            raise SystemExit(f"proteins in both {a} and {b}: {sorted(prots[a] & prots[b])[:5]}")
    if args.max_train_variants and len(parts["train"]) > args.max_train_variants:
        parts["train"] = parts["train"].sample(args.max_train_variants, random_state=args.seed)
    if args.max_val_proteins:
        keep = sorted(prots["val"])[:args.max_val_proteins]
        parts["val"] = parts["val"][parts["val"]["target_id"].isin(keep)]
    if getattr(args, "full_train_refit", False):
        if (not args.final_eval_only or args.patience or args.load_trainable
                or args.max_train_variants or args.max_val_proteins
                or getattr(args, "save_selection_rules", [])):
            raise ValueError("full-train refit requires fresh, uncapped, final-eval-only training without early stopping or selection rules")
        parts["train"] = pd.concat([parts["train"], parts["val"]], ignore_index=True)
        # Retained solely as an explicitly in-sample diagnostic, never a holdout.
        print("FULL TRAIN REFIT: former validation proteins included in training; val/* metrics are IN-SAMPLE diagnostics only", flush=True)
    seqs = dict(zip(t["targets"]["target_id"], t["targets"]["sequence"]))
    split_file = next((c for c in (d / "splits" / f"{args.split}.parquet",
                                   scratch_root() / "processed" / "splits" / f"{args.split}.parquet")
                       if c.is_file()), None)
    if "code_wt" in sp.columns:     # ESM-MSR's protein key: backgrounds collapse to their parent
        grp = dict(zip(sp["target_id"], sp["code_wt"]))
        parts = {k: f.assign(code_wt=f["target_id"].map(grp)) for k, f in parts.items()}
    out = {k: f.reset_index(drop=True) for k, f in parts.items()}
    out.update(sequences=seqs, dir=d, split_file=split_file)
    return out


def describe(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "proteins": int(frame["target_id"].nunique()),
            "unique_variants": int(frame[["target_id", "mutant"]].drop_duplicates().shape[0]),
            **{f"n_mut_{k}": int(v) for k, v in frame["n_mut"].value_counts().sort_index().items()}}


class ProteinUniformSampler:
    """Protein uniformly at random, then ``micro_batch`` of its rows uniformly
    without replacement (all of them if it has fewer)."""

    def __init__(self, frame: pd.DataFrame, micro_batch: int, seed: int,
                 double_ratio: float = 0):
        self.frame = frame
        self.rows = {p: g.to_numpy() for p, g in
                     frame.groupby("target_id", sort=True).groups.items()}
        self.prots = sorted(self.rows)
        self.micro_batch = micro_batch
        self.rng = np.random.default_rng(seed)
        self.double_ratio = double_ratio
        if double_ratio < 0:
            raise ValueError("double sampling ratio must be nonnegative")
        self.pools = {}
        if double_ratio:
            for p, idx in self.rows.items():
                g = frame.loc[idx]
                negative = (g.n_mut == 2) & (g.score < 0)
                self.pools[p] = (idx[~negative], idx[negative],
                                 int(((g.n_mut == 2) & (g.score > 0)).sum()))

    def draw(self) -> pd.DataFrame:
        p = self.prots[int(self.rng.integers(len(self.prots)))]
        r = self.rows[p]
        if self.double_ratio:
            keep, negatives, positives = self.pools[p]
            cap = min(len(negatives), int(self.double_ratio*(positives+1)))
            chosen = negatives if cap == len(negatives) else self.rng.choice(negatives, cap, replace=False)
            r = np.concatenate([keep, chosen])
        idx = r if len(r) <= self.micro_batch else self.rng.choice(r, self.micro_batch,
                                                                   replace=False)
        return self.frame.loc[idx]


# ── model / evaluation helpers ────────────────────────────────────────────────

def trainable_state(model) -> dict:
    return {n: p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state: dict) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state.items():
            named[n].copy_(t.to(named[n].device))
    model.clear_reference_cache()


def score(model, batches, frame, composition=None) -> np.ndarray:
    idx, pred = model.predict(batches, composition=composition)
    out = np.full(len(frame), np.nan)
    if len(idx):
        out[idx.numpy()] = pred.numpy()
    return out


def predictions(model, batches, frame, cfg) -> pd.DataFrame:
    """``pred``: the arm's own composition. For a joint model also ``pred_add``
    (S-add: the same parameters, every variant as the sum of its singles)."""
    out = frame[["target_id", "mutant", "n_mut", "score"]
                + [c for c in ("score_se", "code_wt") if c in frame.columns]].copy()
    out["pred"] = score(model, batches, frame)
    if cfg.composition == "joint":
        out["pred_add"] = score(model, batches, frame, composition="additive")
    return out


def eval_summary(pred: pd.DataFrame, min_n: int, selection: str = "double_rmse",
                 protein_col: str = "target_id") -> dict:
    ev = M.evaluate_frame(pred, "pred", min_n=min_n, protein_col=protein_col)
    s = ev["summary"]
    row = {f"{d}_{m}": M.macro(s, d, m)
           for d in ("single", "double", "all", "epistasis") for m in M.METRICS}
    rs = ev["retrieval_summary"].set_index("subset")
    if "all" in rs.index:
        row["ndcg96_all"] = float(rs.loc["all", "ndcg_macro"])
        row["ndcg_t0_all"] = float(rs.loc["all", "ndcg_t0_macro"])
    from mutant_cache.eval.mutate_everything import detection
    for dep, frame in (("all", pred[pred.n_mut > 0]), ("double", pred[pred.n_mut == 2])):
        rt = detection(frame)
        row[f"ndcg30_{dep}"] = float(rt.ndcg.mean())
        row[f"precision30_{dep}"] = float(rt.precision.mean())
    return {"row": row, "key": M.selection_key(s, selection, retrieval=row), "eval": ev}


def write_eval(ev: dict, run_dir: Path, tag: str) -> None:
    ev["per_protein"].to_csv(run_dir / f"per_protein_{tag}.csv", index=False)
    ev["summary"].to_csv(run_dir / f"summary_{tag}.csv", index=False)
    ev["retrieval"].to_csv(run_dir / f"retrieval_{tag}.csv", index=False)
    ev["retrieval_summary"].to_csv(run_dir / f"retrieval_summary_{tag}.csv", index=False)
    ev["epistasis_table"].to_csv(run_dir / f"epistasis_groups_{tag}.csv", index=False)


# ── training ──────────────────────────────────────────────────────────────────

def run(args) -> dict:
    run_dir = resolve_output(args)
    with ExperimentTracker(args, run_dir) as tracker:
        return _run(args, run_dir, tracker)


def _run(args, run_dir, tracker) -> dict:
    t_run0 = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # An explicit cuda:N does not initialize CUDA via current_device().
        # Initialize the selected device before touching allocator statistics.
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    fr = prepare(args)
    tr, va, te = fr["train"], fr["val"], fr["test"]
    single_exposures = getattr(args, "single_only_exposures", 0)
    if not 0 <= single_exposures <= args.exposures:
        raise ValueError("single-only exposures must lie within the total budget")
    data_desc = {k: describe(f) for k, f in (("train", tr), ("val", va),
                                             ("test", te if args.final_test else te.iloc[:0]))
                 if len(f)}
    for k, v in data_desc.items():
        print(f"  {k}: {v}")

    backbone = load_backbone(args, device, dtype)
    if not args.train_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
    attach_lora(args, backbone)
    used = sorted(set(tr["target_id"]) | set(va["target_id"])
                  | (set(te["target_id"]) if args.final_test else set()))
    seqs = {t: fr["sequences"][t] for t in used}
    augmentor = None
    if getattr(args, "reference_augmentation", "none") != "none":
        from mutant_cache.training_protocol import ReferenceAugmentor
        augmentor = ReferenceAugmentor(tr, seqs, mode=args.reference_augmentation,
            fraction=args.augmentation_fraction, bank_size=args.backgrounds_per_protein, seed=args.seed)
        seqs.update(augmentor.sequences)
    aa = dict(backbone.aa_to_id)
    tok = getattr(backbone, "tokenizer", None)
    bkw = dict(tokenizer=tok, target_col="score", se_col=None,
               assay_index={t: 0 for t in set(seqs) | set(tr.score_set_id) | set(va.score_set_id)
                            | set(te.score_set_id)},
               max_union_positions=args.max_union_positions or None)
    ebs = args.eval_batch_size or args.micro_batch
    va_batches = [b.to(device) for b in build_variant_batches(va, seqs, aa, batch_size=ebs,
                                                               **bkw)[0]]

    cfg = build_config(args, backbone, 1)
    model = VariantEffectModel(backbone, cfg).to(device)
    lossfn = build_loss(args)
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"  head={cfg.head} zero_anchor={cfg.zero_anchor} composition={cfg.composition} "
          f"| head params {n_head:,}")

    if args.load_trainable:
        ck = torch.load(args.load_trainable, map_location="cpu")
        load_trainable(model, ck["trainable_state"])
        print(f"  loaded trainable state from {args.load_trainable}")
    groups = build_param_groups(model, args)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    opt_groups = describe_param_groups(opt.param_groups)
    for g in opt_groups:
        print(f"  optimizer group {g['name']}: {g['n_params']:,} params lr {g['lr']:g}")

    rows_per_step = args.micro_batch * args.proteins_per_step
    steps_per_exposure = max(1, math.ceil(len(tr) / rows_per_step))
    total = args.exposures * steps_per_exposure
    warmup = math.ceil(args.warmup_frac * total)
    single_steps = round(single_exposures*steps_per_exposure)
    print(f"\n  {len(tr):,} train rows / {rows_per_step} rows per step -> "
          f"{steps_per_exposure} steps per exposure x {args.exposures} = {total} steps "
          f"(warmup {warmup})")
    eval_every = max(1, steps_per_exposure // args.evals_per_exposure)
    if args.patience:
        print(f"  validation every {eval_every} steps; early stop after {args.patience} "
              f"validations without improvement")
    sampler = ProteinUniformSampler(tr, args.micro_batch, args.seed,
                                    getattr(args, "double_subsample_ratio", 0))
    single_sampler = None
    if single_steps:
        singles = tr[tr.n_mut == 1]
        if singles.empty:
            raise ValueError("singles curriculum requires measured training singles")
        single_sampler = ProteinUniformSampler(singles, args.micro_batch, args.seed)
    protocol = {k: getattr(args, k, default) for k, default in dict(
        huber_delta=1., single_weight=1., double_weight=1., stabilizing_double_weight=1.,
        task_reduction="rows", double_subsample_ratio=0., single_only_exposures=0.,
        reference_augmentation="none", augmentation_fraction=.25, backgrounds_per_protein=4).items()}
    ranking_weight = getattr(args, 'stability_rank_weight', 0.)
    if ranking_weight < 0:
        raise ValueError('ranking weight must be nonnegative')
    protocol['stability_rank_weight'] = ranking_weight
    weighted_loss = ranking_weight > 0 or args.loss == "huber_dg" or any((protocol["single_weight"] != 1.,
        protocol["double_weight"] != 1., protocol["stabilizing_double_weight"] != 1.,
        protocol["task_reduction"] != "rows"))
    if weighted_loss and args.loss not in ("mse_dg", "huber_dg"):
        raise ValueError("stability weighting requires MSE or Huber on raw delta-delta-G")
    if weighted_loss and lossfn.cfg.inverse_variance:
        raise ValueError("inverse-variance and task-weighted stability losses cannot be combined")
    _sync()
    t_prep = time.perf_counter() - t_run0

    if augmentor is not None:
        augmentor.audit.to_csv(run_dir / "training_reference_bank.csv", index=False)
    (run_dir / "training_protocol.json").write_text(json.dumps(protocol, indent=2))
    diagnostics = TrainingDiagnostics(args, run_dir, tracker)
    diagnostics.prepare(model, tr, seqs, aa, bkw, ebs, device, lossfn, opt.param_groups)
    diagnostics.total_steps = total
    augmented_ids = set() if augmentor is None else set(augmentor.sequences)
    history, best, best_state = [], None, None
    selection_bests = {}
    tracked_rules = list(dict.fromkeys([args.selection] + list(getattr(args, "save_selection_rules", []))))
    exposure_mut = {}
    t_train = t_eval = 0.0
    seen = 0

    def do_eval(step, exposure, train_loss):
        nonlocal best, best_state, t_eval
        _sync()
        t0 = time.perf_counter()
        model.eval()
        pv = predictions(model, va_batches, va, cfg)
        es = eval_summary(pv, args.eval_min_n, args.selection, args.eval_protein_col)
        row = {"step": step, "exposure": exposure, "train_loss": train_loss,
               "lr_scale": lr_scale(max(step - 1, 0), total, args.schedule, warmup),
               **{f"val_{k}": v for k, v in es["row"].items()}}
        if "pred_add" in pv:
            add = eval_summary(pv.assign(pred=pv["pred_add"]), args.eval_min_n,
                               protein_col=args.eval_protein_col)
            row.update({f"val_add_{k}": v for k, v in add["row"].items()})
        history.append(row)
        # lowest key; strict "<" keeps the earliest checkpoint on a tie. The
        # untrained model (step 0) is logged but never selectable.
        if step > 0:
            state = None
            for rule in tracked_rules:
                key = M.selection_key(es["eval"]["summary"], rule, retrieval=es["row"])
                old = selection_bests.get(rule)
                if (rule == args.selection or math.isfinite(key[0])) and (old is None or key < tuple(old["key"])):
                    if state is None:
                        state = trainable_state(model)
                    selected = dict(key=key, step=step, exposure=exposure)
                    selection_bests[rule] = selected
                    if rule == args.selection:
                        best, best_state = selected, state
                    if args.save_model:
                        checkpoint_dir = run_dir / "checkpoints"
                        checkpoint_dir.mkdir(exist_ok=True)
                        path = checkpoint_dir / f"{rule}.pt"
                        temporary = path.with_suffix(".pt.tmp")
                        torch.save(dict(trainable_state=state,
                            args={**vars(args), "selection": rule}, config=asdict(cfg), selection=selected), temporary)
                        temporary.replace(path)
        pd.DataFrame(history).to_csv(run_dir / "history.csv.tmp", index=False)
        (run_dir / "history.csv.tmp").replace(run_dir / "history.csv")
        (run_dir / "selection_checkpoints.json").write_text(json.dumps(selection_bests, indent=2))
        tracker.validation(row)
        tracker.selections(selection_bests)
        diagnostics.evaluate(step, exposure, pv, es["eval"])
        model.train()
        _sync()
        t_eval += time.perf_counter() - t0
        print(f"    exposure {exposure} step {step}: train {train_loss:.4f}  "
              f"val double rmse {row['val_double_rmse']:.4f} rho {row['val_double_spearman']:.4f}"
              f"  single rmse {row['val_single_rmse']:.4f} rho {row['val_single_spearman']:.4f}"
              f"  eps rmse {row['val_epistasis_rmse']:.4f}", flush=True)

    if not args.final_eval_only:
        do_eval(0, 0, float("nan"))
    model.train()
    run_loss, run_n = 0.0, 0
    stopped_at = None
    _sync()
    t0 = time.perf_counter()
    for step in range(total):
        diagnostics.begin_step(step+1, total)
        s = lr_scale(step, total, args.schedule, warmup)
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * s
        opt.zero_grad(set_to_none=True)
        for _ in range(args.proteins_per_step):
            mb = (single_sampler if step < single_steps else sampler).draw()
            if augmentor is not None and step >= single_steps:
                mb = augmentor.apply(mb)
            diagnostics.record_batch(mb, augmented_ids)
            for k, n in mb["n_mut"].value_counts().items():
                exposure_mut[int(k)] = exposure_mut.get(int(k), 0) + int(n)
            bs, _ = build_variant_batches(mb, seqs, aa, batch_size=len(mb), **bkw)
            collected_pred, collected_target, collected_depth = [], [], []
            for b in bs:
                b = b.to(device)
                out = model(b)
                if weighted_loss:
                    collected_pred.append(out.pred)
                    collected_target.append(b.target)
                    collected_depth.append(b.n_mut)
                else:
                    l, _ = lossfn(out.pred, b.target, group=None, se=None)
                    (l * b.n / len(mb) / args.proteins_per_step).backward()
                    run_loss += float(l.detach()) * b.n
                    run_n += b.n
            if weighted_loss:
                from mutant_cache.training_protocol import stability_loss
                l = lossfn.cfg.mse_weight * stability_loss(torch.cat(collected_pred), torch.cat(collected_target),
                    torch.cat(collected_depth), kind=args.loss, delta=protocol["huber_delta"],
                    single_weight=protocol["single_weight"], double_weight=protocol["double_weight"],
                    positive_weight=protocol["stabilizing_double_weight"], reduction=protocol["task_reduction"])
                if ranking_weight:
                    from mutant_cache.models.loss import pairwise_rank_loss
                    if mb.target_id.nunique() != 1 or mb.score_set_id.nunique() != 1:
                        raise ValueError('stability ranking microbatches must have one protein/assay')
                    ranks = []
                    depths = torch.cat(collected_depth)
                    pred, target = torch.cat(collected_pred), torch.cat(collected_target)
                    for depth in depths.unique():
                        mask = depths == depth
                        if mask.sum() > 1:
                            ranks.append(pairwise_rank_loss(pred[mask], target[mask]))
                    if ranks:
                        l = l + ranking_weight * torch.stack(ranks).mean()
                (l/args.proteins_per_step).backward()
                run_loss += float(l.detach())*len(mb)
                run_n += len(mb)
            seen += len(mb)
        diagnostics.before_clip(opt.param_groups)
        if args.grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                                       args.grad_clip)
            diagnostics.observe_clip(grad_norm)
        opt.step()
        diagnostics.after_step(opt.param_groups, step+1)
        if tracker.interval and ((step+1) % tracker.interval == 0 or step+1 == total):
            tracker.progress(step+1, total, seen, run_loss/max(run_n, 1),
                "singles" if step < single_steps else "mixed", opt.param_groups)
        log_every = getattr(args, "log_every", 0)
        if log_every and (step+1) % log_every == 0:
            progress = dict(step=step+1, total_steps=total, rows_seen=seen,
                mean_training_loss=run_loss/max(run_n, 1), phase="singles" if step < single_steps else "mixed")
            (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))
            print(f"    progress {progress}", flush=True)
        if ((not args.final_eval_only and (step + 1) % eval_every == 0)
                or step + 1 == total):
            _sync()
            t_train += time.perf_counter() - t0
            do_eval(step + 1, round((step + 1) / steps_per_exposure, 3),
                    run_loss / max(run_n, 1))
            run_loss, run_n = 0.0, 0
            if (args.patience and best is not None and step + 1 < total and step+1 > single_steps
                    and sum(h["step"] > max(best["step"], single_steps) for h in history) >= args.patience):
                stopped_at = step + 1
                print(f"  early stop at step {stopped_at}: no improvement in "
                      f"{args.patience} validations since step {best['step']}", flush=True)
                break
            _sync()
            t0 = time.perf_counter()

    if best_state is not None:
        load_trainable(model, best_state)
    elif total:
        raise RuntimeError("no checkpoint was selected")
    model.eval()

    # ── final scoring of the selected checkpoint ──────────────────────────────
    _sync()
    t0 = time.perf_counter()
    pv = predictions(model, va_batches, va, cfg)
    val_eval = M.evaluate_frame(pv, "pred", min_n=args.eval_min_n,
                                protein_col=args.eval_protein_col)
    diagnostics.selected_checkpoint(pv, val_eval, stopped_at or total)
    t_val = time.perf_counter() - t0
    pv.assign(split="val", seed=args.seed).to_csv(run_dir / "predictions_val.csv", index=False)
    write_eval(val_eval, run_dir, "val")
    if "pred_add" in pv:
        write_eval(M.evaluate_frame(pv.assign(pred=pv["pred_add"]), "pred",
                                    min_n=args.eval_min_n, protein_col=args.eval_protein_col),
                   run_dir, "val_sadd")
    test_summary, t_test = None, None
    if args.final_test:
        te_batches = [b.to(device) for b in build_variant_batches(te, seqs, aa, batch_size=ebs,
                                                                   **bkw)[0]]
        if getattr(args, "benchmark_inference", False):
            from mutant_cache.comparison import benchmark_inference
            measured = benchmark_inference(model, te_batches, te)
            (run_dir / "inference_cost.json").write_text(json.dumps(measured, indent=2))
        _sync()
        t0 = time.perf_counter()
        pt = predictions(model, te_batches, te, cfg)
        _sync()
        t_test = time.perf_counter() - t0
        test_eval = M.evaluate_frame(pt, "pred", min_n=args.eval_min_n,
                                     protein_col=args.eval_protein_col)
        pt.assign(split="test", seed=args.seed).to_csv(run_dir / "predictions_test.csv",
                                                       index=False)
        write_eval(test_eval, run_dir, "test")
        if "pred_add" in pt:
            write_eval(M.evaluate_frame(pt.assign(pred=pt["pred_add"]), "pred",
                                        min_n=args.eval_min_n,
                                        protein_col=args.eval_protein_col),
                       run_dir, "test_sadd")
        test_summary = {"summary": test_eval["summary"].to_dict("records"),
                        "retrieval": test_eval["retrieval_summary"].to_dict("records")}
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    _sync()
    timing = {"prep_s": t_prep, "train_s": t_train, "train_eval_s": t_eval,
              "final_val_s": t_val, "final_test_s": t_test, "total_steps": total, "stopped_at_step": stopped_at,
              "steps_per_exposure": steps_per_exposure, "warmup": warmup,
              "rows_per_step": rows_per_step, "train_rows_seen": seen,
              "steady_train_rows_per_s": seen / t_train if t_train else float("nan"),
              "cold_total_s": time.perf_counter() - t_run0}
    memory = {"ref_cache_mb": model.reference_cache_nbytes() / 2 ** 20}
    if device.type == "cuda":
        memory.update(peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30,
                      peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2 ** 30,
                      gpu=torch.cuda.get_device_name(device))
    n_active = sum(g["n_params"] for g in opt_groups)
    extra = {"arm": {"joint": "S", "additive": "A"}[cfg.composition],
             "training_protocol": {**protocol, "single_only_steps": single_steps,
                 "augmented_rows": 0 if augmentor is None else augmentor.transformed_rows},
             "selection_checkpoints": selection_bests,
             "data": data_desc, "exposure_by_n_mut": exposure_mut,
             "split_file": None if fr["split_file"] is None else str(fr["split_file"]),
             "split_sha1": None if fr["split_file"] is None else file_sha1(fr["split_file"]),
             "variants_sha1": file_sha1(fr["dir"] / "variants.parquet"),
             "eval": {"protein_col": args.eval_protein_col, "min_n": args.eval_min_n},
             "selection": {"rule": "fixed_final" if args.final_eval_only else args.selection,
                           "criterion": "predeclared final update" if args.final_eval_only else M.SELECTION_RULES[args.selection], **(best or {})},
             "validation_role": "in_sample_former_validation" if getattr(args, "full_train_refit", False) else "held_out",
             "full_train_refit": getattr(args, "full_train_refit", False),
             "head": cfg.head, "active_head_params": n_head, "active_params": n_active,
             "optimizer_groups": opt_groups, "lora": lora_record(backbone),
             "precision": args.precision, "timing": timing, "memory": memory,
             "model_config": asdict(cfg), "loss_config": lossfn.cfg.to_dict(),
             "val_summary": val_eval["summary"].to_dict("records"),
             "val_retrieval": val_eval["retrieval_summary"].to_dict("records"),
             "test_summary": test_summary}
    metrics = {"val": {k: M.macro(val_eval["summary"], *k.split("_"))
                       for k in ("double_rmse", "single_rmse", "double_spearman",
                                 "single_spearman", "epistasis_rmse")}}
    write_run_record(args, run_dir, metrics, extra)
    (run_dir / "results.json").write_text(json.dumps(
        {"metrics": metrics, "extra": extra, "history": history}, indent=2, default=float))
    tracker.complete(extra)
    diagnostics.finish()
    if args.save_model:
        torch.save({"trainable_state": trainable_state(model), "args": vars(args),
                    "config": asdict(cfg)}, run_dir / "model.pt")
    print(f"\n  selected exposure {best and best['exposure']} (step {best and best['step']}); "
          f"val {metrics['val']}\n  -> {run_dir}")
    return {"status": "ok", "run_dir": str(run_dir), "metrics": metrics, "extra": extra,
            "history": history, "model": model}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_tracking_args(p)
    add_diagnostic_args(p)
    g = p.add_argument_group("megascale")
    g.add_argument("--exposures", type=int, default=10,
                   help="training budget in training-set equivalents (N_train sampled rows)")
    g.add_argument("--micro-batch", type=int, default=64,
                   help="rows per microbatch; one microbatch = one protein")
    g.add_argument("--proteins-per-step", type=int, default=4,
                   help="microbatches (proteins) per optimizer step")
    g.add_argument("--warmup-frac", type=float, default=0.05)
    g.add_argument("--eval-min-n", type=int, default=M.MIN_N,
                   help="per-protein rows needed for a metric to be defined")
    g.add_argument("--eval-protein-col", choices=("target_id", "code_wt"), default="target_id",
                   help="per-protein grouping for metrics; code_wt = ESM-MSR's (engineered "
                        "backgrounds pooled with their parent domain)")
    g.add_argument("--selection", choices=sorted(M.SELECTION_RULES), default="double_rmse",
                   help="checkpoint-selection rule on validation (TODO_ESM_MSR §5.5); "
                        "spearman = macro all-depth Spearman, ESM-MSR's early-stopping metric")
    g.add_argument("--final-eval-only", action="store_true",
                   help="validate once, after the last step (no checkpoint selection); "
                        "for runs where validation dominates the cost, e.g. --backend exact")
    g.add_argument("--full-train-refit", action="store_true",
                   help="merge training and internal validation proteins for a fresh fixed-budget final fit; val metrics become in-sample diagnostics")
    g.add_argument("--evals-per-exposure", type=int, default=1,
                   help="validations per exposure (checkpoint-selection resolution)")
    g.add_argument("--patience", type=int, default=0,
                   help="stop after this many validations without a new best (0 = off); "
                        "the LR schedule still spans --exposures")
    g.add_argument("--max-val-proteins", type=int, default=0,
                   help="smoke tests only: score the first N val proteins")
    g.add_argument("--load-trainable", default=None,
                   help="model.pt from a previous run (--save-model): load its adapters "
                        "and head; with --exposures 0 the run only scores it")
    g.add_argument("--final-test", action="store_true",
                   help="score the test partition once, after selection. Only for "
                        "frozen configurations")
    g.add_argument("--benchmark-inference", action="store_true",
                   help="measure synchronized cold/warm scoring of the final test library")
    g.add_argument("--huber-delta", type=float, default=1.)
    g.add_argument("--stability-rank-weight", type=float, default=0.,
                   help="auxiliary within-protein, within-depth RankNet loss weight on raw stability MSE/Huber")
    g.add_argument("--single-weight", type=float, default=1.)
    g.add_argument("--double-weight", type=float, default=1.)
    g.add_argument("--stabilizing-double-weight", type=float, default=1.)
    g.add_argument("--task-reduction", choices=("rows", "depth"), default="rows")
    g.add_argument("--double-subsample-ratio", type=float, default=0.)
    g.add_argument("--single-only-exposures", type=float, default=0.)
    g.add_argument("--reference-augmentation", choices=("none", "reverse", "background"), default="none")
    g.add_argument("--augmentation-fraction", type=float, default=.25)
    g.add_argument("--backgrounds-per-protein", type=int, default=4)
    g.add_argument("--save-selection-rules", nargs="*", choices=sorted(M.SELECTION_RULES), default=[])
    g.add_argument("--log-every", type=int, default=0)
    # TODO_ESM_MSR §3 starting point; every one is still a flag
    p.set_defaults(dataset="esm_msr_mega", split="esm_msr_hyperopt", minimal=True,
                   head="sum_mean_count", zero_anchor=True, loss="mse_dg",
                   target_col="score", se_col=None, no_marginals=True, baselines=[],
                   lora_r=8, lora_alpha=16.0, lora_lr=1e-3, lr=3e-3)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if run(args).get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

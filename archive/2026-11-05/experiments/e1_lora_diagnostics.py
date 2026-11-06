# e1_lora_diagnostics.py -- TODO.md 2026-11-01, E1: LoRA correctness / numerical diagnostics
# on the real ESM-C 300M. One tiny deterministic avGFP batch (16 singles + 32 multis from
# metl_avgfp_dms_ds1280_r0 TRAIN), original minimal mean head, uniform RankNet on raw score,
# r=8 alpha=16 default targets (q,o,gate,up,down), AdamW (head 3e-3, adapters 3e-3, wd 0.01),
# grad clip 1, CONSTANT lr (no schedule), 20 optimizer steps. Arms: variant-only, shared-only,
# both; plus full-backbone (lr 1e-5) as a reference. Precisions: fp32; bf16 base + fp32
# adapters (default 'activation' arithmetic); bf16 base + fp32 adapters + fp32 arithmetic.
# Logs per step: loss, grad norm / last update / distance-from-init per optimizer group,
# lora_A grad norm, reference recomputations, time, peak memory. At steps 0/1/10/20:
# per-projection ||delta|| / ||base output|| on the variant stream, and ragged (cache) vs
# clamped-recompute oracle agreement of predictions AND per-group gradients.
# Checks: frozen base bit-identical after training; eval after steps == fresh reference.
# Raw script, as run. Output: experiments/e1_lora_diagnostics{.json,_steps.csv,_probes.csv}
import json, time, copy
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from mutant_cache import harness
from mutant_cache.backbone import load_esmc
from mutant_cache.cache import LoRAConfig, attach_shared_lora, attach_variant_lora, lora_parameters
from mutant_cache.cache import variant_cache as vc
from mutant_cache.models import VariantEffectModel, VariantModelConfig, build_variant_batches
from mutant_cache.models.data_compat import apply_split, load_split, load_tables

DEV = "cuda"
S = "/home/andrew/mutant_cache_scratch/processed"
OUT = "experiments/e1_lora_diagnostics"
STEPS, PROBE = 20, (0, 1, 10, 20)

t = load_tables(f"{S}/metl_avgfp_dms")
v, seq = t["variants"], t["targets"]["sequence"][0]
tid = t["targets"]["target_id"][0]
tr = apply_split(v, load_split(f"{S}/metl_avgfp_dms", "metl_avgfp_dms_ds1280_r0"), "train")
tr = tr.sort_values("mutant")
df = pd.concat([tr[tr.n_mut == 1].head(16), tr[tr.n_mut >= 2].head(32)]).reset_index(drop=True)
print(f"batch: {len(df)} variants, depth histogram {df.n_mut.value_counts().sort_index().to_dict()}")

args = harness.build_parser().parse_args(
    ["--dataset", "metl_avgfp_dms", "--minimal", "--loss", "rank", "--target-col", "score",
     "--lr", "3e-3", "--lora-lr", "3e-3", "--backbone-lr", "1e-5", "--weight-decay", "0.01"])
lossfn = harness.build_loss(args)

# --- instrumentation: per-projection base / delta norms on the variant stream ----------
REC = {"on": False, "name": None, "base": defaultdict(float), "delta": defaultdict(float)}
_orig_delta, _orig_add = vc._delta, vc._add


def _delta(ad, name, x, rowmask):
    REC["name"] = name
    return _orig_delta(ad, name, x, rowmask)


def _add(base, dlt):
    if REC["on"] and dlt is not None:
        REC["base"][REC["name"]] += float(base.float().square().sum())
    return _orig_add(base, dlt)


vc._delta, vc._add = _delta, _add


def hook_set(bb):
    """forward hooks on every LoRADelta: summed squared delta per (set, projection)."""
    hs = []
    for which in ("variant", "shared"):
        ad = getattr(bb, f"{which}_adapters", None)
        if ad is None:
            continue
        for la in ad:
            for proj, mod in la.deltas.items():
                def h(m, i, o, key=(which, proj)):
                    if REC["on"]:
                        REC["delta"][key] += float(o.float().square().sum())
                hs.append(mod.register_forward_hook(h))
    return hs


def probe_ratios(model, b):
    REC["base"].clear(); REC["delta"].clear()
    with torch.no_grad():
        b.ref = model.backbone.forward_reference(b.ref_ids)   # reference outside the probe
        REC["on"] = True
        model(b)
        REC["on"] = False
        b.ref = None
    return {f"ratio_{w}_{p}": (REC["delta"][(w, p)] / REC["base"][p]) ** 0.5
            for (w, p) in REC["delta"] if REC["base"][p] > 0}


def oracle_agreement(model, b, groups):
    """cache (ragged) vs recompute (clamped oracle): preds and grads, same weights."""
    res = {}
    for be in ("cache", "recompute"):
        model.zero_grad(set_to_none=True)
        model.train()
        pred = model(b, backend=be).pred
        l, _ = lossfn(pred, b.target, group=None)
        l.backward()
        res[be] = (pred.detach().float(),
                   {g["name"]: [None if p.grad is None else p.grad.detach().float().clone()
                                for p in g["params"]] for g in groups})
    model.zero_grad(set_to_none=True)
    (pc, gc), (pr, gr) = res["cache"], res["recompute"]
    out = {"pred_maxabs_diff": float((pc - pr).abs().max()), "pred_scale": float(pr.abs().max())}
    for name in gc:
        num = sum(float((a - c).square().sum()) for a, c in zip(gc[name], gr[name]) if a is not None)
        den = sum(float(c.square().sum()) for c in gr[name] if c is not None)
        out[f"grad_relerr_{name}"] = (num / den) ** 0.5 if den > 0 else float("nan")
    return out


def run(arm, precision, compute):
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    bb = load_esmc("esmc_300m", device=DEV, dtype=dtype)
    full = arm == "full"
    bb.freeze_base(not full)
    torch.manual_seed(0)
    cfg = LoRAConfig(r=8, alpha=16.0, compute=compute)
    if arm in ("variant", "both"):
        attach_variant_lora(bb, cfg, freeze_base=not full)
    if arm in ("shared", "both"):
        attach_shared_lora(bb, copy.deepcopy(cfg), freeze_base=not full)
    a = copy.copy(args); a.train_backbone = full
    torch.manual_seed(0)
    model = VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True)).to(DEV)
    (b,), _ = build_variant_batches(df, {tid: seq}, dict(bb.aa_to_id), tokenizer=bb.tokenizer,
                                    batch_size=len(df), target_col="score", se_col=None)
    b = b.to(DEV)
    groups = harness.build_param_groups(model, a)
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    tracker = harness.UpdateTracker(opt.param_groups)
    base0 = None if full else {n: p.detach().cpu().clone() for n, p in bb.base_parameters()}
    n_ref = {"calls": 0}
    orig_fr = bb.forward_reference

    def counted(*x, **k):
        n_ref["calls"] += 1
        return orig_fr(*x, **k)
    bb.forward_reference = counted
    hooks = hook_set(bb)
    meta = {"arm": arm, "precision": precision, "compute": compute,
            "groups": harness.describe_param_groups(opt.param_groups),
            "adapter_dtypes": sorted({str(p.dtype) for p in lora_parameters(bb)}) if arm != "full" else [],
            "reference_moves": model._reference_moves()}
    rows, probes = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(STEPS + 1):
        if step in PROBE:
            pr = {"step": step, **probe_ratios(model, b), **oracle_agreement(model, b, opt.param_groups)}
            for which in ("variant", "shared"):
                m = getattr(bb, f"{which}_adapters", None)
                if m is not None:
                    pr[f"normA_{which}"] = harness._norm(p for n, p in m.named_parameters() if "lora_A" in n)
                    pr[f"normB_{which}"] = harness._norm(p for n, p in m.named_parameters() if "lora_B" in n)
            probes.append(pr)
        if step == STEPS:
            break
        model.train()
        tracker.snapshot()                       # pre-step params (CPU), outside the timer
        torch.cuda.synchronize(); t0 = time.perf_counter()
        calls0 = n_ref["calls"]
        opt.zero_grad(set_to_none=True)
        pred = model(b).pred
        l, _ = lossfn(pred, b.target, group=None)
        l.backward()
        gA = {}
        for which in ("variant", "shared"):
            m = getattr(bb, f"{which}_adapters", None)
            if m is not None:
                gA[f"gradA_{which}"] = harness._norm(p.grad for n, p in m.named_parameters()
                                                     if "lora_A" in n and p.grad is not None)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append({"step": step + 1, "loss": float(l), "time_s": dt,
                     "ref_recomputed": n_ref["calls"] - calls0, **gA, **tracker.row()})
    meta["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    meta["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
    if base0 is not None:
        meta["frozen_base_bit_identical"] = all(torch.equal(base0[n], p.detach().cpu())
                                                for n, p in bb.base_parameters())
    model.eval()
    with torch.no_grad():
        p1 = model(b).pred.float()
        model.clear_reference_cache()
        p2 = model(b).pred.float()
        meta["final_pred"] = p2.cpu().tolist()
    meta["eval_vs_fresh_reference_maxabs"] = float((p1 - p2).abs().max())
    for h in hooks:
        h.remove()
    del model, bb, opt, tracker
    torch.cuda.empty_cache()
    return meta, rows, probes


CONFIGS = ([(arm, "fp32", "activation") for arm in ("variant", "shared", "both", "full")]
           + [(arm, "bf16", "activation") for arm in ("variant", "shared", "both")]
           + [(arm, "bf16", "fp32") for arm in ("variant", "shared", "both")])
results, steprows, proberows = [], [], []
for arm, prec, comp in CONFIGS:
    t0 = time.perf_counter()
    meta, rows, probes = run(arm, prec, comp)
    tag = f"{arm}/{prec}/{comp}"
    meta["wall_s"] = time.perf_counter() - t0
    results.append(meta)
    steprows += [{"config": tag, **r} for r in rows]
    proberows += [{"config": tag, **r} for r in probes]
    print(f"{tag:24s} loss {rows[0]['loss']:.4f} -> {rows[-1]['loss']:.4f}  "
          f"ref recomputed/step {np.mean([r['ref_recomputed'] for r in rows]):.1f}  "
          f"frozen ok {meta.get('frozen_base_bit_identical')}  "
          f"oracle pred diff (step 20) {probes[-1]['pred_maxabs_diff']:.2e}  "
          f"peak {meta['peak_allocated_gb']:.2f} GB  "
          f"{np.mean([r['time_s'] for r in rows[1:]]) * 1e3:.0f} ms/step", flush=True)
json.dump({"configs": results, "steps": steprows, "probes": proberows},
          open(f"{OUT}.json", "w"), indent=1, default=float)
pd.DataFrame(steprows).to_csv(f"{OUT}_steps.csv", index=False)
pd.DataFrame(proberows).to_csv(f"{OUT}_probes.csv", index=False)

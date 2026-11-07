"""Bounded, deterministic diagnostics, independent of training/selection RNGs."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from mutant_cache.eval import megascale as M


DEFAULTS = dict(diagnostics=False, diagnostic_log_every=50, diagnostic_eval_every=1,
    diagnostic_profile_every=5, diagnostic_probe_proteins=8, diagnostic_probe_rows=128,
    diagnostic_plot_rows=512, diagnostic_activation_rows=32, diagnostic_activation_proteins=2,
    diagnostic_parameter_samples=8192, diagnostic_seed=0)


def add_diagnostic_args(parser):
    group = parser.add_argument_group("sampled diagnostics")
    group.add_argument("--diagnostics", action="store_true", help="enable bounded local/W&B diagnostics")
    for name, default in DEFAULTS.items():
        if name != "diagnostics":
            group.add_argument("--" + name.replace("_", "-"), type=int, default=default)


def balanced_rows(frame, limit, columns, seed):
    """Round-robin strata, with a private RNG and stable input ordering."""
    if len(frame) <= limit:
        return frame.copy()
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(g.index).tolist() for _, g in frame.groupby(columns, sort=True, observed=True)]
    selected = []
    while pools and len(selected) < limit:
        for pool in pools:
            selected.append(pool.pop())
            if len(selected) == limit:
                break
        pools = [p for p in pools if p]
    return frame.loc[selected].copy()


def training_probe(frame, proteins=8, rows=128, seed=0):
    """Measured TRAIN rows only. Reserve room for complete A/B/AB triples.

    Then fill by mutation depth and score >= 0. The manifest records the
    resulting, deliberately stratified probe; it is not a full-train estimate.
    """
    frame = frame.sort_values(["target_id", "score_set_id", "mutant"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    targets = sorted(frame.target_id.unique())
    targets = sorted(rng.choice(targets, min(proteins, len(targets)), replace=False).tolist())
    selected = []
    for tid in targets:
        group = frame[frame.target_id == tid]
        singles = {(r.score_set_id, r.mutant): r.Index for r in group[group.n_mut == 1].itertuples()}
        triples = []
        for r in group[group.n_mut == 2].itertuples():
            parts = r.mutant.split(":")
            keys = [(r.score_set_id, m) for m in parts]
            if len(keys) == 2 and all(k in singles for k in keys):
                triples.append([r.Index, *(singles[k] for k in keys)])
        chosen = set()
        # At most 3/4 of the row budget is reserved for measured triples.
        for j in rng.permutation(len(triples))[:rows // 4]:
            chosen.update(triples[int(j)])
        left = group.drop(index=list(chosen)).assign(_positive=lambda x: x.score >= 0)
        if len(chosen) < rows and len(left):
            chosen.update(balanced_rows(left, rows-len(chosen), ["n_mut", "_positive"], seed).index)
        selected.extend(sorted(chosen))
    return frame.loc[selected].reset_index(drop=True)


@contextmanager
def diagnostic_eval(model):
    """Extra evaluations must not consume training RNGs or change its caches."""
    np_state, py_state = np.random.get_state(), random.getstate()
    training = model.training
    caches = {name: dict(getattr(model, name)) for name in ("_ref_cache", "_random_ref_cache")}
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            try:
                yield
            finally:
                model.train(training)
                for name, cache in caches.items():
                    setattr(model, name, cache)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def tensor_norm(tensors):
    tensors = list(tensors)
    if not tensors:
        return 0.
    return float(torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(t.detach().float()) for t in tensors])))


class ParameterSample:
    """Fixed, evenly spaced coordinates; bounded even for full fine tuning."""
    def __init__(self, parameters, size):
        parameters = list(parameters)
        total = sum(p.numel() for p in parameters)
        coords = np.linspace(0, total-1, min(size, total), dtype=np.int64)
        self.parts = []
        offset = 0
        for p in parameters:
            idx = coords[(coords >= offset) & (coords < offset+p.numel())]-offset
            if len(idx):
                self.parts.append((p, torch.as_tensor(idx, device=p.device)))
            offset += p.numel()
        self.initial = self.values().clone()

    def values(self):
        return torch.cat([p.detach().reshape(-1).index_select(0, idx).float() for p, idx in self.parts])


def objective_loss(frame, args, lossfn):
    """Clean-reference objective, averaged equally over targets like sampling.

    Point objectives use all rows within a target. Ranking objectives use
    bounded chunks because pair/list losses depend on their comparison group.
    """
    from mutant_cache.training_protocol import stability_loss
    values = []
    for _, group in frame.groupby("target_id", sort=True):
        if args.loss in ("mse_dg", "huber_dg"):
            p = torch.tensor(group.pred.to_numpy(), dtype=torch.float64)
            t = torch.tensor(group.score.to_numpy(), dtype=torch.float64)
            depth = torch.tensor(group.n_mut.to_numpy())
            loss = stability_loss(p, t, depth, kind=args.loss, delta=args.huber_delta,
                single_weight=args.single_weight, double_weight=args.double_weight,
                positive_weight=args.stabilizing_double_weight, reduction=args.task_reduction)
            values.append(float(loss) * lossfn.cfg.mse_weight)
        else:
            chunks = []
            for start in range(0, len(group), args.micro_batch):
                g = group.iloc[start:start+args.micro_batch]
                loss, _ = lossfn(torch.tensor(g.pred.to_numpy()), torch.tensor(g.score.to_numpy()), group=None, se=None)
                chunks.append((len(g), float(loss)))
            values.append(sum(n*v for n, v in chunks)/len(group))
    return float(np.mean(values)) if values else float("nan")


def frame_diagnostics(frame, evaluation, args, lossfn):
    values = {"loss": objective_loss(frame, args, lossfn)}
    for row in evaluation["summary"].to_dict("records"):
        key = f"{row['depth']}_{row['metric']}"
        values[key] = row["macro"]
        values[key+"_protein_median"] = row["median"]
        values[key+"_n_proteins"] = row["n_proteins"]
        values[key+"_n_proteins_total"] = row["n_proteins_total"]
    for row in evaluation["pooled"].to_dict("records"):
        for metric in M.METRICS:
            values[f"{row['depth']}_{metric}_pooled"] = row[metric]
    if "pred_add" in frame:
        add_pp = M.per_protein_metrics(frame, "pred_add", min_n=args.eval_min_n,
            protein_col=args.eval_protein_col)
        for row in M.summarize(add_pp, args.eval_protein_col).to_dict("records"):
            values[f"add_{row['depth']}_{row['metric']}"] = row["macro"]
        for row in M.pooled_metrics(frame, "pred_add").to_dict("records"):
            for metric in M.METRICS:
                values[f"add_{row['depth']}_{metric}_pooled"] = row[metric]
    for depth, group in evaluation["per_protein"].groupby("depth", sort=True):
        for metric in M.METRICS:
            v = group[metric].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                for q in (10, 90):
                    values[f"{depth}_{metric}_protein_p{q}"] = float(np.percentile(v, q))
    for depth, group in (("all", frame[frame.n_mut > 0]), ("single", frame[frame.n_mut == 1]),
                         ("double", frame[frame.n_mut == 2])):
        if group.empty:
            continue
        p, t = group.pred.to_numpy(float), group.score.to_numpy(float)
        e = p-t
        base = f"calibration/{depth}/"
        values.update({base+"prediction_mean": float(p.mean()), base+"prediction_std": float(p.std()),
            base+"target_mean": float(t.mean()), base+"target_std": float(t.std()),
            base+"bias": float(e.mean())})
        var = float(np.var(p))
        if var > 0:
            slope = float(np.mean((p-p.mean())*(t-t.mean()))/var)
            values.update({base+"slope": slope, base+"intercept": float(t.mean()-slope*p.mean())})
        if t.std() > 0:
            values[base+"std_ratio"] = float(p.std()/t.std())
        for name, mask in (("score_ge_0", t >= 0), ("score_lt_0", t < 0)):
            values[f"class/{depth}/{name}/rows"] = int(mask.sum())
            if mask.any():
                values[f"class/{depth}/{name}/rmse"] = float(np.sqrt(np.mean(e[mask]**2)))
                values[f"class/{depth}/{name}/mae"] = float(np.mean(np.abs(e[mask])))
    doubles = frame[frame.n_mut == 2]
    if len(doubles):
        additive = doubles.pred_add if "pred_add" in doubles else doubles.pred
        residual = doubles.pred.to_numpy()-additive.to_numpy()
        values["interaction/predicted_rms"] = float(np.sqrt(np.mean(residual**2)))
        # Paired macro differences, not pooled errors dominated by large proteins.
        gains = []
        for _, group in doubles.assign(_add=additive).groupby(args.eval_protein_col):
            gains.append(float(np.sqrt(np.mean((group._add-group.score)**2))
                               - np.sqrt(np.mean((group.pred-group.score)**2))))
        values["interaction/double_rmse_gain_over_additive_macro"] = float(np.mean(gains))
    eps = evaluation["epistasis_table"]
    values["interaction/complete_groups"] = len(eps)
    if len(eps):
        values["interaction/observed_epsilon_rms"] = float(np.sqrt(np.mean(eps.eps_obs**2)))
        values["interaction/predicted_epsilon_rms"] = float(np.sqrt(np.mean(eps.eps_pred**2)))
        block = M._block(eps.eps_obs.to_numpy(float), eps.eps_pred.to_numpy(float), 1)
        values.update({"epistasis_"+k+"_pooled": block[k] for k in M.METRICS})
    for row in evaluation["retrieval_summary"].to_dict("records"):
        for key in ("ndcg_macro", "ndcg_t0_macro", "precision_macro", "recall_macro", "n_proteins"):
            values["retrieval96/"+str(row["subset"])+"/"+key] = row[key]
    return values


class ActivationProfile:
    """Hooks exist only for a bounded eval probe; no training tensors retained."""
    def __init__(self, model):
        self.model = model
        self.sums, self.counts, self.handles = {}, {}, []

    def real_rows(self, tensor):
        b = self.batch
        if self.ragged:
            return tensor.reshape(-1, tensor.shape[-1])[:self.tokens]
        if tensor.shape[:2] == b.var_mask.shape:
            return tensor[b.var_mask]
        if tensor.ndim == 3 and tensor.shape[0] == b.n:
            return tensor[:, b.positions][b.var_mask]
        raise ValueError("unsupported activation shape for the diagnostic mask")

    def record(self, name, values):
        if not values.numel():
            return
        s = values.detach().float().square().sum()
        self.sums[name] = self.sums.get(name, 0) + s
        self.counts[name] = self.counts.get(name, 0) + values.numel()

    def base_projection(self, block, name, x):
        if name in ("q", "k", "v"):
            module = block.attn.layernorm_qkv[1]
            d = self.model.backbone.d_model
            lo = {"q": 0, "k": d, "v": 2*d}[name]
            w = module.weight[lo:lo+d]
            bias = None if module.bias is None else module.bias[lo:lo+d]
        elif name == "o":
            w, bias = block.attn.out_proj.weight, block.attn.out_proj.bias
        else:
            module = block.ffn[3] if name in ("fc2", "down") else block.ffn[1]
            w, bias = module.weight, module.bias
            if name in ("gate", "up"):
                d = block.ffn[3].in_features
                lo = 0 if name == "gate" else d
                w = w[lo:lo+d]
                bias = None if bias is None else bias[lo:lo+d]
        return F.linear(x.to(w.dtype), w, bias)

    def __enter__(self):
        try:
            if hasattr(self.model, "head_mlp"):
                def head_input(module, inputs):
                    self.head_calls += 1
                    if self.head_calls == 1:
                        self.record("head/input", inputs[0][self.batch.n_mut > 0])
                self.handles.append(self.model.head_mlp[0].register_forward_pre_hook(head_input))
                if len(self.model.head_mlp) > 1:
                    def head_hidden(module, inputs, output):
                        if self.head_calls == 1:
                            self.record("head/hidden", output[self.batch.n_mut > 0])
                    self.handles.append(self.model.head_mlp[1].register_forward_hook(head_hidden))
            for i, block in enumerate(self.model.backbone.blocks):
                for name, module in (("pre_attention", block.attn.layernorm_qkv[0]),
                                     ("post_attention", block.ffn[0])):
                    def before(module, inputs, key=f"layer_{i}/{name}"):
                        self.record(key, self.real_rows(inputs[0]))
                    self.handles.append(module.register_forward_pre_hook(before))
                for which in ("variant", "shared"):
                    adapters = getattr(self.model.backbone, which+"_adapters", None)
                    if adapters is None:
                        continue
                    for name, delta in adapters[i].deltas.items():
                        def after(module, inputs, output, block=block, name=name,
                                  key=f"layer_{i}/{which}_{name}"):
                            x, y = self.real_rows(inputs[0]), self.real_rows(output)
                            self.record(key+"/delta", y)
                            self.record(key+"/base", self.base_projection(block, name, x))
                        self.handles.append(delta.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def collect(self, batch):
        # The residual model runs a singles pass and a joint pass; hooks assume one
        # batch, so profile its additive (singles) pass as for additive composition.
        if self.model.cfg.composition == "additive" or self.model.cfg.interaction_residual:
            from mutant_cache.models.variant_model import singles_batch
            batch, _ = singles_batch(batch)
        self.batch = batch
        self.head_calls = 0
        self.tokens = int(batch.var_mask.sum())
        self.ragged = self.model._use_ragged(self.model.cfg.backend)
        output = self.model(batch, composition="joint")
        self.record("final_hidden", output.hidden)
        self.record("prediction", output.pred[batch.n_mut > 0])

    def values(self):
        out = {k+"/rms": math.sqrt(float(v)/self.counts[k]) for k, v in self.sums.items()}
        for key in self.sums:
            if key.endswith("/delta"):
                base = key[:-6]+"/base"
                if self.sums.get(base, 0) > 0:
                    out[key[:-6]+"/delta_to_base_rms"] = math.sqrt(float(self.sums[key]/self.sums[base]))
        return out

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TrainingDiagnostics:
    def __init__(self, args, run_dir, tracker):
        self.args, self.tracker = args, tracker
        self.settings = {k: getattr(args, k, v) for k, v in DEFAULTS.items()}
        self.enabled = self.settings["diagnostics"]
        self.path = Path(run_dir)/"diagnostics"
        self.eval_count = 0
        self.clip_norms, self.samples = [], {}
        self.exposure = Counter()
        self.window_exposure = Counter()
        self.by_reference = Counter()
        self.assay_loss_sum = Counter()
        self.assay_loss_rows = Counter()
        self.training_seconds = 0.
        if self.enabled:
            if any(not isinstance(v, int) or v <= 0 for k, v in self.settings.items()
                   if k not in ("diagnostics", "diagnostic_seed")):
                raise ValueError("diagnostic rates and budgets must be positive integers")
            self.path.mkdir(parents=True, exist_ok=True)

    def emit(self, values, step, kind):
        from mutant_cache.tracking import finite_scalars
        values = finite_scalars(values)
        with (self.path/"history.jsonl").open("a") as f:
            f.write(json.dumps(dict(step=step, kind=kind, **values), allow_nan=False)+"\n")
        self.tracker.log(values, step)

    def prepare(self, model, frame, sequences, aa, bkw, batch_size, device, lossfn, groups):
        if not self.enabled:
            return
        from mutant_cache.models import build_variant_batches
        self.model, self.lossfn = model, lossfn
        self.samples = {g["name"]: ParameterSample(g["params"], self.args.diagnostic_parameter_samples) for g in groups}
        self.probe = training_probe(frame, self.args.diagnostic_probe_proteins,
            self.args.diagnostic_probe_rows, self.args.diagnostic_seed)
        self.probe.to_csv(self.path/"train_probe.csv", index=False)
        self.batches = [b.to(device) for b in build_variant_batches(
            self.probe, sequences, aa, batch_size=batch_size, **bkw)[0]]
        activation = training_probe(self.probe, self.args.diagnostic_activation_proteins,
            self.args.diagnostic_activation_rows, self.args.diagnostic_seed)
        self.activation_batches = [b.to(device) for b in build_variant_batches(
            activation, sequences, aa, batch_size=self.args.diagnostic_activation_rows, **bkw)[0]]
        metadata = dict(**self.settings,
            probe_rows=len(self.probe), probe_targets=self.probe.target_id.nunique(),
            probe_sha256=hashlib.sha256((self.path/"train_probe.csv").read_bytes()).hexdigest(),
            activation_rows=len(activation), clean_reference_objective="equal target mean",
            positive_class_threshold=0., benchmark_detection_threshold=.5)
        (self.path/"settings.json").write_text(json.dumps(metadata, indent=2))
        self.tracker.summary({"diagnostics/"+k: v for k, v in metadata.items()})

    def record_batch(self, batch, augmented_ids):
        if not self.enabled:
            return
        observed = dict(rows=len(batch), mutation_tokens=int(batch.n_mut.sum()),
            score_ge_0=int((batch.score >= 0).sum()), score_ge_0_5=int((batch.score >= .5).sum()),
            augmented_reference=int(batch.target_id.isin(augmented_ids).sum()),
            **{f"depth_{int(k)}": int(v) for k, v in batch.n_mut.value_counts().items()})
        self.exposure.update(observed)
        self.window_exposure.update(observed)
        self.by_reference.update(batch.target_id.value_counts().to_dict())

    def begin_step(self, step, total):
        if self.enabled:
            self.step_started = time.perf_counter()
        self.sample_step = self.enabled and (step % self.args.diagnostic_log_every == 0 or step == total)

    def record_assay_loss(self, assay, loss, rows):
        """Accumulate the actual minibatch objective, weighted by sampled rows.

        Call only for homogeneous assay minibatches. Includes configured depth
        weights/ranking terms; it is not a physical-unit error metric.
        """
        if self.enabled:
            self.assay_loss_sum[str(assay)] += float(loss) * rows
            self.assay_loss_rows[str(assay)] += rows

    def assay_loss_values(self):
        values = {}
        for assay, rows in self.assay_loss_rows.items():
            prefix = f"train_assay/{assay}"
            values[prefix+"/loss"] = self.assay_loss_sum[assay]/rows
            values[prefix+"/window_rows"] = rows
        self.assay_loss_sum.clear()
        self.assay_loss_rows.clear()
        return values

    def before_clip(self, groups):
        if not self.sample_step:
            return
        self.pending, self.previous = {}, {}
        named = dict(self.model.named_parameters())
        for group in groups:
            name = group["name"]
            parameters = group["params"]
            self.pending[f"optimization/{name}/grad_norm_pre_clip"] = tensor_norm(
                p.grad for p in parameters if p.grad is not None)
            self.pending[f"optimization/{name}/parameter_norm"] = tensor_norm(parameters)
            self.pending[f"optimization/{name}/missing_gradient_fraction"] = sum(
                p.numel() for p in parameters if p.grad is None)/sum(p.numel() for p in parameters)
            grads = [p.grad.detach() for p in parameters if p.grad is not None]
            self.pending[f"optimization/{name}/nonfinite_gradient_elements"] = sum(
                int((~torch.isfinite(g)).sum()) for g in grads)
            self.pending[f"optimization/{name}/zero_gradient_fraction"] = sum(
                int((g == 0).sum()) for g in grads)/max(sum(g.numel() for g in grads), 1)
            for ab in ("lora_A", "lora_B"):
                identities = {id(p) for p in parameters}
                grads = [p.grad for key, p in named.items() if id(p) in identities and ab in key and p.grad is not None]
                if grads:
                    self.pending[f"optimization/{name}/{ab}_grad_norm_pre_clip"] = tensor_norm(grads)
            if name not in self.samples:
                self.samples[name] = ParameterSample(parameters, self.args.diagnostic_parameter_samples)
            self.previous[name] = self.samples[name].values().clone()
        self.pending["optimization/grad_norm_pre_clip_sampled"] = math.sqrt(sum(
            self.pending[f"optimization/{g['name']}/grad_norm_pre_clip"]**2 for g in groups))

    def observe_clip(self, norm):
        if self.enabled and norm is not None:
            self.clip_norms.append(norm.detach())

    def after_step(self, groups, step):
        if not self.enabled:
            return
        if self.sample_step and next(self.model.parameters()).is_cuda:
            torch.cuda.synchronize(next(self.model.parameters()).device)
        self.training_seconds += time.perf_counter()-self.step_started
        if not self.sample_step:
            return
        values = self.pending
        values.update(self.assay_loss_values())
        for group in groups:
            name = group["name"]
            sample = self.samples[name]
            current, previous = sample.values(), self.previous[name]
            update = float((current-previous).square().mean().sqrt())
            scale = float(previous.square().mean().sqrt())
            values.update({f"optimization/{name}/update_rms_sampled": update,
                f"optimization/{name}/relative_update_sampled": update/max(scale, 1e-12),
                f"optimization/{name}/distance_from_initial_rms_sampled": float((current-sample.initial).square().mean().sqrt()),
                f"optimization/{name}/sampled_coordinates": current.numel()})
        if self.clip_norms:
            norms = torch.stack(self.clip_norms).float().cpu().numpy()
            good = norms[np.isfinite(norms)]
            values["optimization/nonfinite_gradient_steps"] = int((~np.isfinite(norms)).sum())
            values["optimization/gradient_observations"] = len(norms)
            if len(good):
                values["optimization/grad_norm_pre_clip_mean"] = float(good.mean())
                values["optimization/grad_norm_pre_clip_max"] = float(good.max())
                values["optimization/clipped_fraction"] = float(np.mean(good > self.args.grad_clip))
            self.clip_norms.clear()
        elif not self.args.grad_clip:
            values["optimization/clipped_fraction"] = 0.
        n = max(self.exposure["rows"], 1)
        for key, count in self.exposure.items():
            values["sampling/"+key] = count
            if key not in ("rows", "mutation_tokens"):
                values["sampling/"+key+"_fraction"] = count/n
                values["sampling/window_"+key+"_fraction"] = self.window_exposure[key]/max(self.window_exposure["rows"], 1)
        values["sampling/references_seen"] = len(self.by_reference)
        self.window_exposure.clear()
        values["performance/train_rows_per_s"] = self.exposure["rows"]/max(self.training_seconds, 1e-12)
        values["performance/train_mutation_tokens_per_s"] = self.exposure["mutation_tokens"]/max(self.training_seconds, 1e-12)
        if next(self.model.parameters()).is_cuda:
            device = next(self.model.parameters()).device
            values.update({"memory/allocated_gb": torch.cuda.memory_allocated(device)/2**30,
                "memory/reserved_gb": torch.cuda.memory_reserved(device)/2**30})
        self.emit(values, step, "optimization")
        self.previous.clear()

    def evaluate(self, step, exposure, val_predictions, val_evaluation, force=False):
        if not self.enabled:
            return
        self.eval_count += 1
        final_step = getattr(self, "total_steps", -1)
        media = force or step == 0 or step == final_step or self.eval_count % self.args.diagnostic_profile_every == 0
        if not (media or self.eval_count % self.args.diagnostic_eval_every == 0):
            return
        from mutant_cache.megascale import predictions, eval_summary
        started = time.perf_counter()
        with diagnostic_eval(self.model):
            train = predictions(self.model, self.batches, self.probe, self.model.cfg)
            train_eval = eval_summary(train, self.args.eval_min_n, self.args.selection, self.args.eval_protein_col)
            for split, frame, ev, retrieval in (("train_probe", train, train_eval["eval"], train_eval["row"]),
                    ("val", val_predictions, val_evaluation, {})):
                values = frame_diagnostics(frame, ev, self.args, self.lossfn)
                # Preserve the existing nDCG definitions for both splits.
                if split == "train_probe":
                    values.update({k: v for k, v in retrieval.items() if k.startswith(("ndcg", "precision"))})
                self.emit({split+"/"+k: v for k, v in values.items()}, step, split)
                per_protein = ev["per_protein"].assign(step=step, split=split, exposure=exposure)
                per_protein.to_csv(self.path/"per_protein.csv", mode="a", index=False,
                    header=not (self.path/"per_protein.csv").exists())
                if media:
                    subset = balanced_rows(frame[frame.n_mut > 0], self.args.diagnostic_plot_rows,
                        ["target_id", "n_mut"], self.args.diagnostic_seed)
                    points = subset[["score", "pred", "target_id", "n_mut"]].rename(columns={"score": "observed", "pred": "predicted"})
                    points.to_csv(self.path/f"{split}_scatter_step_{step}.csv", index=False)
                    tables = {split+"/per_protein": per_protein.head(self.args.diagnostic_plot_rows)}
                    scatters = {split+"/predicted_vs_observed": points}
                    eps = ev["epistasis_table"]
                    if len(eps):
                        ep = eps.iloc[:self.args.diagnostic_plot_rows][["eps_obs", "eps_pred"]].rename(
                            columns={"eps_obs": "observed", "eps_pred": "predicted"})
                        scatters[split+"/interaction_observed_vs_predicted"] = ep
                    self.tracker.diagnostic_media(tables, scatters, step)
            if media:
                from mutant_cache.cache.variant_cache import reference_kv
                totals, counts = {}, {}
                for batch in self.activation_batches:
                    # Prime reference computations outside the variant-only hooks.
                    ref = self.model.reference_states(batch)
                    if self.model.cfg.random_reference:
                        ref = self.model._randomize(ref, batch.target_id)
                    reference_kv(self.model.backbone, ref)
                    with ActivationProfile(self.model) as profile:
                        profile.collect(batch)
                    for key, value in profile.sums.items():
                        totals[key] = totals.get(key, 0) + value
                        counts[key] = counts.get(key, 0) + profile.counts[key]
                profile.sums, profile.counts = totals, counts
                self.emit({"activation/"+k: v for k, v in profile.values().items()}, step, "activation")
        elapsed = time.perf_counter()-started
        self.emit({"performance/diagnostic_eval_s": elapsed}, step, "timing")

    def selected_checkpoint(self, val_predictions, val_evaluation, training_step):
        if not self.enabled:
            return
        from mutant_cache.megascale import predictions, eval_summary
        with diagnostic_eval(self.model):
            train = predictions(self.model, self.batches, self.probe, self.model.cfg)
            train_eval = eval_summary(train, self.args.eval_min_n, self.args.selection, self.args.eval_protein_col)
            values = {}
            for split, frame, ev in (("train_probe", train, train_eval["eval"]), ("val", val_predictions, val_evaluation)):
                values.update({"selected/"+split+"/"+k: v for k, v in frame_diagnostics(
                    frame, ev, self.args, self.lossfn).items()})
            self.emit(values, training_step, "selected_checkpoint")

    def finish(self):
        if self.enabled:
            pd.DataFrame(sorted(self.by_reference.items()), columns=["reference_id", "rows_seen"]).to_csv(
                self.path/"reference_exposure.csv", index=False)

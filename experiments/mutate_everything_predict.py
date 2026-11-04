"""Run in the *upstream* conda environment (fair-esm, not EvolutionaryScale esm).

Score the supplied canonical labels, including ALL requested doubles. Upstream
test.py exports only stabilizing doubles and cannot serve as a matched scorer.
Arguments use upstream's parser; extra flags: --repo, --labels, --predictions.
"""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

PIN = "034dd91e7a449e68b0f0993043bd32ce281f056c"
p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--labels", type=Path, required=True)
p.add_argument("--predictions", type=Path, required=True)
p.add_argument("--checkpoint-only-backbone", action="store_true",
               help="construct the original fair-esm architecture without an extra pretrained download; "
                    "strict checkpoint loading must supply every weight")
p.add_argument("--requested-decoding", action="store_true",
               help="gather supplied mutations through the original decoder; supports any mutation depth")
bridge, remaining = p.parse_known_args()
commit = subprocess.check_output(["git", "-C", str(bridge.repo), "rev-parse", "HEAD"], text=True).strip()
if commit != PIN:
    raise ValueError(f"upstream must be pinned to {PIN}, got {commit}")
dirty = subprocess.check_output(["git", "-C", str(bridge.repo), "status", "--porcelain", "--untracked-files=no"], text=True)
if dirty:
    raise ValueError("upstream tracked source has local modifications")
sys.path.insert(0, str(bridge.repo.resolve()))
from test import get_args_parser
from modeling.mutate_everything import MutateEverything

args = get_args_parser().parse_args(remaining)
args.test = True
args.eval = True
args.finetune = None
if args.backbone != "esm2_t33_650M_UR50D":
    raise ValueError("this bridge supports only the sequence ESM-2 comparator")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if bridge.checkpoint_only_backbone:
    # The released checkpoint contains the entire trained backbone. Use the
    # unmodified fair-esm constructor, then require a strict full-model load.
    # This avoids downloading initialization weights which would be overwritten.
    import esm
    from esm.model.esm2 import ESM2
    def checkpoint_backbone():
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        return ESM2(33, 1280, 20, alphabet, token_dropout=True), alphabet
    esm.pretrained.esm2_t33_650M_UR50D = checkpoint_backbone
model = MutateEverything(args)
if hasattr(torch.serialization, "safe_globals"):
    with torch.serialization.safe_globals([argparse.Namespace, Path, type(Path("."))]):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
else:  # upstream's original torch environment, before safe_globals support
    kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    checkpoint = torch.load(args.resume, map_location="cpu", **kwargs)
state = checkpoint["model"]
if all(k.startswith("module.") for k in state):
    state = {k[len("module."):]: v for k, v in state.items()}
model.load_state_dict(state, strict=True)
print(f"strictly loaded {len(state)} checkpoint tensors", flush=True)
model.to(args.device).eval()
_, alphabet = model.backbone.get_alphabet()
converter = alphabet.get_batch_converter()
labels = pd.read_csv(bridge.labels)
if labels.duplicated(["target_id", "mutant"]).any():
    raise ValueError("duplicate input labels")
if not bridge.requested_decoding and not labels.n_mut.isin([1, 2]).all():
    raise ValueError("bridge currently scores singles and doubles only")
aa = "ACDEFGHIKLMNPQRSTVWY"  # upstream data.one_letters: alphabetical
device = torch.device(args.device)
def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)
timings, output = [], []
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
with torch.no_grad():
    for target, group in labels.groupby("reference_id" if "reference_id" in labels else "target_id", sort=True):
        if group.wt_seq.nunique() != 1:
            raise ValueError(f"{target}: ambiguous reference")
        seq = group.wt_seq.iloc[0]
        _, _, tokens = converter([(str(target), seq)])
        tokens = tokens.to(device)
        sync()
        t0 = time.perf_counter()
        if bridge.requested_decoding:
            from mutant_cache.baselines.mutate_everything import score_requested
            prediction = score_requested(model, tokens, seq, group.mutant.tolist())
        else:
            prediction = model(tokens, {"seqs": [seq]})
        sync()
        timings.append(dict(target_id=target, rows=len(group), seconds=time.perf_counter()-t0,
                            length=len(seq)))
        single = None if bridge.requested_decoding else prediction["mut1_ddg"][0]
        double = None if bridge.requested_decoding else prediction.get("mut2_ddg")
        for row_index, row in enumerate(group.to_dict("records")):
            subs = sorted([(int(m[1:-1])-1, m[0], m[-1]) for m in row["mutant"].split(":")])
            for pos, wt, alt in subs:
                if seq[pos] != wt:
                    raise ValueError(f"{target}: WT mismatch {row['mutant']}")
            if bridge.requested_decoding:
                ddg = prediction[row_index]
            elif len(subs) == 1:
                pos, _, alt = subs[0]
                ddg = single[pos, aa.index(alt)]
            else:
                if double is None:
                    raise ValueError("checkpoint did not return double predictions")
                p1, _, a1 = subs[0]
                p2, _, a2 = subs[1]
                ddg = double[0, p1, aa.index(a1), p2, aa.index(a2)]
            # Official engine_train.py negates model ddG for stabilizing scores.
            output.append({**row, "pred": -float(ddg.cpu())})
        del prediction, single, double
        print(f"scored {target}: {len(group)} variants in {timings[-1]['seconds']:.3f}s", flush=True)
bridge.predictions.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(output).to_csv(bridge.predictions, index=False)
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
record = dict(upstream_commit=commit, checkpoint_sha256=file_hash(args.resume),
              labels_sha256=file_hash(bridge.labels), backbone=args.backbone,
              regime="dense singles and doubles decoded per reference", timings=timings,
              timing_scope="synchronized model forward only; excludes tokenization, extraction, loading",
              peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
record.update(checkpoint_only_backbone=bridge.checkpoint_only_backbone,
              checkpoint_tensors=len(state), checkpoint_load_strict=True)
record['requested_decoding'] = bridge.requested_decoding
if bridge.requested_decoding:
    record['regime'] = 'supplied library through original mutation decoder'
bridge.predictions.with_suffix(".json").write_text(json.dumps(record, indent=2))

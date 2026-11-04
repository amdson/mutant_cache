"""Score external canonical labels with a frozen trained checkpoint.

No training, selection or label-dependent calibration. ``reference_id`` may
identify mutant references separately from ``target_id`` for reverse tests.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mutant_cache.harness import load_backbone, attach_lora, build_config
from mutant_cache.megascale import load_trainable, score
from mutant_cache.models import VariantEffectModel, build_variant_batches
from mutant_cache.data.mutate_everything import sha256


def score_labels(model, frame, *, batch_size=512, additive=True):
    d = frame.reset_index(drop=True).copy()
    keys = d['reference_id'] if 'reference_id' in d else d.target_id
    if d.assign(_reference=keys).groupby('_reference').wt_seq.nunique().gt(1).any():
        raise ValueError('reference ID has multiple sequences')
    seqs = dict(zip(keys, d.wt_seq))
    work = d.assign(target_id=keys, score_set_id=keys)
    # The trained minimal stability head is shared across proteins.
    batches, _ = build_variant_batches(work, seqs, model.backbone.aa_to_id,
        assay_index={k: 0 for k in keys.unique()}, batch_size=batch_size,
        tokenizer=model.backbone.tokenizer, target_col='score', se_col=None)
    device = next(model.parameters()).device
    batches = [b.to(device) for b in batches]
    model.clear_reference_cache()
    d['pred'] = score(model, batches, work)
    if additive and model.cfg.composition == 'joint':
        d['pred_add'] = score(model, batches, work, composition='additive')
    if not np.isfinite(d['pred'].to_numpy()).all():
        raise ValueError('missing or nonfinite predictions')
    return d


def run(checkpoint, labels, output, device='cuda:0', batch_size=512):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**ck['args'])
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    backbone = load_backbone(args, device, dtype)
    backbone.freeze_base()
    attach_lora(args, backbone)
    model = VariantEffectModel(backbone, build_config(args, backbone, 1)).to(device)
    load_trainable(model, ck['trainable_state'])
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for path in labels:
        path = Path(path)
        print('START scoring', path.stem, flush=True)
        pred = score_labels(model, pd.read_csv(path), batch_size=batch_size)
        pred['seed'] = args.seed
        dest = output / f'{path.stem}_predictions.csv'
        pred.to_csv(dest, index=False)
        print('COMPLETE', path.stem, len(pred), 'rows', flush=True)
    (output / 'provenance.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        seed=args.seed, composition=args.composition, device=str(device),
        labels={str(p): sha256(p) for p in labels}, training_steps=0,
        calibration='none', checkpoint_selection='internal validation only'), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--labels', required=True, type=Path, nargs='+')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', default=512, type=int)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()

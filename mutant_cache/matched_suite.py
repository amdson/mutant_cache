"""Frozen-checkpoint, strict matched-row multi-panel evaluation.

python -m mutant_cache.matched_suite --config suite.json --output report
Paths in the config are relative to its directory. No training or calibration.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mutant_cache.comparison import compare
from mutant_cache.data.mutate_everything import sha256
from mutant_cache.eval.mutate_everything import canonical_predictions, join_predictions


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'unsafe or empty identifier: {value!r}')
    return value


def _path(base, value):
    p = Path(value)
    return p if p.is_absolute() else base / p


def _read(path):
    # Preserve numeric-looking protein IDs, including leading zeroes.
    return pd.read_csv(path, dtype={'target_id': str, 'reference_id': str})


def _matched(labels, predictions, context):
    canonical = canonical_predictions(labels.assign(pred=0.0))
    _, unmatched = join_predictions(predictions, canonical)
    if len(unmatched):
        raise ValueError(f'{context}: {len(unmatched)} missing or extra rows; refusing subset comparison')


def run(config, output, *, device='cuda:0', batch_size=512, validate_only=False):
    config = Path(config).resolve()
    spec = json.loads(config.read_text())
    base = config.parent
    panels = spec['panels']
    models = spec['models']
    if not panels or not models:
        raise ValueError('at least one panel and model required')
    panel_ids = [_identifier(p['id']) for p in panels]
    model_ids = [_identifier(m['id']) for m in models]
    if len(set(panel_ids)) != len(panel_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError('duplicate panel or model IDs')
    loaded, provenance = {}, {'config_sha256': sha256(config), 'panels': {}, 'models': {}}
    for panel in panels:
        pid = panel['id']
        labels_path, baseline_path = [_path(base, panel[k]) for k in ('labels', 'baseline')]
        labels, baseline = _read(labels_path), canonical_predictions(_read(baseline_path))
        _matched(labels, baseline, f'{pid} comparator')
        if 'expected_rows' in panel and len(labels) != panel['expected_rows']:
            raise ValueError(f'{pid}: expected {panel["expected_rows"]} rows, got {len(labels)}')
        loaded[pid] = labels, baseline, labels_path
        provenance['panels'][pid] = dict(labels=str(labels_path), labels_sha256=sha256(labels_path),
            baseline=str(baseline_path), baseline_sha256=sha256(baseline_path), rows=len(labels),
            proteins=int(labels.target_id.nunique()), comparator=panel.get('comparator', 'unspecified'),
            interpretation=panel.get('interpretation', 'matched benchmark evaluation'))
    # Check every supplied prediction file before loading any checkpoint.
    for model in models:
        fits = model['fits']
        seeds = [f['seed'] for f in fits]
        if not fits or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f'{model["id"]}: fits require distinct integer seeds')
        provenance['models'][model['id']] = dict(training_protocol=model.get('training_protocol',
            'unspecified; full-training-set alignment not established'), fits=[])
        for fit in fits:
            if ('checkpoint' in fit) == ('predictions' in fit):
                raise ValueError('each fit needs exactly one of checkpoint or predictions')
            record = dict(seed=fit['seed'])
            if 'checkpoint' in fit:
                checkpoint = _path(base, fit['checkpoint'])
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                import torch
                ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if ck['args']['seed'] != fit['seed']:
                    raise ValueError(f'{checkpoint}: checkpoint seed differs from declared seed')
                record.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                              checkpoint_args=ck['args'])
                del ck
            else:
                record['predictions'] = {}
                for pid in panel_ids:
                    path = _path(base, fit['predictions'][pid])
                    frame = _read(path)
                    if 'seed' in frame and not frame.seed.eq(fit['seed']).all():
                        raise ValueError(f'{path}: inconsistent seed column')
                    _matched(loaded[pid][0], frame, f'{model["id"]} seed {fit["seed"]} {pid}')
                    record['predictions'][pid] = dict(path=str(path), sha256=sha256(path))
            provenance['models'][model['id']]['fits'].append(record)
    if validate_only:
        return provenance
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty; reports are immutable')
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(spec, indent=2))
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (output/'status.json').write_text(json.dumps({'status': 'running'}))
    try:
        rows, states = [], []
        for model in models:
            scored = {pid: [] for pid in panel_ids}
            for fit in model['fits']:
                if 'checkpoint' in fit:
                    from mutant_cache.benchmark import run as score_checkpoint
                    dest = output/'predictions'/model['id']/f'seed{fit["seed"]}'
                    paths = [loaded[pid][2] for pid in panel_ids]
                    if len({p.stem for p in paths}) != len(paths):
                        raise ValueError('label filename stems must be unique for checkpoint scoring')
                    score_checkpoint(_path(base, fit['checkpoint']), paths, dest, device, batch_size)
                    predictions = {pid: dest/f'{loaded[pid][2].stem}_predictions.csv' for pid in panel_ids}
                else:
                    predictions = {pid: _path(base, fit['predictions'][pid]) for pid in panel_ids}
                for pid in panel_ids:
                    frame = _read(predictions[pid])
                    _matched(loaded[pid][0], frame, f'{model["id"]} seed {fit["seed"]} {pid}')
                    scored[pid].append((fit['seed'], frame))
            for panel in panels:
                pid = panel['id']
                dest = output/'reports'/model['id']/pid
                status = compare(loaded[pid][1], scored[pid], dest,
                    expected=panel.get('expected'), expected_source=panel.get('expected_source'))
                states.append(dict(model=model['id'], panel=pid, **status))
                table = pd.read_csv(dest/'matched_metrics.csv')
                table['model'] = model['id']
                table['panel'] = pid
                rows.append(table)
        all_metrics = pd.concat(rows, ignore_index=True)
        all_metrics.to_csv(output/'metrics.csv', index=False)
        metrics = ['spearman', 'pearson', 'rmse', 'mae', 'ndcg30', 'precision30']
        summary = all_metrics.groupby(['model', 'panel', 'method', 'depth'])[metrics].agg(['mean', 'std'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(output/'summary.csv', index=False)
        headline = all_metrics[all_metrics.depth == 'all'][['model', 'panel', 'method', 'seed',
            'n', 'spearman', 'rmse', 'ndcg30', 'precision30', 'detection_proteins', 'total_proteins']]
        headline.to_csv(output/'headline.csv', index=False)
        (output/'README.md').write_text('Matched benchmark results. See headline.csv for pooled regression and protein-macro detection per seed, summary.csv for seed means/SD, and reports/ for protein-macro regression, paired protein bootstrap intervals, reproduction checks and exact joins.\n\nTraining provenance is recorded in provenance.json; evaluation matching does not establish matching training conditions. Reverse panels and comparator checkpoint variants retain their manifest labels. No model, seed or checkpoint is selected by this runner. No training or calibration occurs.\n')
        result = dict(status='complete', comparisons=states, training_steps=0, calibration='none')
        (output/'status.json').write_text(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        (output/'status.json').write_text(json.dumps(dict(status='failed', error=str(exc)), indent=2))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    result = run(**vars(args))
    print(json.dumps({'status': 'validated' if args.validate_only else result['status']}, indent=2))


if __name__ == '__main__':
    main()

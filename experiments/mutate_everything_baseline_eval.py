"""Evaluate the released ESM comparator with both original and local metrics."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mutant_cache.eval.mutate_everything import canonical_predictions, evaluate, reproduction_gate

root = Path('/home/andrew/mutant_cache_scratch/mutate_everything')
out = root / 'baseline'
frame = canonical_predictions(pd.read_csv(out / 'predictions_cdna2.csv'))
spec = importlib.util.spec_from_file_location('mutate_everything_official_metrics', root / 'repo' / 'metrics.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
raw = pd.read_csv(root / 'data' / 'cdna2_test.csv', low_memory=False)
predictions = {t: {'mutations': g.mutant.tolist(), 'scores': g.pred.tolist()}
               for t, g in frame.groupby('target_id')}
metrics, detection, per_protein_detection, _, merged = official.eval_ddg(raw, predictions)
assert merged.scores.notna().all(), 'upstream merge left missing predictions'
assert len(merged) == len(frame), 'upstream and local evaluation row counts differ'
local = evaluate(frame)
for key, table in local.items():
    table.to_csv(out / f'local_{key}.csv', index=False)
merged.to_csv(out / 'upstream_joined.csv', index=False)
detection.to_csv(out / 'upstream_detection.csv', index=False)
per_protein_detection.drop(columns=['is_tp']).to_csv(out / 'upstream_per_protein_detection.csv', index=False)
matches = []
row = local['table'].query("depth == 'all'").iloc[0]
for ours, theirs in [('spearman', 'spear'), ('rmse', 'rmse'),
                     ('ndcg30', 'ndcg_-0.5_30'), ('precision30', 'prec_-0.5_30')]:
    actual, expected = float(row[ours]), float(metrics[theirs])
    matches.append({'metric': ours, 'local': actual, 'upstream': expected,
                    'matches': bool(np.isclose(actual, expected, atol=1e-10, rtol=0))})
assert all(m['matches'] for m in matches), matches
gate = reproduction_gate(local['table'], {'all': {'ndcg30': 0.38}}, atol=0.005)
record = {'comparator': 'Mutate Everything ESM-2 650M (released checkpoint)',
          'dataset': 'released cDNA2; median duplicates; all 21933 canonical double mutants',
          'n_proteins': int(frame.target_id.nunique()), 'n_rows': len(frame),
          'official_metrics': {k: float(v) for k, v in metrics.items()},
          'evaluator_parity': matches, 'published_ndcg_gate': gate,
          'published_source': 'https://github.com/jozhang97/MutateEverything/tree/034dd91e7a449e68b0f0993043bd32ce281f056c#main-results',
          'macro_metrics': local['macro'].query("depth == 'double'").to_dict('records'),
          'inference_provenance': json.loads((out / 'predictions_cdna2.json').read_text())}
(out / 'baseline_metrics.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)

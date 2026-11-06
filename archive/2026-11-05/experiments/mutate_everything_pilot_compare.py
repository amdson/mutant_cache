"""Matched-row seed-0 pilot comparison; explicitly not a multi-seed claim."""
import json
from pathlib import Path
import pandas as pd
from mutant_cache.comparison import compare
from mutant_cache.eval.mutate_everything import evaluate

scratch = Path('/home/andrew/mutant_cache_scratch')
pilot = scratch / 'runs' / 'mutate_everything_esm2_pilot_20261104'
baseline = scratch / 'mutate_everything' / 'baseline'
ours = pd.read_csv(pilot / 'joint_s0_cdna2_fixed' / 'predictions_test.csv')
theirs = pd.read_csv(baseline / 'predictions_cdna2.csv')
out = pilot / 'comparison_seed0'
expected = {'all': {'ndcg30': 0.38}}
source = 'https://github.com/jozhang97/MutateEverything/tree/034dd91e7a449e68b0f0993043bd32ce281f056c#main-results'
status = compare(theirs, [(0, ours)], out, expected=expected, expected_source=source)
assert status['reproduction']['passed']
assert status['unmatched_rows'] == 0
assert not status['superiority_claim_eligible']
summary = []
for name, frame in [('Mutate Everything ESM-2 (released)', theirs),
                    ('Our joint ESM-2 pilot (seed 0)', ours),
                    ('Our pilot summed singles (same checkpoint)', ours.assign(pred=ours.pred_add))]:
    e = evaluate(frame)
    row = e['table'].query("depth == 'all'").iloc[0].to_dict()
    row['method'] = name
    row['macro_spearman'] = float(e['macro'].query("depth == 'double' and metric == 'spearman'")['macro'].iloc[0])
    summary.append(row)
pd.DataFrame(summary).to_csv(out / 'headline.csv', index=False)
(out / 'pilot_protocol.json').write_text(json.dumps({
    'seeds': [0], 'preliminary_only': True, 'checkpoint_selected_on': 'internal validation only',
    'selection': json.loads((pilot / 'joint_s0' / 'results.json').read_text())['extra']['selection'],
    'test_rows': len(ours), 'test_proteins': int(ours.target_id.nunique()),
    'published_ndcg_reproduced': True, 'unmatched_rows': 0,
    'training_protocol_difference': 'our 104 train / 12 val proteins; released comparator used all 116 training proteins',
    'model_difference': 'same pretrained ESM-2 size; sparse variant LoRA vs fine-tuned dense decoder',
    'next_evidence_needed': 'three seeds, validation-only model choices and paired protein intervals'
}, indent=2))
print(pd.DataFrame(summary)[['method', 'spearman', 'macro_spearman', 'ndcg30', 'precision30', 'rmse']].to_string(index=False))
print(json.dumps(status, indent=2))

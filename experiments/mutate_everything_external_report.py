"""Full-coverage external score reports and original-evaluator parity."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mutant_cache.comparison import compare
from mutant_cache.eval.mutate_everything import evaluate, join_predictions

root = Path('/home/andrew/mutant_cache_scratch/mutate_everything')
external = root / 'external'
spec = importlib.util.spec_from_file_location('original_metrics', root / 'repo/metrics.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
tables, parity = [], []
# Real-checkpoint requested decoding must agree with the existing dense cDNA2 run.
dense = pd.read_csv(root / 'baseline/predictions_cdna2.csv')
requested = pd.read_csv(external / 'baseline/cdna2_oracle_predictions.csv')
j, missing = join_predictions(requested, dense)
assert missing.empty and len(j) == 21933
assert np.allclose(j.pred_ours, j.pred_theirs, atol=3e-5, rtol=3e-5)
parity.append(dict(test='cDNA2 requested vs dense checkpoint decoding', rows=len(j),
                   max_abs_diff=float(np.abs(j.pred_ours-j.pred_theirs).max()), passed=True))
for tag in ['s669', 'ptmul', 'cdna1', 's669_reverse']:
    baseline_file = f'{tag}_single_only_predictions.csv' if tag == 's669' else f'{tag}_predictions.csv'
    theirs = pd.read_csv(external / f'baseline/{baseline_file}')
    ours = pd.read_csv(external / f'ours_joint_s0/{tag}_labels_predictions.csv')
    expected = {'all': {'spearman': 0.47}} if tag == 's669' else None
    source = 'https://github.com/jozhang97/MutateEverything#main-results' if expected else None
    status = compare(theirs, [(0, ours)], external / f'report_{tag}', expected=expected, expected_source=source)
    assert status['unmatched_rows'] == 0
    methods = [('Mutate Everything ESM-2 single-only' if tag == 's669' else 'Mutate Everything ESM-2 joint', theirs), ('Ours joint seed 0', ours),
               ('Ours summed singles, same checkpoint', ours.assign(pred=ours.pred_add))]
    if tag == 's669':
        methods.append(('Mutate Everything ESM-2 joint (secondary)', pd.read_csv(external/'baseline/s669_predictions.csv')))
    for name, frame in methods:
        ev = evaluate(frame)
        for row in ev['table'].to_dict('records'):
            macro = ev['macro']
            m = macro[(macro.depth == row['depth']) & (macro.metric == 'spearman')]
            tables.append(dict(benchmark=tag, method=name, **row,
                               macro_spearman=float(m.macro.iloc[0]) if len(m) else np.nan))
        # Use the original numerical routines with explicit canonical sign.
        # Reverse labels must not be silently majority-sign-flipped.
        official = original.compute_cls_reg_metrics(-frame.pred.to_numpy(), -frame.score.to_numpy())
        local = ev['table'].query("depth == 'all'").iloc[0]
        for a, b in [('spearman','spear'), ('rmse','rmse')]:
            assert np.isclose(local[a], official[b], atol=1e-10, rtol=0)
        parity.append(dict(test=f'{tag}: {name}: original regression evaluator', passed=True))
pd.DataFrame(tables).to_csv(external / 'headline.csv', index=False)
(external / 'parity_checks.json').write_text(json.dumps(parity, indent=2))
print(pd.DataFrame(tables).query("depth == 'all'")[['benchmark','method','spearman','macro_spearman','rmse']].to_string(index=False), flush=True)

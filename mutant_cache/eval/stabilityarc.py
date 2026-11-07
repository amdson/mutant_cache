"""Audit completed outer-fold predictions against published StabilityArc scores."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compare_fold(fold, run_dir, scratch):
    run_dir, scratch = Path(run_dir), Path(scratch)
    predicted = pd.read_csv(run_dir/'predictions_test.csv')
    expected = pd.read_parquet(scratch/'processed'/fold['dataset']/'variants.parquet')
    expected = expected[expected.target_id == fold['heldout']]
    if predicted.mutant.duplicated().any() or set(predicted.target_id) != {fold['heldout']}:
        raise ValueError('duplicate predictions or wrong held-out protein')
    paired = expected.merge(predicted, on=['target_id','mutant'], validate='one_to_one',
                            suffixes=('_expected','_actual'), how='outer', indicator=True)
    if len(paired) != fold['test_rows'] or not (paired['_merge']=='both').all():
        raise ValueError('incomplete outer-test coverage')
    if not np.isfinite(paired.pred).all() or not np.allclose(paired.score_expected,paired.score_actual,rtol=0,atol=1e-12):
        raise ValueError('nonfinite predictions or altered labels')
    if not np.array_equal(paired.n_mut_expected, paired.n_mut_actual):
        raise ValueError('altered mutation depths')
    rows = []
    for subset, depth, key in [('all',None,'all_variant_spearman'),('single',1,'single_mutant_spearman'),('double',2,'multi_mutant_spearman')]:
        frame = predicted if depth is None else predicted[predicted.n_mut==depth]
        rho = float(spearmanr(frame.score,frame.pred).statistic) if len(frame)>1 else None
        if rho is not None and not np.isfinite(rho):
            raise ValueError(f'undefined correlation on eligible assay: {fold["heldout"]}/{subset}')
        baseline = fold['published'][key] if len(frame)>1 else None
        rows.append(dict(protein=fold['heldout'],assay=fold['published']['assay_id'],subset=subset,n=len(frame),
                         our_spearman=rho,published_stabilityarc_spearman=baseline,
                         difference=None if rho is None else rho-baseline,run_dir=str(run_dir)))
    return rows


def report(root, scratch):
    root, scratch = Path(root), Path(scratch)
    plan = json.loads((root/'study.json').read_text())
    rows, missing = [], []
    for item in plan['folds']:
        run = Path(item['run_dir'])
        # queue_runner regards results+checkpoint as completion; never score a
        # partially written predictions file from a still-running fit.
        if not (run/'results.json').is_file() or not (run/'model.pt').is_file():
            missing.append(item['heldout'])
            continue
        rows.extend(compare_fold(item,run,scratch))
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    summary = dict(completed=66-len(missing),total=66,missing=missing,complete=not missing,
                   scope='Full 66-assay single-seed comparison only when complete; otherwise completed subset.',
                   comparator='Published per-assay StabilityArc scores; private row predictions unavailable.',metrics={})
    for subset in ['all','single','double']:
        block = [r for r in rows if r['subset']==subset and r['our_spearman'] is not None]
        if not block: continue
        delta = np.array([r['difference'] for r in block])
        value = dict(assays=len(block),our_macro=float(np.mean([r['our_spearman'] for r in block])),
                     baseline_macro=float(np.mean([r['published_stabilityarc_spearman'] for r in block])),
                     mean_difference=float(delta.mean()),wins=int((delta>0).sum()))
        if not missing:
            rng=np.random.default_rng(23)
            means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            value['paired_assay_bootstrap_95']=np.percentile(means,[2.5,97.5]).tolist()
            value['interval_scope']='Conditional on these seed-0 fits; does not measure training randomness or preprocessing uncertainty.'
        summary['metrics'][subset]=value
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary

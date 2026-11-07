import pandas as pd
import pytest
from mutant_cache.eval.stabilityarc import compare_fold


def fixture(tmp_path):
    data=tmp_path/'processed/toy'; data.mkdir(parents=True)
    frame=pd.DataFrame(dict(target_id=['p']*4,mutant=['A1C','A1D','A1C:C2D','A1D:C2A'],
                            n_mut=[1,1,2,2],score=[1.,2.,3.,4.]))
    frame.to_parquet(data/'variants.parquet',index=False)
    run=tmp_path/'run';run.mkdir()
    frame.assign(pred=[1.,2.,3.,4.]).to_csv(run/'predictions_test.csv',index=False)
    fold=dict(heldout='p',dataset='toy',test_rows=4,published=dict(assay_id='toy_assay',
              all_variant_spearman=.5,single_mutant_spearman=.25,multi_mutant_spearman=.75))
    return fold,run


def test_exact_coverage_and_depth_scores(tmp_path):
    fold,run=fixture(tmp_path)
    rows=compare_fold(fold,run,tmp_path)
    assert [r['n'] for r in rows]==[4,2,2]
    assert [r['our_spearman'] for r in rows]==pytest.approx([1.,1.,1.])
    assert [r['difference'] for r in rows]==pytest.approx([.5,.75,.25])


@pytest.mark.parametrize('corruption',['missing','label','depth','nonfinite'])
def test_changed_predictions_rejected(tmp_path,corruption):
    fold,run=fixture(tmp_path)
    p=run/'predictions_test.csv';f=pd.read_csv(p)
    if corruption=='missing':f=f.iloc[:-1]
    elif corruption=='label':f.loc[0,'score']=99.
    elif corruption=='depth':f.loc[0,'n_mut']=2
    else:f.loc[0,'pred']=float('inf')
    f.to_csv(p,index=False)
    with pytest.raises(ValueError):compare_fold(fold,run,tmp_path)

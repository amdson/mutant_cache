import pandas as pd
import pytest
from mutant_cache.data.stabilityarc import assay_frame

def test_canonical_reconstruction_and_depth():
    f=pd.DataFrame({'mutant':['C2A:A1G','D3A'], 'mutated_sequence':['GAD','ACA'], 'DMS_score':[1.,-2.]})
    out=assay_frame(f,'ACD')
    assert out.mutant.tolist()==['A1G:C2A','D3A']
    assert out.n_mut.tolist()==[2,1]

@pytest.mark.parametrize('change',[{'mutant':'G2A'},{'mutated_sequence':'AAA'},{'DMS_score':float('nan')}])
def test_bad_rows_rejected(change):
    row={'mutant':'C2A','mutated_sequence':'AAD','DMS_score':1.}
    row.update(change)
    with pytest.raises(ValueError):assay_frame(pd.DataFrame([row]),'ACD')

def test_equivalent_duplicate_rows_rejected():
    f=pd.DataFrame({'mutant':['C2A:A1G','A1G:C2A'],'mutated_sequence':['GAD','GAD'],'DMS_score':[1.,2.]})
    with pytest.raises(ValueError,match='duplicate'):assay_frame(f,'ACD')

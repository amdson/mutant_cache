import json
import pandas as pd
import pytest
from mutant_cache.matched_suite import run


def fixture_config(tmp_path):
    rows = [dict(target_id=t, mutant=f'A{i}C', n_mut=1, score=i/4, pred=i/4)
            for t in ['001', '002'] for i in range(1, 5)]
    frame = pd.DataFrame(rows)
    frame.drop(columns='pred').to_csv(tmp_path/'labels.csv', index=False)
    frame.to_csv(tmp_path/'baseline.csv', index=False)
    frame.assign(seed=0).to_csv(tmp_path/'ours.csv', index=False)
    config = dict(panels=[dict(id='panel', labels='labels.csv', baseline='baseline.csv',
        expected_rows=8, expected={'all': {'spearman': 1.}}, expected_source='fixture')],
        models=[dict(id='model', fits=[dict(seed=0, predictions={'panel':'ours.csv'})])])
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    return path


def test_matched_suite_end_to_end(tmp_path):
    config = fixture_config(tmp_path)
    output = tmp_path/'report'
    status = run(config, output)
    assert status['status'] == 'complete'
    assert status['comparisons'][0]['reproduction']['passed']
    assert not status['comparisons'][0]['superiority_claim_eligible']
    head = pd.read_csv(output/'headline.csv')
    assert len(head) == 2 and head.n.eq(8).all()
    assert head.spearman.eq(1.).all()
    joined = pd.read_csv(output/'reports/model/panel/joined_seed0.csv', dtype={'target_id':str})
    assert set(joined.target_id) == {'001','002'}
    with pytest.raises(ValueError, match='empty'):
        run(config, output)


def test_refuses_partial_model_or_comparator_coverage(tmp_path):
    config = fixture_config(tmp_path)
    ours = tmp_path/'ours.csv'
    frame = pd.read_csv(ours, dtype={'target_id':str})
    frame.iloc[:-1].to_csv(ours, index=False)
    with pytest.raises(ValueError, match='missing or extra'):
        run(config, tmp_path/'report', validate_only=True)
    frame.to_csv(ours, index=False)
    baseline = tmp_path/'baseline.csv'
    pd.read_csv(baseline).iloc[:-1].to_csv(baseline,index=False)
    with pytest.raises(ValueError, match='comparator'):
        run(config, tmp_path/'report', validate_only=True)


def test_refuses_changed_labels(tmp_path):
    config = fixture_config(tmp_path)
    path = tmp_path/'ours.csv'
    frame = pd.read_csv(path, dtype={'target_id':str})
    frame.loc[0,'score'] = -10
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError, match='labels disagree'):
        run(config,tmp_path/'report',validate_only=True)

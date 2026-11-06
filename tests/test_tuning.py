import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from mutant_cache.tuning import claim, promote, resolve, write_json


def test_initialize_freezes_plan_data_source_and_explicit_seeds(tmp_path):
    import torch
    from fixtures import make_synthetic_dataset
    from mutant_cache.megascale import build_parser
    from mutant_cache.tuning import initialize
    data,_=make_synthetic_dataset(tmp_path/'data',n_targets=5,seq_len=24,
        n_singles=8,n_doubles=8,n_triples=0)
    args=build_parser().parse_args(['--data-dir',str(data),'--split','protein'])
    torch.save({'args':vars(args)},tmp_path/'template.pt')
    plan=dict(base_overrides={},stages=[dict(id='A')],trials=[
        dict(id='A00',stage='A',seed=1,question='explicit seed',overrides={})])
    write_json(tmp_path/'plan.json',plan)
    initialize(tmp_path/'queue',tmp_path/'plan.json',tmp_path/'template.pt')
    state=json.loads((tmp_path/'queue/registry.json').read_text())
    assert state['trials'][0]['seed']==1
    assert not state['template_weights_loaded']
    assert (tmp_path/'queue/source_snapshot/mutant_cache/tuning.py').exists()
    assert len(json.loads((tmp_path/'queue/data_provenance.json').read_text()))==3
    assert (tmp_path/'queue/STATUS.md').exists()
    tracking = dict(wandb_mode='offline', wandb_project='mutant_cache', wandb_entity='test-team',
        diagnostics=True, diagnostic_probe_rows=32)
    write_json(tmp_path/'tracking.json',tracking)
    initialize(tmp_path/'tracked-queue',tmp_path/'plan.json',tmp_path/'template.pt',tmp_path/'tracking.json')
    state=json.loads((tmp_path/'tracked-queue/registry.json').read_text())
    assert state['base_args']['wandb_mode']=='offline'
    assert state['base_args']['wandb_entity']=='test-team'
    assert state['base_args']['diagnostics'] and state['base_args']['diagnostic_probe_rows']==32
    assert json.loads((tmp_path/'tracked-queue/tracking_config.json').read_text())==tracking
    assert len(state['tracking_config_sha256'])==64


def registry(tmp_path):
    plan = dict(stages=[dict(id='A'),dict(id='B',parent_stage='A')],trials=[])
    state = dict(base_args=dict(lr=.001,load_trainable='must be cleared'),promotions={},trials=[
        dict(id='A00',stage='A',seed=0,status='planned',attempts=0,overrides={}),
        dict(id='A01',stage='A',seed=0,status='planned',attempts=0,overrides=dict(lr=.0003)),
        dict(id='B01',stage='B',seed=0,status='planned',attempts=0,overrides=dict(loss='huber_dg'))])
    write_json(tmp_path/'plan.json',plan);write_json(tmp_path/'registry.json',state)
    return state,plan


def test_concurrent_claims_are_unique_and_stage_waits(tmp_path):
    registry(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        jobs = list(pool.map(lambda gpu:claim(tmp_path,gpu),[0,2]))
    assert {j['id'] for j in jobs} == {'A00','A01'}
    assert all(j['resolved_args']['load_trainable'] is None for j in jobs)
    for job in jobs:
        context = job['resolved_args']['experiment_context']
        assert context['trial_id'] == job['id'] and context['stage'] == 'A'
        assert context['attempt'] == 1 and context['sweep'] == tmp_path.name
    assert claim(tmp_path,1) is None  # stage B cannot use partial stage A results


def test_validation_promotion_carries_control_and_resolves_fresh(tmp_path):
    state,plan=registry(tmp_path)
    for t,rho in zip(state['trials'][:2],[.7,.6]):
        t.update(status='completed',resolved_args=dict(lr=.001),metrics=dict(val_spearman=rho,val_double_rmse=1.),
                 external_spearman=100 if rho==.6 else -100)
    promote(state,plan,tmp_path)
    assert state['promotions']['A']=='A00' # external scores have no effect
    b=state['trials'][2]
    args=resolve(state,plan,b)
    assert args['lr']==.001 and args['loss']=='huber_dg' and args['load_trainable'] is None
    b.update(status='completed',resolved_args=args,metrics=dict(val_spearman=.5,val_double_rmse=.9))
    promote(state,plan,tmp_path)
    assert state['promotions']['B']=='A00' # treatment cannot displace a better control


def test_continuation_preserves_finished_results_and_instruments_imported_parent(tmp_path):
    import torch
    from fixtures import make_synthetic_dataset
    from mutant_cache.megascale import build_parser
    from mutant_cache.tuning import initialize, initialize_continuation, score_result
    data,_=make_synthetic_dataset(tmp_path/'data',n_targets=5,seq_len=24,
        n_singles=8,n_doubles=8,n_triples=0)
    args=build_parser().parse_args(['--data-dir',str(data),'--split','protein'])
    torch.save({'args':vars(args)},tmp_path/'template.pt')
    plan=dict(base_overrides=dict(lr=.001),stages=[dict(id='A'),dict(id='B',parent_stage='A')],trials=[
        dict(id='A00',stage='A',question='control',overrides={}),
        dict(id='B01',stage='B',question='Huber',overrides=dict(loss='huber_dg'))])
    write_json(tmp_path/'plan.json',plan)
    source=tmp_path/'original'
    initialize(source,tmp_path/'plan.json',tmp_path/'template.pt')
    state=json.loads((source/'registry.json').read_text())
    complete=state['trials'][0]
    directory=source/'runs/A00/attempt_1'
    directory.mkdir(parents=True)
    result=dict(extra=dict(val_summary=[dict(depth=d,metric=m,macro=value) for d,m,value in
        [('all','spearman',.7),('double','spearman',.6),('single','spearman',.8),('double','rmse',1.)]],
        selection=dict(step=10)))
    write_json(directory/'results.json',result)
    (directory/'model.pt').write_bytes(b'finished-checkpoint')
    complete.update(status='completed',attempts=1,run_dir=str(directory),resolved_args=state['base_args'],
        metrics=score_result(result))
    partial=source/'runs/B01/attempt_1'
    partial.mkdir(parents=True)
    (partial/'partial.pt').write_bytes(b'partial-checkpoint')
    state['trials'][1].update(status='interrupted',attempts=1,run_dir=str(partial))
    write_json(source/'registry.json',state)
    write_json(source/'STOPPED.json',dict(archive='saved'))
    before=(source/'registry.json').read_bytes()
    write_json(tmp_path/'tracking.json',dict(wandb_mode='offline',diagnostics=True,diagnostic_log_every=50))
    target=tmp_path/'continuation'
    initialize_continuation(target,source,tmp_path/'tracking.json')
    continued=json.loads((target/'registry.json').read_text())
    assert continued['promotions']=={'A':'A00'}
    assert continued['trials'][0]['metrics']==complete['metrics']
    assert continued['trials'][0]['run_dir']==str(directory)
    assert continued['trials'][0]['resolved_args']['diagnostics'] is False
    assert continued['trials'][1]['prior_attempts'][0]['run_dir']==str(partial)
    assert (source/'registry.json').read_bytes()==before
    assert (partial/'partial.pt').read_bytes()==b'partial-checkpoint'
    job=claim(target,0)
    assert job['id']=='B01' and job['attempts']==1
    assert job['resolved_args']['lr']==.001 and job['resolved_args']['loss']=='huber_dg'
    assert job['resolved_args']['wandb_mode']=='offline' and job['resolved_args']['diagnostics'] is True
    assert job['resolved_args']['load_trainable'] is None
    assert job['resolved_args']['output'].startswith(str(target))
    assert json.loads((target/'plan.json').read_text())==plan
    with pytest.raises(ValueError,match='already exists'):
        initialize_continuation(target,source,tmp_path/'tracking.json')


def test_continuation_rejects_an_unarchived_source(tmp_path):
    from mutant_cache.tuning import initialize_continuation
    with pytest.raises(ValueError,match='stopped and archived'):
        initialize_continuation(tmp_path/'new',tmp_path/'old',tmp_path/'tracking.json')


@pytest.mark.parametrize('mode',['huber','reverse','background','shared_head'])
def test_tuning_training_controls(tmp_path,mode):
    from fixtures import make_synthetic_dataset
    from mutant_cache import megascale
    d,_=make_synthetic_dataset(tmp_path/'data',n_targets=5,seq_len=24,n_singles=40,n_doubles=40,n_triples=0)
    extra = {
        'huber':['--loss','huber_dg','--task-reduction','depth','--single-weight','.1',
                 '--stabilizing-double-weight','4','--double-subsample-ratio','8','--single-only-exposures','.5'],
        'reverse':['--reference-augmentation','reverse','--augmentation-fraction','.25'],
        'background':['--reference-augmentation','background','--augmentation-fraction','1'],
        'shared_head':['--head','wt_relative','--shared-lora-r','2']
    }[mode]
    args=megascale.build_parser().parse_args(['--data-dir',str(d),'--dataset','synthetic','--split','protein',
        '--backbone','random:64:4:2','--device','cpu','--precision','fp32','--lora-r','2',
        '--exposures','1','--micro-batch','16','--proteins-per-step','2','--eval-min-n','3',
        '--selection','spearman','--save-model','--save-selection-rules','double_rmse','double_spearman','ndcg30',
        '--output',str(tmp_path/'run'),*extra])
    result=megascale.run(args)
    assert result['status']=='ok'
    assert not (tmp_path/'run/predictions_test.csv').exists()
    assert (tmp_path/'run/checkpoints/spearman.pt').exists()
    assert (tmp_path/'run/checkpoints/double_rmse.pt').exists()
    if mode in ('reverse','background'):
        assert result['extra']['training_protocol']['augmented_rows']>0
    if mode=='huber':
        assert result['extra']['training_protocol']['single_only_steps']>0

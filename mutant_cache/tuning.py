"""A persistent, validation-only staged experiment queue.

The JSON plan is frozen at initialization. Workers run isolated subprocesses
from a source snapshot, claim trials under a file lock, and publish Markdown
and CSV summaries. No external predictions or benchmark scores are consulted.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


@contextlib.contextmanager
def locked(root):
    root = Path(root)
    with (root/'registry.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads((root/'registry.json').read_text())
        yield state
        write_json(root/'registry.json', state)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        stat = Path(f'/proc/{pid}/stat')
        return not stat.exists() or stat.read_text().split()[2] != 'Z'
    except ProcessLookupError:
        return False


def initialize(root, plan, template, tracking_config=None):
    import torch
    root, plan = Path(root), Path(plan)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'registry.json').exists():
        raise ValueError(f'{root}: experiment registry already exists')
    spec = json.loads(plan.read_text())
    ids = [r['id'] for r in spec['trials']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate trial IDs')
    stages = {s['id'] for s in spec['stages']}
    for trial in spec['trials']:
        if trial['stage'] not in stages:
            raise ValueError(f'unknown stage {trial["stage"]}')
    base = torch.load(template, map_location='cpu', weights_only=False)['args']
    base.update(spec['base_overrides'])
    tracking_settings = None
    if tracking_config is not None:
        from mutant_cache.tracking import load_tracking_config
        tracking_settings = load_tracking_config(tracking_config)
        base.update(tracking_settings)
    base.update(final_test=False, benchmark_inference=False, load_trainable=None,
                max_train_variants=0, max_val_proteins=0)
    source = Path(__file__).resolve().parents[1]
    snapshot = root/'source_snapshot'
    shutil.copytree(source/'mutant_cache', snapshot/'mutant_cache',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(plan, root/'plan.json')
    if tracking_settings is not None:
        write_json(root/'tracking_config.json', tracking_settings)
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob('*.py'))}
    write_json(root/'source_sha256.json', hashes)
    from mutant_cache.data.paths import scratch
    from mutant_cache.data.mutate_everything import sha256
    dataset = Path(base.get('data_dir') or scratch()/'processed'/base['dataset'])
    split = dataset/'splits'/f'{base["split"]}.parquet'
    write_json(root/'data_provenance.json', {str(p): sha256(p) for p in
        (dataset/'variants.parquet', dataset/'targets.parquet', split)})
    state = dict(created_utc=now(), base_args=base, template_args_source=str(template),
        template_weights_loaded=False, promotions={}, plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        tracking_config_sha256=None if tracking_settings is None else
            hashlib.sha256((root/'tracking_config.json').read_bytes()).hexdigest(),
        trials=[dict(t, status='planned', attempts=0, seed=t.get('seed', 0)) for t in spec['trials']])
    write_json(root/'registry.json', state)
    refresh(root)


def metric(result, depth, name):
    for row in result['extra']['val_summary']:
        if row['depth'] == depth and row['metric'] == name:
            return float(row['macro'])
    return float('nan')


def score_result(result):
    rho = metric(result, 'all', 'spearman')
    rmse = metric(result, 'double', 'rmse')
    if not math.isfinite(rho) or not math.isfinite(rmse):
        raise ValueError('selection metrics must be finite')
    return dict(val_spearman=rho, val_double_spearman=metric(result,'double','spearman'),
                val_single_spearman=metric(result,'single','spearman'), val_double_rmse=rmse,
                selected_step=result['extra']['selection']['step'])


def promote(state, spec, root):
    by_id = {t['id']: t for t in state['trials']}
    for stage in spec['stages']:
        sid = stage['id']
        if sid in state['promotions'] or sid == 'G':
            continue
        jobs = [t for t in state['trials'] if t['stage'] == sid]
        parent = stage.get('parent_stage')
        if parent and parent not in state['promotions']:
            continue
        if any(t['status'] not in ('completed','failed','alias') for t in jobs):
            continue
        candidates = [t for t in jobs if t['status'] == 'completed']
        if parent:
            candidates.append(by_id[state['promotions'][parent]])
        if not candidates:
            continue
        best = min(candidates, key=lambda t:(-t['metrics']['val_spearman'],t['metrics']['val_double_rmse'],t['id']))
        state['promotions'][sid] = best['id']
    screening = [s['id'] for s in spec['stages'] if s['id'] != 'G']
    if all(s in state['promotions'] for s in screening) and 'selected_recipe' not in state:
        candidates = [by_id[i] for i in set(state['promotions'].values())]
        best = min(candidates,key=lambda t:(-t['metrics']['val_spearman'],t['metrics']['val_double_rmse'],t['id']))
        state['selected_recipe'] = best['id']
        write_json(Path(root)/'frozen_recipe.json',dict(selected_utc=now(),trial=best['id'],
            args=best['resolved_args'],criterion='validation all-depth macro Spearman, then double RMSE',
            external_results_used=False,confirmation_seeds=[1,2]))


def resolve(state, spec, trial):
    by_id = {t['id']:t for t in state['trials']}
    stage = next(s for s in spec['stages'] if s['id'] == trial['stage'])
    parent = None
    if trial['stage'] == 'G':
        if 'selected_recipe' not in state:
            return None
        parent = 'A00' if trial['role'] == 'control' else state['selected_recipe']
        if trial['role'] == 'selected' and parent == 'A00':
            trial.update(status='alias',alias_of=f'G_control_s{trial["seed"]}')
            return None
    elif stage.get('parent_stage'):
        parent = state['promotions'].get(stage['parent_stage'])
        if parent is None:
            return None
    args = dict(by_id[parent]['resolved_args'] if parent else state['base_args'])
    args.update(trial.get('overrides', {}))
    args.update(seed=trial['seed'], final_test=False, benchmark_inference=False,load_trainable=None)
    trial['parent_id'] = parent or 'frozen pilot recipe'
    return args


def claim(root, gpu):
    root = Path(root)
    spec = json.loads((root/'plan.json').read_text())
    with locked(root) as state:
        if state.get('plan_sha256') and hashlib.sha256((root/'plan.json').read_bytes()).hexdigest() != state['plan_sha256']:
            raise ValueError('frozen experiment plan changed; initialize a new version')
        for t in state['trials']:
            if t['status'] == 'running' and not alive(t.get('worker_pid')) and not alive(t.get('child_pid')):
                result_path=Path(t['run_dir'])/'results.json'
                if result_path.exists():
                    t.update(status='completed',metrics=score_result(json.loads(result_path.read_text())),finished_utc=now())
                else:
                    t.update(status='planned' if t['attempts'] < 2 else 'failed',error='worker/child exited without recording completion')
        # An orphaned live child on this lane must finish before another job.
        if any(t['status']=='running' and t.get('gpu')==gpu for t in state['trials']):
            return None
        promote(state, spec, root)
        for t in state['trials']:
            if t['status'] != 'planned':
                continue
            args = resolve(state,spec,t)
            if args is None:
                continue
            t['attempts'] += 1
            run_dir = root/'runs'/t['id']/f'attempt_{t["attempts"]}'
            args.update(output=str(run_dir),out_root=str(root/'runs'),device='cuda:0',tag=t['id'])
            args['experiment_context'] = dict(sweep=root.name, trial_id=t['id'], stage=t['stage'],
                parent=t['parent_id'], attempt=t['attempts'], seed=t['seed'],
                run_root=str(root), plan_sha256=state.get('plan_sha256'),
                tracking_config_sha256=state.get('tracking_config_sha256'))
            t.update(status='running',gpu=gpu,worker_pid=os.getpid(),child_pid=None,
                     started_utc=now(),run_dir=str(run_dir),resolved_args=args)
            return dict(t)
    return None


def run_trial(root, trial_id):
    from mutant_cache.megascale import build_parser, run
    root = Path(root)
    from mutant_cache.data.mutate_everything import sha256
    for filename, expected in json.loads((root/'data_provenance.json').read_text()).items():
        if sha256(filename) != expected:
            raise ValueError(f'frozen training data changed: {filename}')
    state = json.loads((root/'registry.json').read_text())
    trial = next(t for t in state['trials'] if t['id']==trial_id)
    args = build_parser().parse_args([])
    vars(args).update(trial['resolved_args'])
    args.experiment_provenance = {name: json.loads((root/name).read_text())
        for name in ('data_provenance.json', 'source_sha256.json', 'tracking_config.json')
        if (root/name).exists()}
    if args.final_test or args.load_trainable:
        raise ValueError('tuning trials must start fresh and remain validation-only')
    result = run(args)
    if result['extra']['test_summary'] is not None or (Path(args.output)/'predictions_test.csv').exists():
        raise RuntimeError('external scoring is forbidden in a tuning trial')


def worker(root, gpu):
    root = Path(root)
    with (root/f'gpu_{gpu}.lock').open('a') as lane:
        fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            trial = claim(root,gpu)
            refresh(root)
            if trial is None:
                with locked(root) as state:
                    pending = any(t['status'] in ('planned','running') for t in state['trials'])
                if not pending:
                    print('queue complete',flush=True)
                    return
                time.sleep(10)
                continue
            run_dir = Path(trial['run_dir'])
            run_dir.mkdir(parents=True,exist_ok=True)
            write_json(run_dir/'resolved_experiment.json',trial)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',
                       PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
            command = [sys.executable,'-m','mutant_cache.tuning','run-trial',
                       '--root',str(root),'--id',trial['id']]
            print(now(),'START',trial['id'],'GPU',gpu,flush=True)
            with (run_dir/'train.log').open('w') as log:
                child = subprocess.Popen(command,cwd=root/'source_snapshot',env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                with locked(root) as state:
                    next(t for t in state['trials'] if t['id']==trial['id'])['child_pid']=child.pid
                while child.poll() is None:
                    refresh(root)
                    time.sleep(10)
            with locked(root) as state:
                current = next(t for t in state['trials'] if t['id']==trial['id'])
                try:
                    if child.returncode:
                        raise RuntimeError(f'process exit {child.returncode}; see {run_dir}/train.log')
                    result = json.loads((run_dir/'results.json').read_text())
                    current.update(status='completed',metrics=score_result(result),finished_utc=now())
                except Exception as exc:
                    current.update(status='planned' if current['attempts']<2 else 'failed',error=str(exc))
                promote(state,json.loads((root/'plan.json').read_text()),root)
            refresh(root)


def refresh(root):
    root = Path(root)
    # Rendering while holding the lock prevents concurrent workers from
    # publishing a stale report over newer state.
    with locked(root) as state:
        spec = json.loads((root/'plan.json').read_text())
        rows, checkpoint_rows = [], []
        for t in state['trials']:
            args = t.get('resolved_args',t.get('overrides',{}))
            metrics = t.get('metrics',{})
            progress = {}
            tracking = {}
            if t.get('run_dir'):
                path = Path(t['run_dir'])/'tracking.json'
                if path.exists():
                    try: tracking=json.loads(path.read_text())
                    except json.JSONDecodeError: pass
                path = Path(t['run_dir'])/'progress.json'
                if path.exists():
                    try: progress=json.loads(path.read_text())
                    except json.JSONDecodeError: pass
                selections = Path(t['run_dir'])/'selection_checkpoints.json'
                history = Path(t['run_dir'])/'history.csv'
                if selections.exists() and history.exists():
                    try:
                        choices=json.loads(selections.read_text())
                        with history.open() as f: trajectory=list(csv.DictReader(f))
                        for rule, selected in choices.items():
                            row=next(r for r in trajectory if int(r['step'])==selected['step'])
                            checkpoint_rows.append(dict(id=t['id'],rule=rule,step=selected['step'],
                                val_spearman=row['val_all_spearman'],val_double_spearman=row['val_double_spearman'],
                                val_double_rmse=row['val_double_rmse'],val_ndcg30=row['val_ndcg30_double'],
                                checkpoint=str(Path(t['run_dir'])/'checkpoints'/f'{rule}.pt')))
                    except (json.JSONDecodeError,StopIteration,KeyError,ValueError):
                        pass # a live file may be between the two updates
            rows.append(dict(id=t['id'],stage=t['stage'],question=t['question'],
                status=t['status'],seed=t['seed'],parent=t.get('parent_id',''),gpu=t.get('gpu',''),
                step=progress.get('step',''),head=args.get('head',''),loss=args.get('loss',''),
                head_lr=args.get('lr',''),adapter_lr=args.get('lora_lr',''),
                lora_rank=args.get('lora_r',''),shared_rank=args.get('shared_lora_r',''),
                exposures=args.get('exposures',''),single_only_exposures=args.get('single_only_exposures',''),
                single_weight=args.get('single_weight',''),double_positive_weight=args.get('stabilizing_double_weight',''),
                double_subsample_ratio=args.get('double_subsample_ratio',''),task_reduction=args.get('task_reduction',''),
                reference_augmentation=args.get('reference_augmentation',''),
                **metrics,wandb_url=tracking.get('url') or '',
                run_dir=t.get('run_dir',''),error=t.get('error','')))
        columns=list(dict.fromkeys(k for row in rows for k in row))
        temp=root/f'summary.{os.getpid()}.tmp'
        with temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        temp.replace(root/'summary.csv')
        if checkpoint_rows:
            temp=root/f'checkpoint_choices.{os.getpid()}.tmp'
            with temp.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(checkpoint_rows[0]))
                writer.writeheader();writer.writerows(checkpoint_rows)
            temp.replace(root/'checkpoint_choices.csv')
        lines=['# Mutate Everything tuning — live status','',f'Updated {now()}.',
            '', 'Selection uses internal validation only. All trials save separate RMSE, Spearman, '
            'double-Spearman and nDCG@30 checkpoints. Screening seed is 0; confirmation seeds are 1/2.',
            '', '| ID | Study | Status | Seed | GPU | Step | Val rho | Double RMSE | Artifacts |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in rows:
            rho=row.get('val_spearman');rmse=row.get('val_double_rmse')
            artifacts=f'[run]({row["run_dir"]}) · [log]({row["run_dir"]}/train.log)' if row['run_dir'] else ''
            if row['wandb_url']:
                artifacts += f' · [W&B]({row["wandb_url"]})'
            lines.append(f'| {row["id"]} | {row["question"]} | {row["status"]} | {row["seed"]} | '
                f'{row["gpu"]} | {row["step"]} | {rho:.4f} | {rmse:.4f} | {artifacts} |' if rho is not None else
                f'| {row["id"]} | {row["question"]} | {row["status"]} | {row["seed"]} | '
                f'{row["gpu"]} | {row["step"]} | — | — | {artifacts} |')
        lines += ['', '## Promoted controls', '']
        for stage, trial in state['promotions'].items():
            lines.append(f'- {stage}: {trial}')
        lines += ['', f'[Compare checkpoint-selection rules]({root}/checkpoint_choices.csv) · '
                  f'[Full configuration/results CSV]({root}/summary.csv)', '',
                  'Each stage includes its inherited control as a candidate; a new treatment '
                  'is carried forward only if its validation score improves. Adapter and augmentation '
                  'studies branch from the same head-study control. No external benchmark is run by this queue.',
                  '', f'Raw registry: `{root}/registry.json`; configurations: `runs/*/attempt_*/resolved_experiment.json`.']
        report='\n'.join(lines)+'\n'
        path=root/'STATUS.md';temporary=path.with_suffix('.md.tmp');temporary.write_text(report);temporary.replace(path)
        destination=spec.get('report_path')
        if destination:
            path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.tmp');temporary.write_text(report);temporary.replace(path)
        if not any(t['status'] in ('running','planned') for t in state['trials']):
            write_json(root/'complete.json',dict(completed_utc=now(),failed=[t['id'] for t in state['trials'] if t['status']=='failed'],
                selected_recipe=state.get('selected_recipe'),external_results_used=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init');init.add_argument('--root',required=True,type=Path)
    init.add_argument('--plan',required=True,type=Path);init.add_argument('--template',required=True,type=Path)
    init.add_argument('--tracking-config',type=Path,
        help='optional W&B settings JSON, frozen alongside the experiment plan')
    work=sub.add_parser('worker');work.add_argument('--root',required=True,type=Path);work.add_argument('--gpu',required=True,type=int)
    trial=sub.add_parser('run-trial');trial.add_argument('--root',required=True,type=Path);trial.add_argument('--id',required=True)
    report=sub.add_parser('report');report.add_argument('--root',required=True,type=Path)
    a=parser.parse_args()
    if a.command=='init': initialize(a.root,a.plan,a.template,a.tracking_config)
    elif a.command=='worker': worker(a.root,a.gpu)
    elif a.command=='run-trial': run_trial(a.root,a.id)
    else: refresh(a.root)


if __name__=='__main__':
    main()

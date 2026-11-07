"""Count-reconciled ProteinGym inputs for the StabilityArc seven-fold pilot.

The private upstream packed training data are not available. Match the public
66-assay inventory and counts, preserve every ProteinGym row, and audit sequence
reconstruction. This does not claim to reproduce upstream training preprocessing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

from .mutations import parse_mutant, format_mutant

HOLDOUTS = ['BCHB_CHLTE', 'CBX4_HUMAN', 'SPTN1_CHICK', 'SQSTM_MOUSE',
            'DN7A_SACS2', 'ESTA_BACSU', 'A4_HUMAN']
REVISION = '94fc12dc9b73e48e48e818a489155feccafea94a'


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def assay_frame(frame, sequence):
    """Validate canonical substitutions and every provided mutant sequence."""
    records = []
    for r in frame.itertuples(index=False):
        subs = parse_mutant(r.mutant)
        if len(subs) not in (1, 2) or len({p for p, _, _ in subs}) != len(subs):
            raise ValueError(f'invalid mutation depth/positions: {r.mutant}')
        mutated = list(sequence)
        for p, wt, alt in subs:
            if not 1 <= p <= len(sequence) or sequence[p-1] != wt or wt == alt:
                raise ValueError(f'reference mismatch: {r.mutant}')
            mutated[p-1] = alt
        if ''.join(mutated) != r.mutated_sequence or not np.isfinite(r.DMS_score):
            raise ValueError(f'mutant sequence or label mismatch: {r.mutant}')
        records.append({'mutant': format_mutant(subs), 'n_mut': len(subs), 'score': float(r.DMS_score)})
    out = pd.DataFrame(records)
    if out.mutant.duplicated().any():
        raise ValueError('duplicate canonical mutations')
    return out


def build(scratch, *, remaining=False):
    scratch = Path(scratch)
    archive = scratch/'raw/proteingym_v1_3/DMS_ProteinGym_substitutions.zip'
    reference = scratch/'raw/proteingym/DMS_substitutions.csv'
    published = scratch/'stabilityarc/upstream/results/multimutant_additive_summary.json'
    inventory = json.loads(published.read_text())['results']
    holdouts = sorted(r['protein_id'] for r in inventory if r['protein_id'] not in HOLDOUTS) if remaining else HOLDOUTS
    metadata = pd.read_csv(reference).set_index('DMS_id')
    tables, sequences, stats = {}, {}, []
    with zipfile.ZipFile(archive) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith('.csv')}
        for item in inventory:
            aid, protein = item['assay_id'], item['protein_id']
            seq = metadata.loc[aid, 'target_seq']
            frame = assay_frame(pd.read_csv(z.open(members[aid])), seq)
            if len(frame) != item['num_scored_variants'] or int((frame.n_mut == 1).sum()) != item['num_single_mutants'] or int((frame.n_mut == 2).sum()) != item['num_multi_mutants']:
                raise ValueError(f'published counts differ: {aid}')
            if protein in tables:
                raise ValueError('multiple assays per protein require explicit grouping')
            frame = frame.assign(score_set_id=aid, target_id=protein,
                                 score_raw=frame.score, assay_type='stability')
            frame['score_rank'] = frame.score.rank(method='average')/(len(frame)+1)
            frame['score_se'] = np.nan
            tables[protein], sequences[protein] = frame, seq
            stats.append(dict(protein_id=protein, assay_id=aid, length=len(seq),
                              rows=len(frame), singles=int((frame.n_mut==1).sum()),
                              doubles=int((frame.n_mut==2).sum())))
    if len(tables) != 66 or sum(len(f) for f in tables.values()) != 134794:
        raise ValueError('unexpected corpus inventory')
    # Fail on cross-name exact reference duplication; do not alter outer folds.
    if len(set(sequences.values())) != len(sequences):
        raise ValueError('identical references assigned different proteins')
    output = scratch/'stabilityarc'/('full66_v1' if remaining else 'pilot_v1')
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(stats).to_csv(output/'assay_inventory.csv', index=False)
    manifest = dict(upstream_revision=REVISION, inputs={str(p): sha(p) for p in [archive, reference, published]},
                    counts_match_all_66=True, total_rows=134794, holdouts=holdouts,
                    comparison='published per-assay summary; row-level upstream predictions unavailable',
                    normalization='Training/inner-validation proteins only: subtract singles median, divide singles MAD (floor 1e-6), same affine transform for all depths. Outer test labels remain raw DMS_score. Test RMSE is not comparable across these scales.',
                    training_cap='all assays below 40000 rows; no sampling cap removes rows', folds=[])
    targets = pd.DataFrame([dict(target_id=p, sequence=s, uniprot=p, source='ProteinGym_v1.3') for p,s in sequences.items()])
    for index, heldout in enumerate(holdouts):
        name = f'stabilityarc_{"full66" if remaining else "pilot"}_{heldout.lower()}'
        dest = scratch/'processed'/name
        dest.mkdir(parents=True, exist_ok=False)
        (dest/'splits').mkdir()
        others = sorted(set(tables)-{heldout})
        rng = np.random.default_rng(23)
        validation = set(rng.choice(others, size=6, replace=False))
        split = pd.DataFrame([dict(target_id=p, split='test' if p==heldout else 'val' if p in validation else 'train') for p in sorted(tables)])
        frames, norms = [], {}
        for protein, original in tables.items():
            frame = original.copy()
            if protein != heldout:
                singles = frame.loc[frame.n_mut==1, 'score']
                center = float(singles.median()); scale=max(float((singles-center).abs().median()),1e-6)
                frame['score'] = (frame.score-center)/scale
                norms[protein] = dict(center=center, scale=scale)
            frames.append(frame)
        targets.to_parquet(dest/'targets.parquet', index=False)
        pd.concat(frames, ignore_index=True).to_parquet(dest/'variants.parquet', index=False)
        sets = pd.DataFrame([dict(score_set_id=f.score_set_id.iloc[0], target_id=p, n_variants=len(f), max_n_mut=int(f.n_mut.max()), assay_type='stability', reliability=None, license='See ProteinGym source terms') for p,f in tables.items()])
        sets.to_parquet(dest/'score_sets.parquet', index=False)
        split.to_parquet(dest/'splits/outer.parquet', index=False)
        (dest/'normalization.json').write_text(json.dumps(norms,indent=2))
        manifest['folds'].append(dict(heldout=heldout, dataset=name, split='outer',
                                     inner_validation=sorted(validation), training_proteins_after_refit=65,
                                     test_rows=len(tables[heldout]), published=next(r for r in inventory if r['protein_id']==heldout)))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch', required=True)
    p.add_argument('--remaining', action='store_true', help='build the other 59 folds in separate directories')
    args=p.parse_args()
    print(json.dumps(build(args.scratch, remaining=args.remaining),indent=2))

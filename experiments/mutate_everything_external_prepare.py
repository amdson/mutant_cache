"""Prepare released external tests and explicit reverse references; no filtering by results."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pandas as pd
from mutant_cache.data.mutate_everything import normalize
from mutant_cache.data.mutations import parse_mutant, format_mutant, mutate

scratch = Path('/home/andrew/mutant_cache_scratch')
root = scratch / 'mutate_everything'
out = root / 'external'
out.mkdir(exist_ok=True)
targets = pd.read_parquet(scratch / 'processed/mutate_everything_cdna/targets.parquet')
split = pd.read_parquet(scratch / 'processed/mutate_everything_cdna/splits/mutate_everything_cdna.parquet')
training = targets.merge(split, on='target_id').query("split == 'train'")
inventory = {}
with zipfile.ZipFile(root / 'mutate_everything_data.zip') as z:
    for tag, base in [('s669', 's669.csv'), ('ptmul', 'protherm_multiple.csv'), ('cdna1', 'cdna1_test.csv')]:
        name = next(n for n in z.namelist() if n.endswith('/'+base) and '__MACOSX' not in n)
        raw_bytes = z.read(name)
        (out / base).write_bytes(raw_bytes)
        raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
        frame, seqs, audit = normalize(raw)
        frame['wt_seq'] = frame.target_id.map(seqs)
        frame['reference_id'] = frame.target_id
        frame.to_csv(out / f'{tag}_labels.csv', index=False)
        audit.to_csv(out / f'{tag}_exclusions.csv', index=False)
        overlaps = []
        for tid in frame.target_id.unique():
            for r in training.itertuples():
                if seqs[tid] in r.sequence or r.sequence in seqs[tid]:
                    overlaps.append(dict(external_target=tid, train_target=r.target_id,
                                         exact=seqs[tid] == r.sequence))
        inventory[tag] = dict(raw_rows=len(raw), rows=len(frame), proteins=int(frame.target_id.nunique()),
                              depths=frame.n_mut.value_counts().sort_index().to_dict(),
                              raw_sha256=hashlib.sha256(raw_bytes).hexdigest(), excluded_rows=len(audit),
                              exact_or_containment_overlaps=overlaps, homology_audit='not yet recomputed')
        if tag == 's669':
            reverse = frame.copy()
            reverse['forward_mutant'] = frame.mutant
            reverse['wt_seq'] = [mutate(s, m) for s, m in zip(frame.wt_seq, frame.mutant)]
            reverse['reference_id'] = frame.target_id + '::' + frame.mutant
            reverse['mutant'] = [format_mutant([(p, a, w) for p, w, a in parse_mutant(m)]) for m in frame.mutant]
            reverse['score'] = -frame.score
            reverse.to_csv(out / 's669_reverse_labels.csv', index=False)
            for r in reverse.itertuples():
                assert mutate(r.wt_seq, r.mutant) == seqs[r.target_id]
            inventory['s669_reverse'] = dict(rows=len(reverse), proteins=int(reverse.target_id.nunique()),
                unique_references=int(reverse.reference_id.nunique()),
                convention='mutant reference, reverse substitution, negated label; no result-based calibration')
(out / 'inventory.json').write_text(json.dumps(inventory, indent=2))
print(json.dumps(inventory, indent=2), flush=True)

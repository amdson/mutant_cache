"""MMseqs2 audit against all 116 released training proteins; report, never filter tests."""
import json
from pathlib import Path
import subprocess
import pandas as pd

root = Path('/home/andrew/mutant_cache_scratch/mutate_everything')
out = root / 'external/homology'
out.mkdir(exist_ok=True)
train = pd.read_csv('/home/andrew/mutant_cache_scratch/processed/mutate_everything_cdna/comparison_labels_train.csv')
refs = train[['target_id','wt_seq']].drop_duplicates()
with (out/'train.fasta').open('w') as f:
    for r in refs.itertuples(): f.write(f'>{r.target_id}\n{r.wt_seq}\n')
with (out/'queries.fasta').open('w') as f:
    for tag in ['s669','ptmul','cdna1']:
        labels = pd.read_csv(root/f'external/{tag}_labels.csv')
        for r in labels[['target_id','wt_seq']].drop_duplicates().itertuples():
            f.write(f'>{tag}|{r.target_id}\n{r.wt_seq}\n')
command = ['/home/andrew/MMseqs2/build/bin/mmseqs','easy-search',str(out/'queries.fasta'),
    str(out/'train.fasta'),str(out/'alignments.tsv'),str(out/'tmp'),
    '-s','7.5','-c','0.3','--seq-id-mode','1','--threads','4',
    '--format-output','query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits']
with (out/'mmseqs.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
cols=['query','target','pident','alnlen','qlen','tlen','qcov','tcov','evalue','bits']
try:
    hits = pd.read_csv(out/'alignments.tsv',sep='\t',names=cols)
except pd.errors.EmptyDataError:
    hits = pd.DataFrame(columns=cols)
hits.sort_values('pident',ascending=False).drop_duplicates('query').to_csv(out/'best_hits.csv',index=False)
record = dict(command=command, training_proteins=len(refs), hits=len(hits),
    max_reported_identity_percent=float(hits.pident.max()) if len(hits) else None,
    interpretation='sensitivity audit with -c 0.3, default E-value cutoff; absence of hits is not a universal identity bound',
    tests_filtered=False)
(out/'audit.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)

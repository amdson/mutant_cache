# Builds $MUTANT_CACHE_SCRATCH/processed/overfit10: 5 singles + 5 doubles from one
# Tsuboyama train-split domain, spread across the score range. Run as `python -c` originally.
import pandas as pd, numpy as np
from mutant_cache.data.paths import scratch
S=scratch()/'processed'; D=S/'tsuboyama'; O=S/'overfit10'; O.mkdir(exist_ok=True)
v=pd.read_parquet(D/'variants.parquet'); t=pd.read_parquet(D/'targets.parquet'); ss=pd.read_parquet(D/'score_sets.parquet')
sp=pd.read_parquet(S/'splits/protein_level.parquet'); tr=set(sp[sp.split=='train'].target_id)
dbl=v[(v.n_mut==2)&v.target_id.isin(tr)].score_set_id.value_counts()
sid=dbl.index[0]; g=v[(v.score_set_id==sid)&(v.n_mut>0)].dropna(subset=['score'])
pick=[]
for n in (1,2):
    h=g[g.n_mut==n].sort_values('score').reset_index(drop=True)
    pick.append(h.iloc[np.linspace(0,len(h)-1,5).round().astype(int)])
x=pd.concat(pick).reset_index(drop=True)
x.to_parquet(O/'variants.parquet'); t[t.target_id==x.target_id[0]].to_parquet(O/'targets.parquet'); ss[ss.score_set_id==sid].to_parquet(O/'score_sets.parquet')
print(sid, len(t[t.target_id==x.target_id[0]].sequence.iloc[0]),'aa'); print(x[['mutant','n_mut','score','score_rank']].to_string())

# then, on two GPUs (overfit10_final.py = harness with ties -> latest checkpoint):
# COMMON="--data-dir $S/processed/overfit10 --dataset overfit10 --backbone esmc_300m --loss mse_dg --target-col score --batch-size 10 --epochs 300 --lr 3e-3 --dropout 0 --weight-decay 0 --eval-every 50 --baselines none"
# python experiments/overfit10_final.py $COMMON --output $S/runs/overfit10_A_final
# python experiments/overfit10_final.py $COMMON --no-marginals --output $S/runs/overfit10_B_final

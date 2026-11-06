"""Real upstream decoder oracle for singles, doubles and 3–10 substitutions."""
import argparse
from pathlib import Path
import sys
import torch

repo = Path('/home/andrew/mutant_cache_scratch/mutate_everything/repo')
sys.path.insert(0, str(repo))
import esm
from esm.model.esm2 import ESM2
from modeling.mutate_everything import MutateEverything
from mutant_cache.baselines.mutate_everything import AA, score_requested
torch.manual_seed(42)
alphabet = esm.data.Alphabet.from_architecture('ESM-1b')
esm.pretrained.esm2_t33_650M_UR50D = lambda: (ESM2(2, 64, 4, alphabet), alphabet)
args = argparse.Namespace(backbone='esm2_t33_650M_UR50D', freeze_at=0,
    aa_expand='backbone', head_dim=8, multi_dec='epistasis', single_dec='naive', test=True)
model = MutateEverything(args).eval()
for length in [16, 56]:
    seq = (AA*3)[:length]
    mutants, subs = [], []
    for k in [1, 2, 3, 5, 10]:
        s = [(i+1, seq[i], AA[(AA.index(seq[i])+1)%20]) for i in range(k)]
        subs.append(s)
        mutants.append(':'.join(f'{w}{p}{a}' for p,w,a in s))
    _, _, tokens = alphabet.get_batch_converter()([('oracle', seq)])
    batch = dict(seqs=[seq], eval_list=True,
        mut_info_list=[mutants], mut_pos_list=[[[p for p,w,a in s] for s in subs]],
        mut_to_type_list=[[[AA.index(a) for p,w,a in s] for s in subs]],
        mut_fr_type_list=[[[AA.index(w) for p,w,a in s] for s in subs]])
    with torch.no_grad():
        original = model(tokens, batch)
        requested = score_requested(model, tokens, seq, mutants)
    expected = original['pr_ddgs_list'][0].clone()
    expected[0] = original['mut1_ddg'][0, 0, AA.index(subs[0][0][2])]
    expected[1] = original['mut2_ddg'][0, 0, AA.index(subs[1][0][2]), 1, AA.index(subs[1][1][2])]
    torch.testing.assert_close(requested, expected, atol=3e-6, rtol=3e-5)
    print('original decoder parity passed', length, 'residues;', [len(s) for s in subs], 'substitutions', flush=True)

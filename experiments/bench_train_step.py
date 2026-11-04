"""Raw microbenchmark: one variant-LoRA training step (fwd+bwd) and inference, sparse vs exact,
bf16 base. 1AOY (L=69), n random single mutants. GPU 0."""
import time, torch
import pandas as pd
from mutant_cache.backbone import load_esmc
from mutant_cache.cache.lora import attach_variant_lora
from mutant_cache.cache.variant_cache import ragged_from_lists

seq = dict(pd.read_parquet('/home/andrew/mutant_cache_scratch/processed/esm_msr_mega/targets.parquet')[['target_id', 'sequence']].values)['1AOY']
m = load_esmc('esmc_300m', device='cuda', dtype=torch.bfloat16)
attach_variant_lora(m)
ids = m.tokenizer.encode(seq).cuda()
L = len(seq)
ref = m.forward_reference(ids)
aa = list(m.aa_to_id.values())
g = torch.Generator().manual_seed(0)


def bench(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def step(f, grad):
    with torch.set_grad_enabled(grad):
        o = f()
        if grad:
            o.hidden.float().sum().backward()


for n in (64, 256, 1024, 4096):
    p = (torch.randint(L, (n,), generator=g) + 1).tolist()
    a = [aa[int(i)] for i in torch.randint(20, (n,), generator=g)]
    rv = ragged_from_lists([[(pi, ai)] for pi, ai in zip(p, a)]).to('cuda')
    pos1 = torch.tensor([p[0]], device='cuda')           # shared-position batch for exact/padded
    vid1 = torch.tensor([[ai] for ai in a], device='cuda')
    ragged = lambda: m.forward_variants_ragged(ref, rv, output_logits=False)  # noqa: E731
    exact = lambda: m.forward_variants_exact(ref, pos1, vid1, output_logits=False)  # noqa: E731
    r = dict(n=n, ragged_train=bench(lambda: step(ragged, True)),
             exact_train=bench(lambda: step(exact, True)),
             ragged_infer=bench(lambda: step(ragged, False)),
             exact_infer=bench(lambda: step(exact, False)))
    r['train_speedup'] = r['exact_train'] / r['ragged_train']
    r['infer_speedup'] = r['exact_infer'] / r['ragged_infer']
    print({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}, flush=True)

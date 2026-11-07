# ESM3 sequence-only ESM-MSR experiment

ESM3-small-open, 1536 hidden units, 48 blocks, 24 heads; official base weights pinned to `47f0545b2b6daf26a93439a3cd610f4f7f3d5478`. This definition applies the frozen E02 methodology with rank-8 shared/variant adapters and the released ESM-MSR train/validation/test split. Three seeds, six exposures, validation macro Spearman checkpoint selection, then released internal-test scoring. No training has been launched for this definition.

The port preserves the official sequence-only missing-track defaults, including structure-token boundary embeddings and pLDDT/annotation embeddings. With no coordinates, the masked geometry branch contributes zero. Sequence-only cache and full-forward paths use the same input encoder. Coordinates, structure tokens and other conditional inputs are not supported here. The official ESM-MSR comparator can use structure: matching the backbone does not also match conditioning, augmentation or training methodology.

```bash
python -m mutant_cache.train --experiment experiments/stability_paper/msr_esm3_test_v1/experiment.json --seed 0 --dry-run
# After validation and GPU scheduling:
python -m mutant_cache.queue_runner run experiments/stability_paper/msr_esm3_test_v1/jobs.json --gpus 0,2 --dry-run
```

The loader downloads only `data/weights/esm3_sm_open_v1.pth` from `biohub/esm3-sm-open-v1`. Existing Hugging Face authentication is used if needed. No annotation vocabularies or structure decoders are required. Direct use:

```python
from mutant_cache.backbone import load_esm3
model = load_esm3(device='cpu', local_files_only=True)
out = model(model.tokenizer.encode('ACDEFGH'))
```

CPU checks: `OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_esm3.py`. Add `MUTANT_CACHE_TEST_ESM3=1` for the released-weight fp32 parity check (several GB of host memory). Small official-model tests check missing-track defaults, padding, masks, cache oracles, ragged execution, shared/variant adapter gradients, bf16 and training integration. The released test checks sequence logits and prenorm embeddings against the installed official ESM3 implementation.

The earlier ESM-2 test coordinator was stopped before it launched training, pending this backbone change. The exact paper comparator checkpoint and matched prediction-row reproduction remain pending as documented in the ESM-2 study.

"""Sequence-only ESM3 conversion and sparse correctness, without downloads."""
from types import SimpleNamespace
import os
import pytest
import torch

from mutant_cache.backbone.esm3 import ESM3
from mutant_cache.backbone.tokenizer import ESMCTokenizer
from mutant_cache.cache.lora import LoRAConfig, attach_shared_lora, attach_variant_lora


def official_model():
    from esm.models.esm3 import ESM3 as Official
    torch.manual_seed(31)
    return Official(64, 4, 4, 2, None, None, None,
                    SimpleNamespace(sequence=ESMCTokenizer())).eval()


def test_sequence_equivalence_and_missing_track_embeddings():
    ref = official_model()
    local = ESM3.from_official(ref)
    ids = local.tokenizer.encode_batch(['ACDEFGH', 'ACD'])
    ids[0, 3] = local.tokenizer.mask_token_id
    keep = ids != local.tokenizer.pad_token_id
    with torch.no_grad():
        expected = ref(sequence_tokens=ids, sequence_id=keep.long())
        actual = local(ids)
    torch.testing.assert_close(actual.logits[keep], expected.sequence_logits[keep], atol=4e-6, rtol=4e-5)
    # Official ESM3 calls its PRE-normalization stream 'embeddings'.
    torch.testing.assert_close(actual.hidden_prenorm[keep], expected.embeddings[keep], atol=4e-6, rtol=4e-5)
    assert not torch.equal(local.embed(ids), ref.encoder.sequence_embed(ids))


@pytest.mark.parametrize('shared', [False, True])
def test_sparse_oracle_gradients_and_exact_full_positions(shared):
    model = ESM3.from_official(official_model())
    attach_variant_lora(model, LoRAConfig(r=2))
    if shared:
        attach_shared_lora(model, LoRAConfig(r=2))
    for name, param in model.named_parameters():
        if 'lora_B' in name:
            torch.nn.init.normal_(param, std=.02)
    ids = model.tokenizer.encode('ACDEFG')
    ref = model.forward_reference(ids, grad=shared)
    positions = torch.tensor([2, 4])
    variants = torch.tensor([[5, 6], [7, int(ids[4])]])
    mask = torch.tensor([[True, True], [True, False]])
    fast = model.forward_variants(ref, positions, variants, mask)
    oracle = model.forward_variants_bruteforce(ref, positions, variants, mask)
    torch.testing.assert_close(fast.logits, oracle.logits, atol=5e-6, rtol=5e-5)
    from mutant_cache.cache.variant_cache import ragged_from_padded
    ragged = model.forward_variants_ragged(ref, ragged_from_padded(positions, variants, mask))
    torch.testing.assert_close(ragged.logits, oracle.logits[mask], atol=5e-6, rtol=5e-5)
    fast.hidden.square().sum().backward()
    assert all(p.grad is not None for p in model.variant_adapters.parameters())
    if shared:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.shared_adapters.parameters())
    allpos = torch.arange(len(ids))
    mutants = ids[None].clone()
    mutants[0, 2] = 5
    with torch.no_grad():
        full = model.forward_variants_exact(ref, allpos, mutants)
        sparse = model.forward_variants(ref, allpos, mutants)
    torch.testing.assert_close(sparse.logits, full.logits, atol=5e-6, rtol=5e-5)


def test_bf16_missing_track_projections_and_cache():
    model = ESM3.from_official(official_model()).to(torch.bfloat16)
    ids = model.tokenizer.encode('ACDEFG')
    with torch.no_grad():
        assert torch.isfinite(model(ids).logits).all()
        ref = model.forward_reference(ids)
        out = model.forward_variants(ref, torch.tensor([2]), torch.tensor([[5]]))
    assert torch.isfinite(out.logits).all()


def test_training_entry_point(tmp_path, monkeypatch):
    from fixtures import make_synthetic_dataset
    from mutant_cache import megascale
    import mutant_cache.backbone.esm3 as port
    data, _ = make_synthetic_dataset(tmp_path/'data', n_targets=5, seq_len=16,
                                    n_singles=12, n_doubles=12, n_triples=0)
    monkeypatch.setattr(port, 'load_esm3', lambda name, device, dtype:
                        ESM3.from_official(official_model()).to(device=device, dtype=dtype))
    args = megascale.build_parser().parse_args([
        '--data-dir', str(data), '--dataset', 'synthetic', '--split', 'protein',
        '--backbone', 'esm3_sm_open_v1', '--device', 'cpu', '--precision', 'bf16',
        '--lora-r', '2', '--shared-lora-r', '2', '--minimal', '--head', 'wt_relative',
        '--zero-anchor', '--no-marginals', '--exposures', '1', '--micro-batch', '16',
        '--proteins-per-step', '2', '--eval-min-n', '2', '--diagnostics',
        '--diagnostic-activation-rows', '8', '--output', str(tmp_path/'run')])
    result = megascale.run(args)
    assert result['status'] == 'ok'
    assert (tmp_path/'run/predictions_val.csv').is_file()
    assert not (tmp_path/'run/predictions_test.csv').exists()


@pytest.mark.skipif(os.environ.get('MUTANT_CACHE_TEST_ESM3') != '1', reason='opt-in released 1.4B weight test')
def test_released_fp32_equivalence():
    from esm.models.esm3 import ESM3 as Official
    from mutant_cache.backbone.esm3 import esm3_weights_path
    with torch.device('meta'):
        ref = Official(1536, 24, 256, 48, None, None, None,
                       SimpleNamespace(sequence=ESMCTokenizer()))
    ref.load_state_dict(torch.load(esm3_weights_path(local_files_only=True),
                                  map_location='cpu', mmap=True, weights_only=True),
                        strict=True, assign=True)
    ref.float().eval()
    # RoPE's nonpersistent buffers are absent from the checkpoint and remain
    # meta after assign-loading; rebuild the official reference's caches.
    from esm.layers.rotary import RotaryEmbedding
    for block in ref.transformer.blocks:
        block.attn.rotary = RotaryEmbedding(64)
    local = ESM3.from_official(ref)
    ids = local.tokenizer.encode('ACDEFGH')[None]
    with torch.no_grad():
        expected = ref(sequence_tokens=ids)
        actual = local(ids)
    torch.testing.assert_close(actual.logits, expected.sequence_logits, atol=1e-3, rtol=2e-4)
    torch.testing.assert_close(actual.hidden_prenorm, expected.embeddings, atol=2e-3, rtol=2e-4)

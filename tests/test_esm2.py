import pytest
import torch
import os

from mutant_cache.backbone.esm2 import ESM2, ESM2Tokenizer
from mutant_cache.cache.lora import attach_variant_lora, attach_shared_lora, LoRAConfig


def reference_model():
    from transformers import EsmConfig, EsmForMaskedLM
    torch.manual_seed(13)
    return EsmForMaskedLM(EsmConfig(vocab_size=33, hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        position_embedding_type="rotary", token_dropout=True, pad_token_id=1,
        mask_token_id=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0)).eval()


def test_hf_fp32_equivalence():
    ref = reference_model()
    model = ESM2.from_huggingface(ref)
    ids = model.tokenizer.encode_batch(["ACDEFGH", "ACD"])
    ids[0, 3] = 32  # exercise token dropout as well as padding
    with torch.no_grad():
        # transformers 4.57.6 EsmModel omits the embedding attention mask;
        # supply it explicitly to match the documented token-dropout behavior.
        emb = ref.esm.embeddings(ids, attention_mask=ids != 1)
        expected = ref(inputs_embeds=emb, attention_mask=ids != 1, output_hidden_states=True)
        actual = model(ids)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.hidden, expected.hidden_states[-1], atol=2e-6, rtol=2e-5)
    assert ESM2Tokenizer().mask_token_id == 32


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_cache_oracle_and_adapter_gradients(shared, device):
    model = ESM2.from_huggingface(reference_model()).to(device)
    attach_variant_lora(model, LoRAConfig(r=2, targets=("q", "k", "v", "o", "fc1", "fc2")))
    if shared:
        attach_shared_lora(model)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    ids = model.tokenizer.encode("ACDEFG")
    ref = model.forward_reference(ids, grad=shared)
    positions = torch.tensor([2, 4], device=device)
    variants = torch.tensor([[5, 6], [7, int(ids[4])]], device=device)
    mask = torch.tensor([[True, True], [True, False]], device=device)
    fast = model.forward_variants(ref, positions, variants, mask)
    oracle = model.forward_variants_bruteforce(ref, positions, variants, mask)
    torch.testing.assert_close(fast.logits, oracle.logits, atol=3e-6, rtol=3e-5)
    from mutant_cache.cache.variant_cache import ragged_from_padded
    ragged = model.forward_variants_ragged(ref, ragged_from_padded(positions, variants, mask))
    torch.testing.assert_close(ragged.logits, oracle.logits[mask], atol=3e-6, rtol=3e-5)
    fast.hidden.square().sum().backward()
    assert all(p.grad is not None for p in model.variant_adapters.parameters())


def test_full_position_cache_is_exact():
    m = ESM2.from_huggingface(reference_model())
    ids = m.tokenizer.encode("ACDEFG")
    ref = m.forward_reference(ids)
    pos = torch.arange(len(ids))
    mutants = ids[None].repeat(2, 1)
    mutants[:, 2] = torch.tensor([5, 6])
    with torch.no_grad():
        fast = m.forward_variants(ref, pos, mutants)
        exact = m.forward_variants_exact(ref, pos, mutants)
    torch.testing.assert_close(fast.logits, exact.logits, atol=3e-6, rtol=3e-5)


def test_sparse_masks_rejected():
    m = ESM2(32, 4, 2)
    with pytest.raises(ValueError, match="substitutions only"):
        m.embed_sparse(torch.tensor([[32]]))


@pytest.mark.skipif(os.environ.get("MUTANT_CACHE_TEST_ESM2_650M") != "1",
                    reason="opt-in cached 650M fp32 parity test")
def test_pretrained_650m_fp32():
    from transformers import EsmForMaskedLM
    ref = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D",
        local_files_only=True, attn_implementation="eager").eval()
    local = ESM2.from_huggingface(ref)
    device = os.environ.get("MUTANT_CACHE_TEST_DEVICE", "cpu")
    ref, local = ref.to(device), local.to(device)
    ids = local.tokenizer.encode_batch(["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQ", "ACDEFG"] ).to(device)
    with torch.no_grad():
        emb = ref.esm.embeddings(ids, attention_mask=ids != 1)
        expected = ref(inputs_embeds=emb, attention_mask=ids != 1, output_hidden_states=True)
        actual = local(ids)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual.hidden, expected.hidden_states[-1], atol=1e-4, rtol=2e-4)

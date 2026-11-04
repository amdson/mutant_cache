"""ESM-2 in the cache module layout, with Hugging Face checkpoint conversion.

No transformers dependency at inference time. ``from_huggingface`` accepts an
EsmForMaskedLM and is also the small-model fp32 parity test entry point.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .esmc import ESMC, ESMCOutput, apply_rope, rope_inv_freq
from .tokenizer import ESMCTokenizer, SEQUENCE_VOCAB


class ESM2Tokenizer(ESMCTokenizer):
    def __init__(self):
        super().__init__()
        # ESM-2 uses a null slot in place of ESM-C's chain separator.
        self.vocab = ["<null_1>" if t == "|" else t for t in SEQUENCE_VOCAB]
        self.tok_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.mask_token_id = self.tok_to_id["<mask>"]


class ESMGELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class QueryScale(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** -0.5

    def forward(self, x):
        return x * self.scale


class ESM2Attention(nn.Module):
    def __init__(self, d, h, eps):
        super().__init__()
        self.n_heads, self.d_head = h, d // h
        self.layernorm_qkv = nn.Sequential(nn.LayerNorm(d, eps=eps), nn.Linear(d, 3*d))
        self.q_ln, self.k_ln = QueryScale(d // h), nn.Identity()
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, cos, sin, attn_mask=None):
        b, l, d = x.shape
        q, k, v = self.layernorm_qkv(x).chunk(3, -1)
        q = apply_rope(self.q_ln(q).view(b, l, self.n_heads, self.d_head), cos, sin)
        k = apply_rope(k.view(b, l, self.n_heads, self.d_head), cos, sin)
        v = v.view(b, l, self.n_heads, self.d_head)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), attn_mask, scale=1.0)
        return self.out_proj(o.transpose(1, 2).reshape(b, l, d))


class ESM2Block(nn.Module):
    def __init__(self, d, h, intermediate, eps):
        super().__init__()
        self.attn = ESM2Attention(d, h, eps)
        self.ffn = nn.Sequential(nn.LayerNorm(d, eps=eps), nn.Linear(d, intermediate),
                                 ESMGELU(), nn.Linear(intermediate, d))
        self.plain_mlp = True
        self.scaling_factor = 1.0

    def forward(self, x, cos, sin, attn_mask=None):
        x = x + self.attn(x, cos, sin, attn_mask)
        return x + self.ffn(x)


class ESM2(ESMC):
    default_lora_targets = ("q", "o", "fc1", "fc2")
    attention_scale = 1.0  # query is scaled before RoPE

    def __init__(self, d_model=1280, n_heads=20, n_layers=33, intermediate_size=None,
                 layer_norm_eps=1e-5, token_dropout=True, name="esm2_t33_650M_UR50D"):
        super().__init__(d_model, n_heads, 0, name=name)
        self.n_layers = n_layers
        self.tokenizer = ESM2Tokenizer()
        self.aa_to_id = dict(self.tokenizer.aa_to_id)
        self.vocab_size = len(self.tokenizer)
        self.token_dropout = token_dropout
        self.register_buffer("rotary_inv_freq", rope_inv_freq(self.head_dim))
        self.embed = nn.Embedding(self.vocab_size, d_model, padding_idx=1)
        self.transformer.blocks = nn.ModuleList([
            ESM2Block(d_model, n_heads, intermediate_size or 4*d_model, layer_norm_eps)
            for _ in range(n_layers)])
        self.transformer.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.sequence_head = nn.Sequential(nn.Linear(d_model, d_model), ESMGELU(),
            nn.LayerNorm(d_model, eps=layer_norm_eps), nn.Linear(d_model, self.vocab_size))

    def _apply(self, fn, recurse=True):
        # Preserve checkpoint frequencies in fp32 when casting the base to bf16.
        freq = self.rotary_inv_freq.clone()
        super()._apply(fn, recurse=recurse)
        self.rotary_inv_freq = freq.to(device=self.rotary_inv_freq.device)
        return self

    def inv_freq(self, device):
        return self.rotary_inv_freq.to(device)

    def embed_sparse(self, ids):
        if self.token_dropout and (ids == self.tokenizer.mask_token_id).any():
            raise ValueError("ESM-2 sparse cache supports substitutions only; use full forward for masked marginals")
        x = self.embed(ids)
        return x * 0.88 if self.token_dropout else x

    def forward_reference(self, ids, grad=False, cache_kv=True):
        if (ids == self.tokenizer.mask_token_id).any() or (ids == self.tokenizer.pad_token_id).any():
            raise ValueError("ESM-2 reference cache requires one unmasked, unpadded sequence")
        return super().forward_reference(ids, grad=grad, cache_kv=cache_kv)

    def forward(self, ids, attention_mask=None):
        if ids.ndim == 1:
            ids = ids[None]
        ids = ids.to(self.device)
        keep = (ids != self.tokenizer.pad_token_id) if attention_mask is None else attention_mask.to(self.device).bool()
        x = self.embed(ids)
        if self.token_dropout:
            masked = ids == self.tokenizer.mask_token_id
            ratio = masked.sum(-1).float() / keep.sum(-1)
            x = (x.masked_fill(masked[..., None], 0) * 0.88 / (1-ratio)[:, None, None]).to(x.dtype)
        x = x * keep[..., None]
        mask = None if bool(keep.all()) else keep[:, None, None, :]
        cos, sin = self.rope(torch.arange(ids.shape[1], device=self.device), x.dtype)
        inputs = []
        from mutant_cache.cache.variant_cache import _full_block
        for i, blk in enumerate(self.blocks):
            inputs.append(x)
            x = (blk(x, cos, sin, mask) if self.shared_adapters is None else
                 _full_block(self, blk, x, cos, sin, mask, (self.shared_adapters[i], None), None))
        hidden = self.transformer.norm(x)
        return ESMCOutput(self.sequence_head(hidden), hidden, inputs, x)

    @classmethod
    def from_huggingface(cls, reference):
        c = reference.config
        if c.position_embedding_type != "rotary" or c.emb_layer_norm_before:
            raise ValueError("only rotary ESM-2 without embedding LayerNorm is supported")
        m = cls(c.hidden_size, c.num_attention_heads, c.num_hidden_layers,
                c.intermediate_size, c.layer_norm_eps, c.token_dropout)
        def copy(dst, src):
            dst.load_state_dict(src.state_dict(), strict=True)
        copy(m.embed, reference.esm.embeddings.word_embeddings)
        freq = reference.esm.encoder.layer[0].attention.self.rotary_embeddings.inv_freq
        # Published HF checkpoints contain rounded frequencies. Recomputing
        # ideal frequencies changes predictions, even in fp32.
        for layer in reference.esm.encoder.layer:
            if not torch.equal(freq, layer.attention.self.rotary_embeddings.inv_freq):
                raise ValueError("per-layer rotary frequencies differ")
        m.rotary_inv_freq.copy_(freq.float())
        for dst, src in zip(m.blocks, reference.esm.encoder.layer):
            copy(dst.attn.layernorm_qkv[0], src.attention.LayerNorm)
            with torch.no_grad():
                for attr in ("weight", "bias"):
                    getattr(dst.attn.layernorm_qkv[1], attr).copy_(torch.cat([
                        getattr(getattr(src.attention.self, p), attr) for p in ("query", "key", "value")]))
            copy(dst.attn.out_proj, src.attention.output.dense)
            copy(dst.ffn[0], src.LayerNorm)
            copy(dst.ffn[1], src.intermediate.dense)
            copy(dst.ffn[3], src.output.dense)
        copy(m.transformer.norm, reference.esm.encoder.emb_layer_norm_after)
        copy(m.sequence_head[0], reference.lm_head.dense)
        copy(m.sequence_head[2], reference.lm_head.layer_norm)
        with torch.no_grad():
            m.sequence_head[3].weight.copy_(reference.lm_head.decoder.weight)
            m.sequence_head[3].bias.copy_(reference.lm_head.bias)
        # ESM-2 ties the decoder to the embedding matrix.
        m.sequence_head[3].weight = m.embed.weight
        return m.eval()


def load_esm2(name="esm2_t33_650M_UR50D", device="cuda", dtype=torch.float32,
              weights_path=None, local_files_only=False):
    from transformers import EsmForMaskedLM
    ref = EsmForMaskedLM.from_pretrained(weights_path or f"facebook/{name}",
                                        local_files_only=local_files_only,
                                        attn_implementation="eager")
    model = ESM2.from_huggingface(ref)
    model.name = name
    return model.to(device=device, dtype=dtype).eval()

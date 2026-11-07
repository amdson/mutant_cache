"""ESM3-small sequence-only backbone in the sparse-cache module layout.

Matches official ``ESM3.forward(sequence_tokens=...)`` with all other tracks
omitted. Structure-conditioned inference is intentionally not exposed.
"""
from __future__ import annotations

from types import SimpleNamespace
import torch
from torch import nn

from .esmc import ESMC
from .tokenizer import ESMCTokenizer

ESM3_NAME = 'esm3_sm_open_v1'
ESM3_REPO = 'biohub/esm3-sm-open-v1'
ESM3_WEIGHTS = 'data/weights/esm3_sm_open_v1.pth'
ESM3_REVISION = '47f0545b2b6daf26a93439a3cd610f4f7f3d5478'


class SequenceInputs(nn.Module):
    """Keep official missing-track defaults in full and sparse embeddings."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    @property
    def weight(self):
        return self.encoder.sequence_embed.weight

    def forward(self, ids):
        from esm.utils.constants import esm3 as C
        if ids.numel() == 0:
            return self.encoder.sequence_embed(ids)
        shape = ids.shape
        tokens = ids.reshape(-1, shape[-1])
        def fill(value):
            return torch.full_like(tokens, value)
        structure = fill(C.STRUCTURE_MASK_TOKEN)
        for seq, struct in [(C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN),
                            (C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN),
                            (C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN),
                            (C.SEQUENCE_CHAINBREAK_TOKEN, C.STRUCTURE_CHAINBREAK_TOKEN)]:
            structure = structure.masked_fill(tokens == seq, struct)
        with torch.autocast(self.weight.device.type, dtype=self.weight.dtype,
                            enabled=self.weight.dtype in (torch.bfloat16, torch.float16)):
            x = self.encoder(tokens, structure, fill(1).float(), fill(0).float(),
                             fill(C.SS8_PAD_TOKEN), fill(C.SASA_PAD_TOKEN),
                             fill(C.INTERPRO_PAD_TOKEN)[..., None].expand(*tokens.shape, 8),
                             fill(C.RESIDUE_PAD_TOKEN)[..., None].expand(*tokens.shape, 16))
        return x.reshape(*shape, x.shape[-1])


class ESM3(ESMC):
    @classmethod
    def from_official(cls, reference):
        """Convert an fp32 official ESM3; share its pretrained tensors.

        Frameless geometric attention is exactly zero with the supported
        bias-free projection. All sequence transformer weights load strictly.
        """
        blocks = reference.transformer.blocks
        first = blocks[0]
        d, heads = first.attn.d_model, first.attn.n_heads
        for block in blocks:
            if block.use_geom_attn:
                if not block.geom_attn.mask_and_zero_frameless or block.geom_attn.out_proj.bias is not None:
                    raise ValueError('ESM3 sequence conversion requires zero frameless geometry')
        with torch.device('meta'):
            model = cls(d, heads, len(blocks), name=ESM3_NAME)
        # The local stack has no geometry branch: it contributes exactly zero
        # when no coordinates are supplied. Reject any other unmapped weights.
        state = {k: v for k, v in reference.transformer.state_dict().items()
                 if '.geom_attn.' not in k}
        model.transformer.load_state_dict(state, strict=True, assign=True)
        model.sequence_head.load_state_dict(reference.output_heads.sequence_head.state_dict(),
                                            strict=True, assign=True)
        model.embed = SequenceInputs(reference.encoder)
        return model.eval()


def esm3_weights_path(*, revision=ESM3_REVISION, local_files_only=False):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(ESM3_REPO, ESM3_WEIGHTS, revision=revision,
                           local_files_only=local_files_only)


def load_esm3(name=ESM3_NAME, device='cuda', dtype=torch.float32,
              weights_path=None, revision=ESM3_REVISION, local_files_only=False):
    if name != ESM3_NAME:
        raise ValueError(f'unsupported ESM3 backbone {name!r}; use {ESM3_NAME}')
    from esm.models.esm3 import ESM3 as Official
    path = weights_path or esm3_weights_path(revision=revision, local_files_only=local_files_only)
    # Sequence-only forward never needs annotation vocabularies or decoders.
    with torch.device('meta'):
        reference = Official(1536, 24, 256, 48, None, None, None,
                             SimpleNamespace(sequence=ESMCTokenizer()))
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    reference.load_state_dict(state, strict=True, assign=True)
    model = ESM3.from_official(reference)
    model.weights_path = str(path)
    return model.to(device=device, dtype=dtype).eval()

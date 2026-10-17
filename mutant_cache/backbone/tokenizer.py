"""Character-level ESM-C tokenizer.

Same vocabulary and special-token wrapping as the official
``esm.tokenization.EsmSequenceTokenizer`` / ``esm.models.esmc.EsmcTokenizer``
(``<cls> seq <eos>``), without the HuggingFace ``tokenizers`` machinery: a
protein is one token per character, so a dict lookup is the whole tokenizer.
`tests/test_backbone_equivalence.py` checks it against the official one.
"""
from __future__ import annotations

import torch

# Index order is baked into the checkpoint (embedding rows, LM-head columns).
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O", ".", "-", "|",
    "<mask>",
]
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


class ESMCTokenizer:
    def __init__(self):
        self.vocab = list(SEQUENCE_VOCAB)
        self.tok_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.cls_token_id = self.tok_to_id["<cls>"]
        self.bos_token_id = self.cls_token_id
        self.pad_token_id = self.tok_to_id["<pad>"]
        self.eos_token_id = self.tok_to_id["<eos>"]
        self.unk_token_id = self.tok_to_id["<unk>"]
        self.mask_token_id = self.tok_to_id["<mask>"]
        self.aa_to_id = {a: self.tok_to_id[a] for a in CANONICAL_AA}
        self.id_to_aa = {i: a for a, i in self.aa_to_id.items()}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, seq: str, add_special_tokens: bool = True) -> torch.Tensor:
        """``(L+2,)`` LongTensor ``[<cls>, residues..., <eos>]``. Unknown
        characters map to ``<unk>``; ``_`` / ``#`` are not special here."""
        ids = [self.tok_to_id.get(c, self.unk_token_id) for c in seq.upper()]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.eos_token_id]
        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(self, seqs: list[str]) -> torch.Tensor:
        """Right-padded ``(B, max L+2)`` batch (pad id 1)."""
        enc = [self.encode(s) for s in seqs]
        n = max(len(e) for e in enc)
        out = torch.full((len(enc), n), self.pad_token_id, dtype=torch.long)
        for i, e in enumerate(enc):
            out[i, : len(e)] = e
        return out

    def decode(self, ids, strip_special: bool = True) -> str:
        ids = [int(i) for i in (ids.tolist() if torch.is_tensor(ids) else ids)]
        toks = [self.vocab[i] for i in ids]
        if strip_special:
            toks = [t for t in toks if not (t.startswith("<") and t.endswith(">"))]
        return "".join(toks)


_DEFAULT = ESMCTokenizer()


def tokenize(seq: str) -> torch.Tensor:
    return _DEFAULT.encode(seq)

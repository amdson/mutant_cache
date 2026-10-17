"""Baselines: one-hot ridge, frozen-embedding ridge/MLP, masked-marginal zero-shot."""
from mutant_cache.baselines.frozen import (
    EmbeddingMLP, EmbeddingRidge, mean_pooled_embeddings,
)
from mutant_cache.baselines.onehot import (
    OneHotRidgePerAssay, build_pair_index, onehot_features,
)
from mutant_cache.baselines.zero_shot import (
    MaskedMarginalZeroShot, build_llr_matrices, llr_matrix,
    masked_marginal_matrix, score_variants,
)

__all__ = [
    "OneHotRidgePerAssay", "onehot_features", "build_pair_index",
    "EmbeddingRidge", "EmbeddingMLP", "mean_pooled_embeddings",
    "MaskedMarginalZeroShot", "masked_marginal_matrix", "llr_matrix",
    "build_llr_matrices", "score_variants",
]

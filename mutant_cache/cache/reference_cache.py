"""
mutant_cache/cache/reference_cache.py
-------------------------------------
Store of frozen reference states, keyed by ``(target_id, sha1(sequence))``.

Validity
--------
Reference states are a function of the **base weights**. They are valid for as
long as the base is frozen — which is the variant-only-LoRA regime
(`mutant_cache.cache.lora`), where adapters touch only the variant stream. If
base weights are trained, every cached entry is stale after the first step.
Serving a stale stream would look like a modelling result rather than a bug, so
the cache fingerprints the base weights at construction and re-checks on every
miss (and on ``get(..., check=True)``), raising if they moved. On-disk entries
are additionally namespaced by model name, dtype and that fingerprint, so a
different checkpoint can never read them.

Layout
------
* in memory: LRU (``max_entries``), states kept on ``store_device`` (default:
  the model's device; ``"cpu"`` saves GPU memory at a transfer per hit);
* on disk (``persist=True``): ``$MUTANT_CACHE_SCRATCH/refcache/<model>_<dtype>_<fp>/
  <target_id>__<sha1>.pt`` (default scratch ``~/mutant_cache_scratch``).
  Stores ``ids, layer_inputs, final, final_prenorm``; the reference K/V are
  recomputed on load (cheap, one projection per layer).

Cost per entry: n_layers * (L+2) * d * itemsize — ESM-C 300M, L=300, bf16:
30 * 302 * 960 * 2 B ~ 17 MB, plus the same again for the memoized K/V once
`forward_variants` has used it.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch

from mutant_cache.backbone.esmc import ESMC
from .variant_cache import ReferenceStates, forward_reference, reference_kv


def scratch_dir() -> Path:
    return Path(os.environ.get("MUTANT_CACHE_SCRATCH",
                               os.path.expanduser("~/mutant_cache_scratch")))


def seq_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()


def base_fingerprint(model: ESMC) -> str:
    """Cheap hash of a strided sample of every base parameter."""
    h = hashlib.sha1()
    with torch.no_grad():
        for name, p in model.base_parameters():
            flat = p.detach().reshape(-1)
            sample = flat[:: max(1, flat.numel() // 64)][:64].float().cpu()
            h.update(name.encode())
            h.update(sample.numpy().tobytes())
    return h.hexdigest()[:12]


class ReferenceCache:
    def __init__(self, model: ESMC, max_entries: Optional[int] = 128,
                 persist: bool = False, root: Optional[str | os.PathLike] = None,
                 store_device: Optional[str | torch.device] = None,
                 cache_kv: bool = True):
        self.model = model
        self.max_entries = max_entries
        self.persist = persist
        self.store_device = torch.device(store_device) if store_device else None
        self.cache_kv = cache_kv
        self._fingerprint = base_fingerprint(model)
        dtype = str(model.dtype).replace("torch.", "")
        self.root = (Path(root) if root else scratch_dir() / "refcache") / \
            f"{model.name}_{dtype}_{self._fingerprint}"
        self._entries: "OrderedDict[tuple[str, str], ReferenceStates]" = OrderedDict()
        self.hits = self.misses = self.disk_hits = 0

    # -- validity ------------------------------------------------------------ #
    def check_base_unchanged(self) -> None:
        if base_fingerprint(self.model) != self._fingerprint:
            raise RuntimeError(
                "ReferenceCache: base weights changed since the cache was built, so "
                "every cached reference state is stale. Reference states are only "
                "valid with a frozen base (variant-only LoRA); call clear() and "
                "rebuild the cache, or run without it.")

    # -- keys / paths -------------------------------------------------------- #
    @staticmethod
    def key(target_id: str, sequence: str) -> tuple[str, str]:
        return (str(target_id), seq_sha1(sequence))

    def _path(self, key) -> Path:
        tid = re.sub(r"[^A-Za-z0-9._-]+", "_", key[0])[:100]
        return self.root / f"{tid}__{key[1]}.pt"

    # -- main API ------------------------------------------------------------ #
    def get(self, target_id: str, sequence: str, check: bool = False) -> ReferenceStates:
        """Reference states for ``sequence`` on the model's device/dtype."""
        key = self.key(target_id, sequence)
        if check:
            self.check_base_unchanged()
        ref = self._entries.get(key)
        if ref is not None:
            self.hits += 1
            self._entries.move_to_end(key)
        else:
            self.misses += 1
            self.check_base_unchanged()
            ref = self._load(key) if self.persist else None
            if ref is None:
                ids = self.model.tokenizer.encode(sequence)
                ref = forward_reference(self.model, ids, grad=False, cache_kv=False)
                if self.persist:
                    self._save(key, ref)
            else:
                self.disk_hits += 1
            if self.cache_kv:   # ref is on the model device here
                with torch.no_grad():
                    reference_kv(self.model, ref)
            if self.store_device is not None:
                ref = ref.to(device=self.store_device)
            self._entries[key] = ref
            if self.max_entries and len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        if ref.final.device != self.model.device:
            ref = ref.to(device=self.model.device)
        return ref

    def __contains__(self, item) -> bool:
        return self.key(*item) in self._entries

    def clear(self, disk: bool = False) -> None:
        self._entries.clear()
        if disk and self.root.exists():
            for p in self.root.glob("*.pt"):
                p.unlink()

    # -- disk ---------------------------------------------------------------- #
    def _save(self, key, ref: ReferenceStates) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        torch.save({"key": key, "ids": ref.ids.cpu(),
                    "layer_inputs": torch.cat(ref.layer_inputs, 0).cpu(),
                    "final": ref.final.cpu(), "final_prenorm": ref.final_prenorm.cpu()},
                   tmp)
        os.replace(tmp, path)

    def _load(self, key) -> Optional[ReferenceStates]:
        path = self._path(key)
        if not path.exists():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(blob["key"]) != tuple(key):
            return None
        dev, dt = self.model.device, self.model.dtype
        return ReferenceStates(
            ids=blob["ids"].to(dev),
            layer_inputs=[t[None].to(dev, dt) for t in blob["layer_inputs"]],
            final=blob["final"].to(dev, dt), final_prenorm=blob["final_prenorm"].to(dev, dt))

    # -- stats --------------------------------------------------------------- #
    def nbytes(self) -> int:
        return sum(r.nbytes() for r in self._entries.values())

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses,
                "disk_hits": self.disk_hits,
                "hit_rate": self.hits / total if total else 0.0,
                "mb": self.nbytes() / 2 ** 20}

import pytest
import torch

from mutant_cache.backbone import load_esmc
from mutant_cache.cache import ReferenceCache, attach_variant_lora, detach_variant_lora

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_A = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
SEQ_B = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


@pytest.fixture(scope="module")
def model():
    return load_esmc("esmc_300m", device=DEV, dtype=torch.float32).freeze_base()


def _same(r1, r2):
    return (torch.equal(r1.ids, r2.ids) and torch.equal(r1.final, r2.final)
            and all(torch.equal(a, b) for a, b in zip(r1.layer_inputs, r2.layer_inputs)))


def test_memory_lru(model, tmp_path):
    rc = ReferenceCache(model, max_entries=1, root=tmp_path)
    ra = rc.get("A", SEQ_A)
    assert _same(ra, model.forward_reference(model.tokenizer.encode(SEQ_A)))
    assert ra.kv is not None
    assert rc.get("A", SEQ_A) is ra and rc.hits == 1
    rc.get("B", SEQ_B)
    assert ("A", SEQ_A) not in rc and ("B", SEQ_B) in rc
    # same target id, different sequence -> different entry
    rc.get("B", SEQ_A)
    assert rc.misses == 3


def test_disk_persistence(model, tmp_path):
    rc = ReferenceCache(model, persist=True, root=tmp_path)
    ra = rc.get("urn:mavedb:0001/x", SEQ_A)
    files = list(rc.root.glob("*.pt"))
    assert len(files) == 1
    rc2 = ReferenceCache(model, persist=True, root=tmp_path)
    ra2 = rc2.get("urn:mavedb:0001/x", SEQ_A)
    assert rc2.disk_hits == 1 and _same(ra, ra2)
    pos = torch.tensor([3, 10], device=DEV)
    vid = torch.tensor([[5, 6]], device=DEV)
    with torch.no_grad():
        assert torch.equal(model.forward_variants(ra, pos, vid).hidden,
                           model.forward_variants(ra2, pos, vid).hidden)


def test_cpu_store(model, tmp_path):
    rc = ReferenceCache(model, root=tmp_path, store_device="cpu")
    r = rc.get("A", SEQ_A)
    assert r.final.device == model.device and r.kv is not None
    assert next(iter(rc._entries.values())).final.device.type == "cpu"


def test_lora_keeps_cache_valid_base_change_invalidates(model, tmp_path):
    rc = ReferenceCache(model, root=tmp_path)
    rc.get("A", SEQ_A)
    attach_variant_lora(model)
    try:
        rc.get("B", SEQ_B, check=True)       # adapters are not base weights
    finally:
        detach_variant_lora(model)
    w = model.blocks[0].attn.out_proj.weight
    with torch.no_grad():
        w.view(-1)[0] += 1.0
    try:
        with pytest.raises(RuntimeError):
            rc.get("C", SEQ_A + "K")
    finally:
        with torch.no_grad():
            w.view(-1)[0] -= 1.0

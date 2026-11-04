import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from mutant_cache.data.mutate_everything import normalize, build, SPLIT
from mutant_cache.eval.mutate_everything import detection, join_predictions, paired_seed_bootstrap
from mutant_cache.comparison import compare


def raw(tid="p", seq="ACD"):
    return pd.DataFrame(dict(pdb_id=[tid]*3, wt_seq=[seq]*3,
        mut_info=[f"{seq[0]}1G", f"{seq[1]}2W", f"{seq[0]}1G:{seq[1]}2W"], ddg=[-0.5, -0.1, 1.]))


def test_loader_audit_and_median():
    d = raw()
    d = pd.concat([d, d.iloc[[0]].assign(ddg=-1.5),
                   d.iloc[[0]].assign(mut_info="W1G"), d.iloc[[0]].assign(ddg=np.inf)])
    f, seqs, audit = normalize(d)
    assert f.set_index("mutant").loc["A1G", "score"] == 1.
    assert set(audit.reason) == {"wt_mismatch", "nonfinite_score"}
    assert seqs == {"p": "ACD"}


def test_build_and_split(tmp_path, monkeypatch):
    monkeypatch.setenv("MUTANT_CACHE_SCRATCH", str(tmp_path))
    tr = pd.concat([raw("p1"), raw("p2", "CDE"), raw("alias")])
    train, test = tmp_path/"train.csv", tmp_path/"test.csv"
    tr.to_csv(train, index=False)
    raw("test", "DEF").to_csv(test, index=False)
    build(train, test)
    out = tmp_path/"processed"/"mutate_everything_cdna"
    splits = pd.read_parquet(out/"splits"/f"{SPLIT}.parquet").set_index("target_id").split
    assert splits["p1"] == splits["alias"]
    assert splits["test"] == "test"
    assert set(splits) == {"train", "val", "test"}
    assert json.loads((out/"manifest.json").read_text())["files"]["train"]["sha256"]
    raw("test", "ACD").to_csv(test, index=False)
    with pytest.raises(ValueError, match="overlap"):
        build(train, test)


def predictions():
    f, _, _ = normalize(raw())
    return f.assign(pred=[0., 0., 0.])


def test_detection_threshold_ties_and_eligibility():
    d = predictions()
    r = detection(d).iloc[0]
    assert r.precision == pytest.approx(1/3)  # +0.1 is not a detection
    assert r.ndcg == pytest.approx(ndcg_score(np.maximum(d.score, 0).to_numpy()[None],
                                             np.zeros((1, 3)), k=30))
    assert not detection(d.assign(score=0.1)).iloc[0].eligible


def test_join_audits_and_rejects_sign_errors():
    d = predictions()
    matched, unmatched = join_predictions(d, d.iloc[:2])
    assert len(matched) == 2 and len(unmatched) == 1
    with pytest.raises(ValueError, match="labels disagree"):
        join_predictions(d, d.assign(score=-d.score))
    with pytest.raises(ValueError, match="duplicate"):
        join_predictions(pd.concat([d, d]), d)


def test_report_gate_and_bootstrap(tmp_path):
    d = predictions()
    result = compare(d, [(s, d.assign(pred=d.score)) for s in (0, 1, 2)], tmp_path)
    assert not result["superiority_claim_eligible"]  # no verified published numbers
    assert (tmp_path/"unmatched.csv").exists()
    pair = pd.DataFrame([dict(seed=s, target_id=t, difference=v)
                         for s in (0, 1, 2) for t, v in (("a", 1.), ("b", 2.))])
    ci = paired_seed_bootstrap(pair, n_boot=100)
    assert ci["mean_diff"] == 1.5 and ci["n_proteins"] == 2
    with pytest.raises(ValueError, match="complete"):
        paired_seed_bootstrap(pair.iloc[:-1])


def test_higher_order_breakdown():
    from mutant_cache.eval.mutate_everything import evaluate
    f = pd.DataFrame(dict(target_id=['p']*4, mutant=['A1G', 'A1G:C2W',
        'A1G:C2W:D3E', 'A1G:C2W:D3F:E4G'], n_mut=[1, 2, 3, 4],
        score=[-1., 0.5, 1., 2.], pred=[-0.8, 0.6, 0.9, 1.9]))
    ev = evaluate(f)
    tab = ev['table'].set_index('depth')
    assert tab.loc['all', 'n'] == 4
    assert tab.loc['double', 'n'] == 1
    assert tab.loc['multi', 'n'] == 2
    assert tab.loc['multi', 'spearman'] == pytest.approx(1.)


def test_reverse_reference_scoring_preserves_protein_groups(tmp_path):
    import torch
    from mutant_cache.backbone.esm2 import ESM2
    from mutant_cache.models import VariantEffectModel, VariantModelConfig
    from mutant_cache.benchmark import score_labels
    cfg = VariantModelConfig(minimal=True, head='sum_mean_count', zero_anchor=True)
    model = VariantEffectModel(ESM2(32, 4, 2), cfg).eval()
    frame = pd.DataFrame(dict(target_id=['protein', 'protein'], reference_id=['r1', 'r2'],
        wt_seq=['GCD', 'AWD'], mutant=['G1A', 'W2C'], n_mut=[1, 1], score=[1., -1.]))
    scored = score_labels(model, frame)
    assert scored.target_id.tolist() == ['protein', 'protein']
    assert scored.reference_id.tolist() == ['r1', 'r2']
    assert scored['pred'].notna().all()
    # Distinct mutant references must occupy distinct cache entries.
    assert len(model._ref_cache) == 2


def test_training_integration(tmp_path, monkeypatch):
    from mutant_cache import megascale
    from mutant_cache.backbone.esm2 import ESM2
    monkeypatch.setenv("MUTANT_CACHE_SCRATCH", str(tmp_path))
    train, test = tmp_path/"train.csv", tmp_path/"test.csv"
    pd.concat([raw("a"), raw("b", "CDE"), raw("c", "DEF")]).to_csv(train, index=False)
    raw("test", "EFH").to_csv(test, index=False)
    build(train, test)
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    m = ESM2(32, 4, 2)
    monkeypatch.setattr(megascale, "load_backbone", lambda *args: m)
    args = megascale.build_parser().parse_args([
        "--dataset", "mutate_everything_cdna", "--split", SPLIT,
        "--backbone", "esm2_t33_650M_UR50D", "--device", "cpu", "--precision", "fp32",
        "--exposures", "1", "--micro-batch", "3", "--proteins-per-step", "1",
        "--eval-min-n", "2", "--final-test", "--benchmark-inference",
        "--output", str(tmp_path/"run")])
    result = megascale.run(args)
    assert result["status"] == "ok"
    assert json.loads((tmp_path/"run"/"inference_cost.json").read_text())["rows"] == 3
    assert m.lora_config.targets == ("q", "o", "fc1", "fc2")

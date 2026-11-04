import pandas as pd
import pytest
import torch
from mutant_cache.training_protocol import stability_loss, ReferenceAugmentor
from mutant_cache.megascale import ProteinUniformSampler
from mutant_cache.data.mutations import mutate


def test_point_losses_and_gradient_match_torch():
    target = torch.tensor([-2., 0., 3.])
    depths = torch.tensor([1, 2, 2])
    for kind, oracle in [('mse_dg', torch.nn.functional.mse_loss),
                         ('huber_dg', torch.nn.functional.huber_loss)]:
        a = torch.tensor([1., 1., 1.], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        ours = stability_loss(a, target, depths, kind=kind)
        expected = oracle(b, target)
        ours.backward(); expected.backward()
        torch.testing.assert_close(ours, expected)
        torch.testing.assert_close(a.grad, b.grad)


def test_depth_weighting_matches_explicit_task_means():
    p = torch.tensor([0., 1., -1., 0.], requires_grad=True)
    t = torch.tensor([1., -1., 1., 2.])
    n = torch.tensor([1, 2, 2, 2])
    point = torch.nn.functional.huber_loss(p, t, reduction='none')
    expected = .1*point[0] + (point[1]+4*point[2]+4*point[3])/9
    actual = stability_loss(p,t,n,kind='huber_dg',single_weight=.1,positive_weight=4,reduction='depth')
    torch.testing.assert_close(actual,expected)


def test_double_subsampling_cap():
    f = pd.DataFrame(dict(target_id=['p']*104,n_mut=[1]*2+[2]*102,
                          score=[0.,1.]+[1.,2.]+[-1.]*100))
    sampler = ProteinUniformSampler(f,200,seed=2,double_ratio=8)
    sample = sampler.draw()
    assert ((sample.n_mut==2) & (sample.score<0)).sum()==24 # 8*(two stabilizing+1)
    assert len(sample)==28


def training_rows():
    return pd.DataFrame(dict(target_id=['p']*4,score_set_id=['p']*4,
        mutant=['A1G','C2W','D3E','C2W:D3E'],n_mut=[1,1,1,2],score=[-1.,.5,.2,.4]))


@pytest.mark.parametrize('mode',['reverse','background'])
def test_reference_augmentation_uses_measured_energy_differences(mode):
    f = training_rows()
    aug = ReferenceAugmentor(f,{'p':'ACD'},mode=mode,fraction=1,bank_size=1,seed=3)
    transformed = aug.apply(f)
    assert set(aug.audit.target_id)=={'p'}
    energies = {mutate('ACD',r.mutant):r.score for r in f.itertuples()}
    energies['ACD']=0.
    for r in transformed.itertuples():
        reference = aug.sequences[r.target_id]
        seq = mutate(reference,r.mutant)
        assert r.score==pytest.approx(energies[seq]-energies[reference])
        assert r.n_mut==(r.mutant.count(':')+1 if r.mutant else 0)
    assert len(transformed)==len(f)

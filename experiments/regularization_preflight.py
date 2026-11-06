import torch
from mutant_cache import harness
from mutant_cache.backbone.esmc import ESMC
from mutant_cache.models import VariantEffectModel, VariantModelConfig
args = harness.build_parser().parse_args(['--lora-r', '4', '--lora-dropout', '0.3', '--head-weight-decay', '1.0', '--lora-weight-decay', '0.1'])
bb = ESMC(64, 4, 2)
bb.freeze_base(True)
harness.attach_lora(args, bb)
assert bb.lora_config.dropout == .3
m = VariantEffectModel(bb, VariantModelConfig(n_assays=1, minimal=True, head='sum_mean_count', dropout=.5))
groups = harness.build_param_groups(m, args)
assert {g['name']: g['weight_decay'] for g in groups} == {'head': 1., 'variant_lora': .1}
assert any(isinstance(x, torch.nn.Dropout) and x.p == .3 for x in bb.variant_adapters.modules())
assert any(isinstance(x, torch.nn.Dropout) and x.p == .5 for x in m.head_mlp.modules())
print('Regularization wiring verified: adapter dropout 0.3, head dropout 0.5, independent decay 1.0/0.1')

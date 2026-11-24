#!/usr/bin/env python3
"""How far is a training step with SDPA attention from the same step with flash-attention? One GPU, real weights,
deterministic mode, the same batch and seed: loss and every gradient with the backbone as loaded
(flash_attention_2), then after the `sdpa` fast path switched it. The two kernels compute the same attention, so
the difference should be at rounding level; two runs with flash-attention give the noise floor.

    python3 coverage/sdpa_switch.py <starVLA tree> <framework> <backbone dir> <out.json>
"""
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.pop('STARVLA_FAST', None)
tree, framework, model_dir, out = sys.argv[1:5]
sys.path.insert(0, tree)
os.chdir(tree)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from starVLA import fast  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
cfg = OmegaConf.load('examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml')
cfg.framework.name = framework
cfg.framework.qwenvl.base_vlm = model_dir
cfg.framework.qwenvl.attn_implementation = 'flash_attention_2'
model = build_framework(cfg).cuda().train()
rng = np.random.default_rng(0)
examples = [{
    'image': [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
    'lang': ' '.join(['pick up the bowl and place it on the plate'] * (1 + i % 3)),
    'action': rng.uniform(-1, 1, (16, 7)).astype(np.float32),
    'state': rng.uniform(-1, 1, (1, 7)).astype(np.float32),
} for i in range(16)]


def step():
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss = model(examples)['action_loss']
    loss.backward()
    return float(loss), torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])


def distance(a, b):
    return {'loss': [a[0], b[0]], 'loss_relative_difference': abs(a[0] - b[0]) / abs(a[0]),
            'gradient_relative_difference': float((a[1] - b[1]).norm() / a[1].norm()),
            'gradient_cosine': float(torch.nn.functional.cosine_similarity(a[1].double(), b[1].double(), dim=0))}


flash_a, flash_b = step(), step()
before = fast.attention_implementation(model.qwen_vl_interface.model.model.language_model, model.qwen_vl_interface.model)
os.environ['STARVLA_FAST'] = 'sdpa'
applied, skipped = fast._run('backbone', model.qwen_vl_interface)
after = fast.attention_implementation(model.qwen_vl_interface.model.model.language_model, model.qwen_vl_interface.model)
res = {'framework': framework, 'backbone': os.path.basename(model_dir.rstrip('/')), 'loaded_with': before,
       'applied': applied, 'after': after, 'gradient_elements': int(flash_a[1].numel()),
       'flash_vs_flash': distance(flash_a, flash_b), 'flash_vs_sdpa': distance(flash_a, step())}
open(out, 'w').write(json.dumps(res, indent=1) + '\n')
print(json.dumps(res))

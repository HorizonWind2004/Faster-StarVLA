#!/usr/bin/env python3
"""How far is a training step with a fast path that changes rounding from the same step without it? One GPU, real
weights, deterministic mode, the same batch and seed: loss and every gradient with the default code, then after the
named fast paths were applied. Two runs of the default code give the noise floor.

    python3 coverage/rounding_distance.py <starVLA tree> <framework> <backbone dir> <sdpa|flash_attention_2> \
        <fast paths, comma separated> <out.json>

CHECK_CONFIG names the training YAML to take the framework's settings from (default: the LIBERO recipe), CHECK_SET
lists further `key=value` settings, and CHECK_CHECKPOINT a checkpoint to load into the framework (PI0 / PI05, which
have no backbone directory: pass "-" for it). `attn:eager` in place of the fast paths switches the backbone to
that attention implementation instead: a change of implementation that is not one of ours, for comparison.
"""
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.pop('STARVLA_FAST', None)
tree, framework, model_dir, attn, paths, out = sys.argv[1:7]
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
cfg = OmegaConf.load(os.environ.get('CHECK_CONFIG', 'examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml'))
cfg.framework.name = framework
if model_dir != '-':
    cfg.framework.qwenvl.base_vlm = model_dir
    cfg.framework.qwenvl.attn_implementation = attn
for item in os.environ.get('CHECK_SET', '').split():
    OmegaConf.update(cfg, *item.split('=', 1), force_add=True)
model = build_framework(cfg)
if os.environ.get('CHECK_CHECKPOINT'):
    model.load_pretrained_checkpoint(os.environ['CHECK_CHECKPOINT'])
model = model.cuda().train()
head = model.config.framework.get('action_model', {})
rng = np.random.default_rng(0)
examples = [{
    'image': [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
    'lang': ' '.join(['pick up the bowl and place it on the plate'] * (1 + i % 3)),
    'action': rng.uniform(-1, 1, (int(os.environ.get('CHECK_ACTION_STEPS', '16')), int(head.get('action_dim', 7)))).astype(np.float32),
    'state': rng.uniform(-1, 1, (1, int(os.environ.get('CHECK_STATE_DIM', head.get('state_dim', 7))))).astype(np.float32),
} for i in range(16)]


def group(name):
    if '.visual.' in name or '.vision_tower.' in name:
        return 'vision tower'
    return 'language model' if name.startswith(('qwen_vl_interface.', 'vlm.')) else 'action head'


def step():
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    with torch.autocast('cuda', dtype=torch.bfloat16):  # as the trainer calls it
        loss = model(examples)['action_loss']
    loss.backward()
    return float(loss), {name: p.grad.detach().float().flatten() for name, p in model.named_parameters() if p.grad is not None}


def sums(a, b):
    """(a.a, b.b, a.b, |a - b|^2) in fp64, accumulated in chunks (the gradients are billions of elements)."""
    total = torch.zeros(4, dtype=torch.float64, device=a.device)
    for x, y in zip(a.split(1 << 26), b.split(1 << 26)):
        x, y = x.double(), y.double()
        total += torch.stack([x.dot(x), y.dot(y), x.dot(y), (x - y).pow(2).sum()])
    return total


def report(total):
    aa, bb, ab, dd = total.tolist()
    return {'norm': aa ** 0.5, 'relative_difference': (dd / aa) ** 0.5, 'cosine': ab / (aa * bb) ** 0.5}


def distance(a, b):
    """A parameter with a gradient in only one of the two steps counts as zero in the other and is listed: compiled
    code returns zeros for a parameter whose output the loss does not use, where eager autograd returns None."""
    parts, one_sided = {}, []
    for name in sorted(set(a[1]) | set(b[1])):
        x, y = a[1].get(name), b[1].get(name)
        if x is None or y is None:
            present = x if x is not None else y
            one_sided.append({'name': name, 'in': 'default' if y is None else 'fast', 'all_zero': not bool(present.any())})
            x, y = (present, torch.zeros_like(present)) if y is None else (torch.zeros_like(present), present)
        parts[group(name)] = parts.get(group(name), 0) + sums(x, y)
    whole = report(sum(parts.values()))
    return {'loss': [a[0], b[0]], 'loss_relative_difference': abs(a[0] - b[0]) / abs(a[0]),
            'gradient_relative_difference': whole['relative_difference'], 'gradient_cosine': whole['cosine'],
            'by_module': {key: report(value) for key, value in parts.items()},
            'gradient_in_one_step_only': {'parameters': len(one_sided), 'all_zero': all(e['all_zero'] for e in one_sided),
                                          'examples': one_sided[:6]}}


default_a = step()
noise_floor = distance(default_a, step())
if paths.startswith('attn:'):  # not a fast path: another attention implementation of the same model, as a yardstick
    backbone = model.qwen_vl_interface.model
    backbone.set_attn_implementation(paths[5:])
    for module in backbone.modules():
        if getattr(module, '_attn_implementation', None) == attn:
            module._attn_implementation = paths[5:]
    applied, skipped = [paths], []
else:
    os.environ['STARVLA_FAST'] = paths
    applied, skipped = fast._run('framework', model)
    if hasattr(model, 'qwen_vl_interface'):
        more = fast._run('backbone', model.qwen_vl_interface)
        applied, skipped = applied + more[0], skipped + more[1]
res = {'framework': framework, 'backbone': os.path.basename(model_dir.rstrip('/')), 'loaded_with': attn,
       'paths': paths, 'applied': applied, 'not_applicable': skipped,
       'gradient_elements': sum(int(value.numel()) for value in default_a[1].values()),
       'default_vs_default': noise_floor, 'default_vs_fast': distance(default_a, step())}
open(out, 'w').write(json.dumps(res, indent=1) + '\n')
print(json.dumps(res))

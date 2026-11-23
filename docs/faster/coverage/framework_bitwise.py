#!/usr/bin/env python3
"""A whole framework on real weights, one GPU: with the exact backbone fast paths, one training step (loss and every
gradient) and one predict_action call are bitwise identical to the default code. Same seed for the framework's own
randomness (flow-matching noise and time), bf16 autocast, a training-shaped batch.

    python3 coverage/framework_bitwise.py <starVLA tree> <framework> <backbone dir> <sdpa|flash_attention_2> <out.json>
"""
import hashlib
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.pop('STARVLA_FAST', None)
tree, framework, model_dir, attn, out = sys.argv[1:6]
batch = int(os.environ.get('CHECK_BATCH', '16'))
sys.path.insert(0, tree)
os.chdir(tree)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from starVLA import fast  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

PATHS = os.environ.get('CHECK_PATHS', 'attn_mask,hidden_only,vision_attn,vision_pos')
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
# the fused SDPA kernels have a non-deterministic backward (the action heads use them in fp32): use the math kernel,
# for the reference and the candidate alike, so that a run can be compared with a run at all
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
cfg = OmegaConf.load('examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml')
cfg.framework.name = framework
cfg.framework.qwenvl.base_vlm = model_dir
cfg.framework.qwenvl.attn_implementation = attn
model = build_framework(cfg).cuda()
rng = np.random.default_rng(0)
examples = []
for i in range(batch):
    examples.append({
        'image': [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
        'lang': ' '.join(['pick up the bowl and place it on the plate'] * (1 + i % 3)),
        'action': rng.uniform(-1, 1, (16, 7)).astype(np.float32),
        'state': rng.uniform(-1, 1, (1, 7)).astype(np.float32),
    })


def digest(t):
    t = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()


def train_step():
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss = model(examples)['action_loss']
    loss.backward()
    return float(loss), {n: digest(p.grad) for n, p in model.named_parameters() if p.grad is not None}


def predict():
    model.eval()
    torch.manual_seed(0)
    np.random.seed(0)
    obs = [{k: v for k, v in e.items() if k != 'action'} for e in examples]
    return digest(model.predict_action(examples=obs)['normalized_actions'])


ref_a, ref_b = train_step(), train_step()
ref_act_a, ref_act_b = predict(), predict()
os.environ['STARVLA_FAST'] = PATHS
applied, skipped = fast._run('backbone', model.qwen_vl_interface)
cand, cand_act = train_step(), predict()
diff = [n for n in ref_a[1] if ref_a[1][n] != cand[1].get(n)]
res = {
    'framework': framework, 'backbone': os.path.basename(model_dir.rstrip('/')), 'attention': attn, 'batch': batch,
    'applied': applied, 'not_applicable': skipped,
    'reference_reproducible': ref_a == ref_b and ref_act_a == ref_act_b,
    'reference_rerun_gradients_different': sum(ref_a[1][n] != ref_b[1][n] for n in ref_a[1]),
    'loss': [ref_a[0], cand[0]], 'loss_identical': ref_a[0] == cand[0],
    'gradients': len(ref_a[1]), 'gradients_different': len(diff), 'first_different': diff[:3],
    'actions_identical': ref_act_a == cand_act,
}
res['PASS'] = bool(res['reference_reproducible'] and res['loss_identical'] and not diff and res['actions_identical'])
open(out, 'w').write(json.dumps(res, indent=1) + '\n')
print(json.dumps(res))

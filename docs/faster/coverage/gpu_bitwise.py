#!/usr/bin/env python3
"""Real weights on one GPU: the exact backbone fast paths leave hidden_states[-1] and every gradient bitwise
identical. bf16 autocast, deterministic algorithms, a training-shaped batch (16 samples x 2 views of 224 x 224).

    python3 coverage/gpu_bitwise.py <starVLA tree> <backbone dir> <out.json>
"""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
tree, model_dir, out = sys.argv[1:4]
sys.path.insert(0, tree)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

PATHS = os.environ.get('CHECK_PATHS', 'attn_mask,hidden_only,vision_attn,vision_pos')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
DTYPE = os.environ.get('CHECK_DTYPE', 'bf16')
model = AutoModelForImageTextToText.from_pretrained(
    model_dir, attn_implementation='sdpa', dtype=torch.bfloat16 if DTYPE == 'bf16' else torch.float32)
model = model.cuda().train()
processor = AutoProcessor.from_pretrained(model_dir)
processor.tokenizer.padding_side = 'left'
rng = np.random.default_rng(0)
messages = []
for i in range(16):
    images = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    text = 'Your task is ' + ' '.join(['pick up the bowl and place it on the plate'] * (1 + i % 3)) + '.'
    messages.append([{'role': 'user', 'content': [{'type': 'image', 'image': im} for im in images]
                      + [{'type': 'text', 'text': text}]}])
inputs = processor.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors='pt').to('cuda')


def digest(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def run():
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DTYPE == 'bf16'):
        outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
    hidden.float().pow(2).mean().backward()
    grads = {n: digest(p.grad) for n, p in model.named_parameters() if p.grad is not None}
    return digest(hidden), grads, type(outputs).__name__


ref_a, ref_b = run(), run()
from starVLA import fast  # noqa: E402

os.environ['STARVLA_FAST'] = PATHS
kinds = os.environ.get('CHECK_VISION_KINDS')  # diagnosis: batch only 'frames' (full attention) or only 'windows'
if kinds:
    from starVLA.fast import vision_attn

    if kinds == 'frames':
        vision_attn._window_layout = lambda *a: None
    else:
        vision_attn._frame_layout = lambda *a: None
applied, skipped = fast._run('backbone', SimpleNamespace(model=model))
cand = run()
diff = [n for n in ref_a[1] if ref_a[1][n] != cand[1].get(n)]
res = {
    'backbone': os.path.basename(os.path.realpath(model_dir).rstrip('/')), 'model_class': type(model).__name__,
    'tokens': list(inputs['input_ids'].shape), 'images': int(inputs['image_grid_thw'].shape[0]),
    'grid': inputs['image_grid_thw'][0].tolist(), 'applied': applied, 'vision_kinds': kinds, 'dtype': DTYPE,
    'vision_attn_probes': {' x '.join(map(str, k[:2])) + f' ({k[2]} heads x {k[3]}, grad {k[6]})': v
                           for k, v in getattr(sys.modules.get('starVLA.fast.vision_attn'), 'decisions', {}).items()}, 'not_applicable': skipped,
    'reference_reproducible': ref_a[:2] == ref_b[:2], 'output_class': cand[2],
    'hidden_identical': ref_a[0] == cand[0], 'gradients': len(ref_a[1]), 'gradients_different': len(diff),
    'first_different': diff[:5],
    'PASS': ref_a[:2] == ref_b[:2] and ref_a[0] == cand[0] and not diff and sorted(applied) == sorted(PATHS.split(',')),
}
open(out, 'w').write(json.dumps(res, indent=1) + '\n')
print(json.dumps(res))

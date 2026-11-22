"""Digest what a tree's data pipeline yields, on the CPU: run with the tree as working directory and on PYTHONPATH.

  python loader_probe.py --out probe.json --batches B --samples N -- <train_starvla.py arguments>

Builds the config as the entry point does and calls the tree's build_dataloader. Records (a) dataset[i] for i < N and
(b) the first B batches of the DataLoader. A sample must be a dict holding 'action' (array), 'image' (list of PIL
images or arrays) and 'lang' (str); other keys are ignored, so a tree may add fields but not change these.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def digest(sample):
    if not isinstance(sample, dict) or not {'action', 'image', 'lang'} <= set(sample):
        raise SystemExit(f'a sample must be a dict with action / image / lang, got {type(sample).__name__}: '
                         f'{sorted(sample) if isinstance(sample, dict) else ""}')
    h = hashlib.sha256()
    action = np.asarray(sample['action'])
    h.update(f'{action.dtype}|{action.shape}|'.encode() + action.tobytes())
    for image in sample['image']:
        arr = np.asarray(image)
        h.update(f'|{arr.dtype}|{arr.shape}|'.encode() + arr.tobytes())
    h.update(b'|' + str(sample['lang']).encode())
    if 'state' in sample:
        state = np.asarray(sample['state'])
        h.update(f'|{state.dtype}|{state.shape}|'.encode() + state.tobytes())
    return h.hexdigest()[:24]


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--samples', type=int, default=96)
    a = p.parse_args(argv[:split])
    train = argv[split + 1:]
    sys.path.insert(0, '.')
    from omegaconf import OmegaConf
    from starVLA.dataloader import build_dataloader
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
    cfg = OmegaConf.load(train[train.index('--config_yaml') + 1])
    rest = [x for i, x in enumerate(train) if x != '--config_yaml' and (i == 0 or train[i - 1] != '--config_yaml')]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(rest)))
    cfg = apply_config_compat(cfg)
    cfg.output_dir = tempfile.mkdtemp(prefix='probe-out-')
    loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    out = {'len_dataset': len(loader.dataset), 'len_loader': len(loader), 'batch_size': loader.batch_size,
           'samples': [], 'batches': []}
    it = iter(loader)                  # workers are forked before this process decodes any video (PyAV + fork hangs)
    for _ in range(a.batches):
        batch = next(it)
        if not isinstance(batch, (list, tuple)):
            raise SystemExit(f'a batch must be a list of samples, got {type(batch).__name__}')
        out['batches'].append([digest(s) for s in batch])
    out['samples'] = [digest(loader.dataset[i]) for i in range(a.samples)]
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'samples': len(out['samples']), 'batches': len(out['batches'])}))


if __name__ == '__main__':
    main()
    import os
    os._exit(0)                                  # do not wait for persistent data-loader workers

"""Step timing of a source tree's real starVLA entry point (starVLA/training/train_starvla.py).

Started by `accelerate launch` with the tree as working directory and on PYTHONPATH:

  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 2 \
      <bench>/timing.py --out <prefix> --warmup W --window N --repeats R -- <train_starvla.py arguments>

The tree's train_starvla.py runs unmodified as __main__ in this process. A training step is one learning-rate
scheduler step (the reference steps its scheduler once per optimizer step; the learning rates are exact items of the
judge, so every valid candidate does too). After W steps, R consecutive windows of N steps are timed; window edges
are bracketed by torch.cuda.synchronize(), nothing else is synchronized by the bench. After the last window the
process writes <prefix>.r<rank>.json and exits. Per step it keeps the loss handed to DeepSpeedEngine.backward (a
device tensor, read after the last window) and the host time (diagnostics).

--stages K: after the warmup, K steps with torch.cuda.synchronize() at the edges of DeepSpeedEngine.forward, .backward
and .step; the mean wall time of each stage and of the rest of the loop (batch fetch, logging, host reads) goes to
<prefix>.stages.r<rank>.json. The syncs remove the overlap between host and GPU work, so the stage times add up to
more than a timed step; they show where the work is, not the step time.

--cprofile K: cProfile of the main thread over the first K steps after the warmup (rank 0 writes <prefix>.prof and a
text table); the run stops after them. With CUDA_LAUNCH_BLOCKING=1 in the environment GPU time is charged to the
calling Python frame.
"""
import argparse
import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path

import torch

T_START = time.perf_counter()


def compile_count():
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


def kill_descendants():
    children = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                ppid = int(f.read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    stack, found = [os.getpid()], []
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(found)


def rank_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))


def fingerprint(train_argv):
    import importlib.metadata as md
    versions = {}
    for pkg in ('torch', 'torchvision', 'transformers', 'accelerate', 'deepspeed', 'flash_attn', 'numpy', 'av',
                'pillow', 'triton'):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    dev = torch.cuda.current_device()
    return {'argv': train_argv, 'versions': versions, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(dev), 'gpu_count': torch.cuda.device_count(),
            'math_mode': {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                          'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                          'float32_matmul_precision': torch.get_float32_matmul_precision(),
                          'cudnn_benchmark': torch.backends.cudnn.benchmark,
                          'deterministic': torch.are_deterministic_algorithms_enabled()},
            'cpus': len(os.sched_getaffinity(0)), 'threads': torch.get_num_threads()}


def main():
    argv = sys.argv[1:]
    split = argv.index('--')
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--entry', default='starVLA/training/train_starvla.py')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--cprofile', type=int, default=0)
    p.add_argument('--stages', type=int, default=0)
    a = p.parse_args(argv[:split])
    train_argv = argv[split + 1:]

    import deepspeed.runtime.engine as ds_engine
    import torch.optim.lr_scheduler as lrs

    st = {'edges': {}, 'wall': [], 'losses': [], 'first_step_s': None, 'prof': None}
    last = a.warmup + (a.cprofile or a.stages or a.repeats * a.window)
    stages = {'on': False, 't': None, 'acc': {}, 'steps': 0}

    def mark(name):
        if stages['on']:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if stages['t'] is not None:
                stages['acc'][name] = stages['acc'].get(name, 0.0) + now - stages['t']
            stages['t'] = now

    def staged(cls, method, name):
        real = getattr(cls, method)

        def wrapped(self, *args, **kwargs):
            mark('loop')
            out = real(self, *args, **kwargs)
            mark(name)
            return out
        setattr(cls, method, wrapped)

    real_backward = ds_engine.DeepSpeedEngine.backward

    def backward(self, loss, *args, **kwargs):
        st['losses'].append(loss.detach())
        return real_backward(self, loss, *args, **kwargs)
    ds_engine.DeepSpeedEngine.backward = backward
    if a.stages:
        staged(ds_engine.DeepSpeedEngine, 'forward', 'forward')
        staged(ds_engine.DeepSpeedEngine, 'backward', 'backward')
        staged(ds_engine.DeepSpeedEngine, 'step', 'optimizer_step')

    def finish():
        rank, world = rank_world()
        edges = st['edges']
        windows = []
        if a.stages:
            Path(f'{a.out}.stages.r{rank}.json').write_text(json.dumps(
                {'steps': a.stages, 'ms_per_step': {k: 1000 * v / a.stages for k, v in stages['acc'].items()}}) + '\n')
        if not (a.cprofile or a.stages):
            for k in range(a.repeats):
                (t0, c0), (t1, c1) = edges[k], edges[k + 1]
                windows.append({'seconds': t1 - t0, 'step_ms': 1000 * (t1 - t0) / a.window, 'compiles': c1 - c0})
        losses = [float(x.float()) for x in st['losses']]
        out = {'rank': rank, 'world_size': world, 'warmup': a.warmup, 'window': a.window, 'repeats': a.repeats,
               'windows': windows, 'first_step_s': st['first_step_s'], 'step_wall': st['wall'], 'losses': losses,
               'peak_mem_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
               'reserved_gib': torch.cuda.memory_reserved() / 2 ** 30,
               'compiles_total': compile_count(), 'fingerprint': fingerprint(train_argv)}
        if st['prof'] is not None:
            import io
            import pstats
            st['prof'].disable()
            if rank == 0:
                st['prof'].dump_stats(a.out + '.prof')
                for sort in ('cumulative', 'tottime'):
                    s = io.StringIO()
                    pstats.Stats(st['prof'], stream=s).sort_stats(sort).print_stats(70)
                    Path(f'{a.out}.{sort}.txt').write_text(s.getvalue())
            out['cprofile_s'] = edges['p1'] - edges['p0']
        Path(f'{a.out}.r{rank}.json').write_text(json.dumps(out) + '\n')
        sys.stdout.flush()
        sys.stderr.flush()
        kill_descendants()
        os._exit(0)

    real_step = lrs.LRScheduler.step

    def step(self, *args, **kwargs):
        out = real_step(self, *args, **kwargs)
        g = self._step_count - 1               # optimizer steps completed (0 during the scheduler's own __init__)
        if g <= 0:
            return out
        st['wall'].append(time.perf_counter() - T_START)
        if g == 1:
            torch.cuda.synchronize()
            st['first_step_s'] = time.perf_counter() - T_START
        if a.stages:
            if g == a.warmup:
                stages['on'] = True
                mark('loop')
            elif g == last:
                mark('loop')
        elif a.cprofile:
            if g == a.warmup:
                import cProfile
                torch.cuda.synchronize()
                st['edges']['p0'] = time.perf_counter()
                st['prof'] = cProfile.Profile()
                st['prof'].enable()
            elif g == last:
                torch.cuda.synchronize()
                st['edges']['p1'] = time.perf_counter()
        elif g >= a.warmup and (g - a.warmup) % a.window == 0:
            torch.cuda.synchronize()
            st['edges'][(g - a.warmup) // a.window] = (time.perf_counter(), compile_count())
        if g == last:
            finish()
        return out
    lrs.LRScheduler.step = step

    entry = str(Path(a.entry).resolve())
    sys.argv = [entry, *train_argv]
    sys.path.insert(0, os.getcwd())
    runpy.run_path(entry, run_name='__main__')
    raise SystemExit('training finished before the timing plan did')


if __name__ == '__main__':
    main()

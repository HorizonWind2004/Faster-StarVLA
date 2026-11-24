#!/usr/bin/env python3
"""Time a training configuration other than the judge's (another framework and/or backbone) with the judge's timing
code: same entry point, warmup, windows and periodic evaluation; only the listed train arguments differ.

    python3 coverage/time_cfg.py --tree starVLA --fast exact --out runs/cov/gr00t_q3 --tag exact \
        --set framework.name=QwenGR00T [--model /path/to/backbone]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / 'bench'))
import bench  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tree', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--fast', default='')
ap.add_argument('--model')
ap.add_argument('--set', nargs='*', default=[])
ap.add_argument('--cprofile', type=int, default=0, help='cProfile of rank 0 over this many steps, with CUDA_LAUNCH_BLOCKING=1')
a = ap.parse_args()
a.out = a.out if a.out.startswith('/') else str(WS / a.out)  # the training process runs in the tree

# extra steps handed to the allocation's batch script (the only place that sees both GPUs): run once, in order
if not os.environ.get('NO_INJECT'):
    for script in sorted((WS / 'coverage' / 'inject').glob('*.sh')):
        script.rename(script.with_suffix('.sh.running'))
        subprocess.run(['bash', str(script.with_suffix('.sh.running'))], cwd=WS)
        script.with_suffix('.sh.running').rename(script.with_suffix('.sh.done'))
skip = WS / 'coverage' / 'skip.txt'
if skip.exists() and Path(a.out).name in skip.read_text().split():
    print(json.dumps({'status': 'SKIPPED', 'tag': a.tag, 'out': a.out}))
    sys.exit(0)
args = list(bench.CFG['train_args'])
pairs = {args[i]: i + 1 for i in range(0, len(args), 2)}
for item in a.set:
    flag, value = item.split('=', 1)
    flag = '--' + flag
    if flag in pairs:
        args[pairs[flag]] = value
    else:
        args += [flag, value]
bench.CFG['train_args'] = args
if a.model:
    bench.CFG['assets'] = dict(bench.CFG['assets'], model=a.model)
switches = {'STARVLA_FAST': a.fast} if a.fast else {}
with bench.GpuLock():
    res = bench.timing_run(a.tree if a.tree.startswith('/') else str(WS / a.tree), switches, a.out, a.tag,
                           **({'cprofile': a.cprofile, 'extra_env': {'CUDA_LAUNCH_BLOCKING': '1'}} if a.cprofile else {}))
res['train_overrides'] = a.set
res['fast'] = a.fast
Path(a.out, f'{a.tag}_summary.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps({k: res.get(k) for k in ('status', 'step_ms', 'window_step_ms', 'peak_mem_gib',
                                          'gpu_util_mean_second_half', 'error')} | {'tag': a.tag, 'out': a.out}))

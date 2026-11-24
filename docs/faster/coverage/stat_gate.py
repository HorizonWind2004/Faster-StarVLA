#!/usr/bin/env python3
"""The judge's statistical gate (gate 3) for a framework other than the headline recipe.

The frozen judge compares a candidate with upstream on the headline recipe only. This runs the same comparison with
the judge's own recorder, trajectory case (60 training steps in deterministic mode, the recipe's periodic evaluation
included) and endpoints, for another framework:

  1. upstream on six seeds: the seed-to-seed standard deviation of every endpoint;
  2. each candidate setting on seed 0: the difference to upstream on the same seed must be within 3 sigma on every
     endpoint (the judge's criterion).

    python3 coverage/stat_gate.py <framework> <out dir> <candidate tree> <STARVLA_FAST value> [<value> ...]

Nothing of the judge is modified; its launcher and recorder are imported and given other training arguments.
"""
import json
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
from bench import bench as judge  # noqa: E402

framework, out, tree, settings = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3], sys.argv[4:]
out.mkdir(parents=True, exist_ok=True)
trainable = out / 'trainable.json'
judge.CFG['numerics']['adapter'] = 'coverage.stat_adapter:Adapter'
spec = json.loads(judge.rel(judge.CFG['trajectory']['cases']).read_text())
seeds = judge.CFG['trajectory']['reference_seeds']
sigma_k = judge.CFG['trajectory']['sigma_k']
reference = judge.CFG['reference']


def record(tree, fast, folder, seed):
    case = dict(spec, name=f'traj_s{seed}', seed=seed, args={**spec.get('args', {}), '--framework.name': framework})
    switches = {'STAT_TRAINABLE': str(trainable), **({'STARVLA_FAST': fast} if fast else {})}
    result = judge.numerics_record(tree, switches, out / folder, [case], losses_only=True,
                                   timeout=judge.TIMEOUT['trajectory'])
    print(f'{folder} seed {seed}: ok={result["ok"]} {result["seconds"]:.0f}s', flush=True)
    return result['ok']


def numerics(*args):
    env = judge.base_env()
    env['PYTHONPATH'] = str(WS)
    subprocess.run([judge.PY, '-m', 'numerics', *args], cwd=WS, env=env, check=True)


if not trainable.exists():  # the adapter writes the list from the reference tree and stops
    record(reference, '', 'formulation', seeds[0])
    if not trainable.exists():
        raise SystemExit(f'the recorder did not run for {framework}: see {out}/formulation/')
if not all(record(reference, '', 'ref', seed) for seed in seeds):
    raise SystemExit('a reference run failed')
numerics('traj-freeze', '--ref', str(out / 'ref'), '--spec', str(judge.rel(judge.CFG['trajectory']['spec'])), '--k', str(sigma_k),
         '--out', str(out / 'frozen.json'))
summary = {'framework': framework, 'seeds': seeds, 'sigma_k': sigma_k, 'candidates': {}}
for fast in settings:
    folder = 'cand_' + fast.replace(',', '_')
    if not record(tree, fast, folder, seeds[0]):
        summary['candidates'][fast] = {'verdict': 'run failed'}
        continue
    numerics('traj-compare', '--frozen', str(out / 'frozen.json'), '--cand', str(out / folder), '--sigma-k', str(sigma_k),
             '--out', str(out / f'{folder}_compare.json'))
    report = json.loads((out / f'{folder}_compare.json').read_text())
    summary['candidates'][fast] = {
        'verdict': report['verdict'],
        'endpoints': {name: {'difference_in_sigma': e.get('mean_in_sigma'), 'state': e['state']}
                      for name, e in report['endpoints'].items()}}
    print(fast, report['verdict'], {n: round(e['difference_in_sigma'], 2) if e['difference_in_sigma'] is not None else None
                                    for n, e in summary['candidates'][fast]['endpoints'].items()}, flush=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

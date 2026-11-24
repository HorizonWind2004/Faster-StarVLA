#!/usr/bin/env python3
"""The judge's statistical gate (gate 3) for a framework other than the headline recipe.

The frozen judge compares a candidate with upstream on the headline recipe only. This runs the same comparison with
the judge's own recorder, trajectory case (60 training steps in deterministic mode, the recipe's periodic evaluation
included) and endpoints, for another framework:

  1. upstream on six seeds: the seed-to-seed standard deviation of every endpoint;
  2. each candidate setting on seed 0: the difference to upstream on the same seed must be within 3 sigma on every
     endpoint (the judge's criterion).

  3. with a candidate on several seeds (`<value>:<number of seeds>`), also the judge's paired-seed test: the mean of
     the differences over the seeds, per endpoint, against the same limit.

    python3 coverage/stat_gate.py <framework> <out dir> <candidate tree> <STARVLA_FAST value>[:<seeds>] ...

Recordings that exist in <out dir> are kept, so more seeds or settings can be added later. Nothing of the judge is
modified; its launcher and recorder are imported and given other training arguments.
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
    if all((out / folder / f'traj_s{seed}.r{rank}.json').exists() for rank in range(judge.CFG['world_size'])):
        return True
    case = dict(spec, name=f'traj_s{seed}', seed=seed, args={**spec.get('args', {}), '--framework.name': framework})
    switches = {'STAT_TRAINABLE': str(trainable), **({'STARVLA_FAST': fast} if fast else {})}
    result = judge.numerics_record(tree, switches, out / folder, [case], losses_only=True,
                                   timeout=judge.TIMEOUT['trajectory'])
    print(f'{folder} seed {seed}: ok={result["ok"]} {result["seconds"]:.0f}s', flush=True)
    return result['ok']


def numerics(*args):
    env = judge.base_env()
    env['PYTHONPATH'] = str(WS)
    subprocess.run([judge.PY, '-m', 'numerics', *args], cwd=WS, env=env, stdout=subprocess.DEVNULL)  # the verdict is in --out


if not trainable.exists():  # the adapter writes the list from the reference tree and stops
    record(reference, '', 'formulation', seeds[0])
    if not trainable.exists():
        raise SystemExit(f'the recorder did not run for {framework}: see {out}/formulation/')
if not all(record(reference, '', 'ref', seed) for seed in seeds):
    raise SystemExit('a reference run failed')
numerics('traj-freeze', '--ref', str(out / 'ref'), '--spec', str(judge.rel(judge.CFG['trajectory']['spec'])), '--k', str(sigma_k),
         '--out', str(out / 'frozen.json'))
summary = {'framework': framework, 'seeds': seeds, 'sigma_k': sigma_k, 'candidates': {}}


def endpoints(report):
    return {name: {'difference_in_sigma': e.get('mean_in_sigma'), 'state': e['state'], 'differences': e.get('diffs')}
            for name, e in report['endpoints'].items()}


for setting in settings:
    fast, _, count = setting.partition(':')
    folder = 'cand_' + fast.replace(',', '_').replace('-', 'no_')
    done = [seed for seed in seeds[:int(count or 1)] if record(tree, fast, folder, seed)]
    if not done:
        summary['candidates'][fast] = {'verdict': 'run failed'}
        continue
    # the judge's own criterion: the mean paired difference over the seeds run, within sigma_k x sigma
    numerics('traj-compare', '--frozen', str(out / 'frozen.json'), '--cand', str(out / folder), '--sigma-k', str(sigma_k),
             '--out', str(out / f'{folder}_compare.json'))
    report = json.loads((out / f'{folder}_compare.json').read_text())
    entry = {'seeds': report['seeds'], 'verdict': report['verdict'], 'endpoints': endpoints(report)}
    if len(report['seeds']) > 1:  # paired-seed test
        numerics('traj-compare', '--frozen', str(out / 'frozen.json'), '--cand', str(out / folder),
                 '--out', str(out / f'{folder}_paired.json'))
        paired = json.loads((out / f'{folder}_paired.json').read_text())
        entry['paired_test'] = {'verdict': paired['verdict'],
                                'endpoints': {n: {'state': e['state'], 'p_paired_t': e['p_paired_t']} for n, e in paired['endpoints'].items()}}
    summary['candidates'][fast] = entry
    print(fast, 'seeds', report['seeds'], report['verdict'],
          {n: round(e['difference_in_sigma'], 2) if e['difference_in_sigma'] is not None else None for n, e in entry['endpoints'].items()},
          flush=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')

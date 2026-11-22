"""starVLA QwenOFT LIBERO training bench on 2 GPUs: one command per question.

Run from the workspace root, inside the GPU allocation, with any Python 3. GPU runs take the lock in bench.json
themselves, one at a time; do NOT wrap these commands in flock.

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick | --skip-timing | --n1]
        the verdict (gates 0-3 below) -> <out>/check.json
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
        one timing run of the tree (2 windows of 100 steps after 20 warmup steps), with its speed relative to the
        frozen reference timing -> <out>/time.json
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--steps K] [--blocking | --stages]
        cProfile of rank 0's main thread over K steps after the warmup (--blocking: CUDA_LAUNCH_BLOCKING=1, so GPU
        time is charged to the calling frame) and GPU utilization -> <out>/profile.*.txt; --stages: instead, wall
        time per stage (DeepSpeed forward / backward / optimizer step / rest of the loop) with a GPU sync at every
        stage edge -> <out>/profile.stages.json
        (the PyTorch profiler / nsys cannot see kernels on this GPU generation: CUPTI has no sm_103 support)
  python bench/bench.py baseline --out <dir> [--freeze]   (formulator) the reference timed reference_runs times

Every GPU run holds the GPU lock, waits until the GPUs are empty, runs the code from a fresh copy of the tree in its
own process group with a timeout, a fresh TMPDIR and cold compile caches, and kills what it started afterwards.
Timing runs carry a fingerprint (argv, switches, math mode, GPU, library versions), are scanned for invalid-run
signatures and fail if torch.compile compiled anything inside a timing window.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's processes only. Variables in bench.json reserved_env are refused.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure   the entry point's setup (bench.json coverage.entry_functions, the module level, and
                 VLATrainer.train outside its per-step loop body) must equal the reference AST, and the tree's data
                 pipeline must yield bitwise the reference's samples and first batches -> else ERROR / INCORRECT.
  1. benefit     check: one fresh run of the candidate against the frozen reference timing. check --final (harness
                 only): reference, candidate, reference; fingerprints must agree (else ERROR); a reference more than
                 timing.frozen_band off the frozen one, two references more than timing.pair_band apart, or a run
                 whose two windows differ by more than timing.window_band -> INVALID (measure again). speedup <=
                 max(1.02, 1 + 2 x reference spread) -> UNAVAILABLE (no_benefit), precision not evaluated.
  2. bitwise     the cases of bench/cases.json recorded in deterministic mode. Exact items (RNG fingerprints,
                 counters, invariants incl. learning rates and what was fed to the VLM, tensor names and shapes,
                 finiteness, cross-rank pattern) must hold, else INCORRECT. Every recorded tensor (loss/, grad/,
                 delta/, optim/) bitwise identical -> PASS (N0).
  3. statistical only when gate 2 is not bitwise: one paired trajectory on the first frozen seed in deterministic
                 mode; every endpoint's |candidate - reference| <= 3 sigma (sigma: the reference's own seed-to-seed
                 standard deviation, frozen from 6 seeds) -> PASS (N1, or N2 when a math mode or dtype changed),
                 else INCORRECT.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
CFG = json.loads((BENCH / 'bench.json').read_text())
PY = CFG['python']
ACCELERATE = str(Path(PY).parent / 'accelerate')
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG['invalid_run_signatures']]
TREE_SKIP = {'.git', '__pycache__', 'wandb', 'playground', 'results', '.pytest_cache'}
TIMEOUT = CFG['timeouts_s']
T = CFG['timing']


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def rel(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    """Joint hash of the frozen evaluator: numerics/ and bench/ code and config, and the frozen records in task/."""
    files = sorted(p for p in (ROOT / 'numerics').rglob('*.py'))
    files += sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix in ('.py', '.json'))
    files += sorted(p for p in (ROOT / 'task').rglob('*.json') if p.name != 'formulation.json')
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


def tree_digest(tree):
    h = hashlib.sha256()
    for p in sorted(Path(tree).rglob('*')):
        if p.is_file() and not (set(p.relative_to(tree).parts) & TREE_SKIP):
            h.update(str(p.relative_to(tree)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


_ENV = None


def base_env():
    global _ENV
    if _ENV is None:
        env_file = os.path.expandvars(CFG['env_file'].replace('$ASSETS', os.environ.get(
            'ASSETS', str(ROOT.parent / 'assets'))))
        out = subprocess.run(['bash', '-c', f'source {env_file} && env -0'], capture_output=True, check=True).stdout
        _ENV = dict(item.split('=', 1) for item in out.decode().split('\0') if '=' in item)
    return dict(_ENV)


def load_switches(config):
    if not config:
        return {}
    env = json.loads(Path(config).read_text()).get('environment', {})
    bad = sorted(set(env) & RESERVED)
    if bad:
        raise SystemExit(f'switch file sets variables the bench manages: {bad}')
    return {str(k): str(v) for k, v in env.items()}


class GpuLock:
    def __enter__(self):
        self.f = open(CFG['gpu_lock'], 'w')
        fcntl.flock(self.f, fcntl.LOCK_EX)
        deadline = time.time() + TIMEOUT['idle_wait']
        while True:
            used = [int(x) for x in subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True).stdout.split()]
            if len(used) < CFG['world_size']:
                raise SystemExit(f'need {CFG["world_size"]} GPUs, see {len(used)}: run inside the GPU allocation')
            if max(used) < 2000:
                return self
            if time.time() > deadline:
                raise SystemExit(f'INVALID: GPUs not idle after {TIMEOUT["idle_wait"]} s (memory used {used} MiB)')
            time.sleep(5)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


class Run:
    """A fresh copy of a tree plus a fresh TMPDIR (compile caches inside) on the scratch disk."""

    def __init__(self, tree):
        self.dir = Path(tempfile.mkdtemp(prefix='starvla-bench-', dir=CFG['scratch']))
        self.tree = self.dir / 'tree'
        shutil.copytree(tree, self.tree, ignore=lambda d, names: [n for n in names if n in TREE_SKIP])
        self.digest = tree_digest(self.tree)
        self.tmp = self.dir / 'tmp'
        self.tmp.mkdir()

    def env(self, switches, pythonpath):
        env = base_env()
        env.update(switches)
        env.update({'PYTHONPATH': str(pythonpath), 'TMPDIR': str(self.tmp),
                    'TORCHINDUCTOR_CACHE_DIR': str(self.tmp / 'inductor'), 'TRITON_CACHE_DIR': str(self.tmp / 'triton'),
                    'ASSETS': env['ASSETS']})
        return env

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def execute(cmd, cwd, env, log_path, timeout):
    """Run in its own process group; kill the group afterwards (data-loader workers) and on timeout."""
    with open(log_path, 'w') as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = 'timeout'
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    return rc


def port():
    return str(20000 + (os.getpid() * 7 + int(time.time())) % 20000)


def scan(log_path):
    text = Path(log_path).read_text(errors='replace')
    return sorted({reason for rx, reason in SIGNATURES if rx.search(text)})


def train_argv(out_dir, overrides=None):
    """The fixed train_starvla.py arguments with {placeholders} filled and `overrides` ({flag: value}) applied."""
    assets = base_env()['ASSETS']
    paths = {k: v.replace('$ASSETS', assets) for k, v in CFG['assets'].items()}
    paths['out'] = str(out_dir)
    argv = [x.format(**paths) for x in CFG['train_args']]
    pairs = [[argv[i], argv[i + 1]] for i in range(0, len(argv), 2)]
    for flag, value in (overrides or {}).items():
        for pair in pairs:
            if pair[0] == flag:
                pair[1] = str(value).format(**paths)
                break
        else:
            pairs.append([flag, str(value).format(**paths)])
    return [x for pair in pairs for x in pair]


# ---------------------------------------------------------------------------------------------- timing

def timing_run(tree, switches, out, tag, cprofile=0, extra_env=None, stages=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(tree)
    try:
        prefix = out / tag
        env = run.env(switches, run.tree)
        env.update(extra_env or {})
        cmd = [ACCELERATE, 'launch', '--config_file', CFG['accelerate_config'], '--num_processes',
               str(CFG['world_size']), '--main_process_port', port(), str(BENCH / 'timing.py'), '--out', str(prefix),
               '--entry', CFG['entry'], '--warmup', str(T['warmup']), '--window', str(T['window']),
               '--repeats', str(T['repeats'])]
        if cprofile:
            cmd += ['--cprofile', str(cprofile)]
        if stages:
            cmd += ['--stages', str(stages)]
        cmd += ['--', *train_argv(run.tmp / 'out')]
        util_log = open(out / f'{tag}.gpu_util.csv', 'w')
        smi = subprocess.Popen(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                '--format=csv,noheader,nounits', '-l', '2'], stdout=util_log, stderr=subprocess.DEVNULL)
        t0 = time.time()
        rc = execute(cmd, run.tree, env, out / f'{tag}.log', TIMEOUT['time'])
        smi.kill()
        util_log.close()
        res = {'tag': tag, 'returncode': rc, 'seconds': time.time() - t0, 'switches': switches,
               'tree_digest': run.digest, 'invalid': scan(out / f'{tag}.log')}
        ranks = [json.loads(p.read_text()) for p in sorted(out.glob(f'{tag}.r*.json'))]
        if rc != 0 or len(ranks) != CFG['world_size']:
            res['status'] = 'ERROR'
            res['error'] = f'returncode {rc}, {len(ranks)} rank reports (see {out}/{tag}.log)'
            return res
        if tree_digest(run.tree) != run.digest:
            res['status'] = 'ERROR'
            res['error'] = 'the tree copy changed during the run'
            return res
        res['fingerprint'] = ranks[0]['fingerprint']
        res['peak_mem_gib'] = max(r['peak_mem_gib'] for r in ranks)
        res['first_step_s'] = max(r['first_step_s'] for r in ranks)
        res['losses_rank0'] = ranks[0]['losses'][:T['warmup'] + T['repeats'] * T['window']]
        util = [int(line.split(',')[1]) for line in (out / f'{tag}.gpu_util.csv').read_text().splitlines() if ',' in line]
        tail = util[len(util) // 2:]
        res['gpu_util_mean_second_half'] = round(statistics.mean(tail), 1) if tail else None
        if stages:
            per_rank = [json.loads(p.read_text())['ms_per_step'] for p in sorted(out.glob(f'{tag}.stages.r*.json'))]
            res['status'] = 'OK'
            res['stage_ms_per_step'] = {k: round(max(r.get(k, 0.0) for r in per_rank), 1) for k in per_rank[0]}
            res['stage_ms_sum'] = round(sum(res['stage_ms_per_step'].values()), 1)
            return res
        if cprofile:
            res['status'] = 'OK'
            res['cprofile_s_per_step'] = ranks[0]['cprofile_s'] / cprofile
            return res
        res['window_step_ms'] = [max(r['windows'][k]['step_ms'] for r in ranks) for k in range(T['repeats'])]
        res['step_ms'] = statistics.mean(res['window_step_ms'])
        res['compiles_in_windows'] = sum(w['compiles'] for r in ranks for w in r['windows'])
        res['window_ratio'] = max(res['window_step_ms']) / min(res['window_step_ms'])
        res['status'] = 'OK'
        if res['invalid'] or res['compiles_in_windows']:
            res['status'] = 'INVALID'
            res['error'] = f'invalid signatures {res["invalid"]}, compiles in windows {res["compiles_in_windows"]}'
        return res
    finally:
        run.close()


def frozen_timing():
    p = rel(T['frozen'])
    return json.loads(p.read_text()) if p.exists() else None


def fingerprint_equal(a, b):
    keys = ('versions', 'cudnn', 'gpu', 'gpu_count', 'cpus')

    def argv(f):                                   # the output directory is per run
        return [x for i, x in enumerate(f['argv']) if i == 0 or f['argv'][i - 1] != '--run_root_dir']
    return all(a.get(k) == b.get(k) for k in keys) and argv(a) == argv(b)


def cmd_time(a):
    switches = load_switches(a.config)
    with GpuLock():
        res = timing_run(a.tree, switches, a.out, 'time')
    fr = frozen_timing()
    if res['status'] == 'OK' and fr:
        res['reference_step_ms'] = fr['step_ms']
        res['speedup'] = fr['step_ms'] / res['step_ms']
    Path(a.out, 'time.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res.get(k) for k in ('status', 'step_ms', 'window_step_ms', 'speedup', 'peak_mem_gib',
                                              'gpu_util_mean_second_half', 'first_step_s', 'error')}))
    return 0 if res['status'] == 'OK' else 1


def cmd_baseline(a):
    runs = []
    with GpuLock():
        for i in range(T['reference_runs']):
            runs.append(timing_run(rel(CFG['reference']), {}, a.out, f'ref{i}'))
            log(f'reference run {i}: {runs[-1].get("status")} {runs[-1].get("window_step_ms")}')
    if any(r['status'] != 'OK' for r in runs):
        print(json.dumps([r.get('error') for r in runs]))
        return 1
    ms = [r['step_ms'] for r in runs]
    frozen = {'step_ms': statistics.mean(ms), 'runs_step_ms': ms, 'spread': (max(ms) - min(ms)) / statistics.mean(ms),
              'window_step_ms': [r['window_step_ms'] for r in runs],
              'window_ratio': statistics.mean(r['window_ratio'] for r in runs),
              'peak_mem_gib': max(r['peak_mem_gib'] for r in runs), 'fingerprint': runs[0]['fingerprint'],
              'gpu_util': [r['gpu_util_mean_second_half'] for r in runs],
              'losses_rank0_first8': runs[0]['losses_rank0'][:8], 'plan': {k: T[k] for k in ('warmup', 'window', 'repeats')}}
    Path(a.out, 'baseline.json').write_text(json.dumps(frozen, indent=1) + '\n')
    if a.freeze:
        rel(T['frozen']).parent.mkdir(parents=True, exist_ok=True)
        rel(T['frozen']).write_text(json.dumps(frozen, indent=1) + '\n')
    print(json.dumps({k: frozen[k] for k in ('step_ms', 'runs_step_ms', 'spread', 'peak_mem_gib', 'gpu_util')}))
    return 0


def cmd_profile(a):
    switches = load_switches(a.config)
    extra = {'CUDA_LAUNCH_BLOCKING': '1'} if a.blocking else {}
    tag = 'profile.stages' if a.stages else 'profile.blocking' if a.blocking else 'profile'
    with GpuLock():
        if a.stages:
            res = timing_run(a.tree, switches, a.out, tag, stages=a.steps)
        else:
            res = timing_run(a.tree, switches, a.out, tag, cprofile=a.steps, extra_env=extra)
    Path(a.out, f'{tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res.get(k) for k in ('status', 'cprofile_s_per_step', 'stage_ms_per_step', 'stage_ms_sum',
                                              'gpu_util_mean_second_half', 'error')}))
    if not a.stages:
        print(f'tables: {a.out}/{tag}.cumulative.txt {a.out}/{tag}.tottime.txt')
    return 0 if res['status'] == 'OK' else 1


# ---------------------------------------------------------------------------------------------- gate 0

def _functions(tree_ast):
    out = {}
    for node in tree_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f'{node.name}.{sub.name}'] = sub
    return out


def _strip_loop(fn):
    """The function with the body of its first `while` loop replaced by `pass` (the per-step loop is free)."""
    fn = ast.parse(ast.unparse(fn)).body[0]
    for node in ast.walk(fn):
        if isinstance(node, ast.While):
            node.body = [ast.Pass()]
            break
    return fn


def structure(tree):
    cov = CFG['coverage']
    problems = []
    ref_ast = ast.parse((rel(CFG['reference']) / CFG['entry']).read_text())
    try:
        cand_ast = ast.parse((Path(tree) / CFG['entry']).read_text())
    except (OSError, SyntaxError) as e:
        return [f'{CFG["entry"]}: {e}']
    rf, cf = _functions(ref_ast), _functions(cand_ast)
    for name in cov['entry_functions']:
        if name not in cf or ast.dump(rf[name]) != ast.dump(cf[name]):
            problems.append(f'{name} differs from the reference')
    owner = cov['loop_owner']
    if owner not in cf or ast.dump(_strip_loop(rf[owner])) != ast.dump(_strip_loop(cf[owner])):
        problems.append(f'{owner} differs from the reference outside its per-step loop body')
    top = lambda t: [ast.dump(n) for n in t.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if top(ref_ast) != top(cand_ast):
        problems.append('module-level statements (imports, accelerator, __main__ block) differ from the reference')
    return problems


def loader_probe(tree, switches, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(tree)
    try:
        cov = CFG['coverage']
        env = run.env(switches, run.tree)
        env['CUDA_VISIBLE_DEVICES'] = ''
        cmd = [PY, str(BENCH / 'loader_probe.py'), '--out', str(out / 'loader_probe.json'), '--batches',
               str(cov['probe_batches']), '--samples', str(cov['probe_samples']), '--', *train_argv(run.tmp / 'out')]
        rc = execute(cmd, run.tree, env, out / 'loader_probe.log', TIMEOUT['probe'])
        if rc != 0 or not (out / 'loader_probe.json').exists():
            return None, f'loader probe failed (returncode {rc}, see {out}/loader_probe.log)'
        return json.loads((out / 'loader_probe.json').read_text()), None
    finally:
        run.close()


def gate0(tree, switches, out):
    problems = structure(tree)
    if problems:
        return {'status': 'ERROR', 'problems': problems}
    probe, err = loader_probe(tree, switches, out)
    if err:
        return {'status': 'ERROR', 'problems': [err]}
    ref = json.loads(rel(CFG['coverage']['reference_probe']).read_text())
    diff = [k for k in ('len_dataset', 'len_loader', 'batch_size', 'samples', 'batches') if probe[k] != ref[k]]
    if diff:
        return {'status': 'INCORRECT', 'problems': [f'data pipeline differs from the reference in: {diff}']}
    return {'status': 'PASS', 'samples': len(ref['samples']), 'batches': len(ref['batches'])}


# ---------------------------------------------------------------------------------------------- gates 2 and 3

def numerics_record(tree, switches, out, cases, losses_only=False, variant=None, timeout=None):
    """accelerate launch -m numerics record ... on a fresh copy of the tree, deterministic mode."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(tree)
    try:
        env = run.env(switches, ROOT)
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        cases_file = run.tmp / 'cases.json'
        cases_file.write_text(json.dumps(cases))
        kwargs = {'repo': str(run.tree)}
        if variant:
            kwargs['variant'] = variant
        cmd = [ACCELERATE, 'launch', '--config_file', str(run.tree / CFG['accelerate_config']), '--num_processes',
               str(CFG['world_size']), '--main_process_port', port(), '--module', 'numerics', 'record', '--adapter',
               CFG['numerics']['adapter'], '--adapter-kwargs', json.dumps(kwargs), '--cases', str(cases_file),
               '--out', str(out), '--deterministic']
        cmd += ['--losses-only'] if losses_only else ['--sketch-dim', str(CFG['numerics']['sketch_dim'])]
        name = cases[0]['name']
        t0 = time.time()
        rc = execute(cmd, ROOT, env, out / f'{name}.log', timeout or TIMEOUT['record'])
        ok = rc == 0 and all((out / f'{c["name"]}.r{r}.json').exists()
                             for c in cases for r in range(CFG['world_size']))
        return {'ok': ok, 'returncode': rc, 'seconds': time.time() - t0, 'log': str(out / f'{name}.log'),
                'tree_digest': run.digest}
    finally:
        run.close()


def record_cases(tree, switches, out, variant=None, only=None):
    cases = json.loads(rel(CFG['numerics']['cases']).read_text())
    results = []
    for case in cases:
        if only and case['name'] not in only:
            continue
        results.append(numerics_record(tree, switches, out, [case], variant=variant))
        log(f'record {case["name"]}: ok={results[-1]["ok"]} {results[-1]["seconds"]:.0f}s')
        if not results[-1]['ok']:
            break
    return results


def compare_records(ref, cand, out):
    thresholds = Path(out) / 'thresholds.empty.json'
    thresholds.write_text(json.dumps({'meta': {'k_fwd': 2.0, 'k_bwd': 3.0, 'floor_fwd': 1e-6, 'floor_bwd': 1e-5},
                                      'gates': {}}))
    env = base_env()
    env['PYTHONPATH'] = str(ROOT)
    subprocess.run([PY, '-m', 'numerics', 'compare', '--ref', str(ref), '--cand', str(cand), '--thresholds',
                    str(thresholds), '--out', str(Path(out) / 'compare.json')], cwd=ROOT, env=env,
                   capture_output=True)
    rep = json.loads((Path(out) / 'compare.json').read_text())
    hard = [e for e in rep['errors'] if not (isinstance(e, dict) and e.get('error') == 'no calibrated gate')]
    res = {'checked': rep['checked'], 'bitwise_identical': rep['bitwise_identical'],
           'exact_failures': rep['exact_failures'][:20], 'errors': hard[:20],
           'precision_mode_changes': rep['precision_mode_changes'][:5], 'dtype_changes': rep['dtype_changes'][:5]}
    if hard:
        res['status'] = 'ERROR'
    elif rep['exact_failures']:
        res['status'] = 'INCORRECT'
    elif rep['checked'] and rep['checked'] == rep['bitwise_identical']:
        res['status'] = 'BITWISE'
    else:
        res['status'] = 'NOT_BITWISE'
    return res


def traj_cases(seeds):
    spec = json.loads(rel(CFG['trajectory']['cases']).read_text())
    return [dict(spec, name=f'traj_s{s}', seed=s) for s in seeds]


def record_traj(tree, switches, out, seeds, variant=None):
    results = []
    for case in traj_cases(seeds):
        results.append(numerics_record(tree, switches, out, [case], losses_only=True, variant=variant,
                                       timeout=TIMEOUT['trajectory']))
        log(f'trajectory {case["name"]}: ok={results[-1]["ok"]} {results[-1]["seconds"]:.0f}s')
        if not results[-1]['ok']:
            break
    return results


def traj_bound(cand, out):
    env = base_env()
    env['PYTHONPATH'] = str(ROOT)
    report = Path(out) / 'traj_compare.json'
    subprocess.run([PY, '-m', 'numerics', 'traj-compare', '--frozen', str(rel(CFG['trajectory']['frozen'])), '--cand',
                    str(cand), '--sigma-k', str(CFG['trajectory']['sigma_k']), '--out', str(report)], cwd=ROOT, env=env,
                   capture_output=True)
    return json.loads(report.read_text()) if report.exists() else {'verdict': 'error'}


# ---------------------------------------------------------------------------------------------- check

def cmd_check(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    switches = load_switches(a.config)
    final = a.final
    if final and not (ROOT / 'task' / 'final-evaluation').exists():
        raise SystemExit('check --final is run by the harness only')
    res = {'tree': str(Path(a.tree).resolve()), 'switches': switches, 'evaluator_sha256': evaluator_sha256(),
           'tree_digest': tree_digest(a.tree), 'mode': 'final' if final else 'n1' if a.n1 else 'quick' if a.quick
           else 'skip-timing' if a.skip_timing else 'check', 'gates': {}, 'started': datetime.now().isoformat()}

    def done(status, **extra):
        res.update(status=status, finished=datetime.now().isoformat(), **extra)
        if evaluator_sha256() != res['evaluator_sha256']:
            res['status'] = 'ERROR'
            res['error'] = 'the evaluator changed during the check'
        (out / 'check.json').write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps({'status': res['status'], 'tier': res.get('tier'),
                          'performance': res.get('performance'),
                          'gates': {k: v.get('status') for k, v in res['gates'].items()}}))
        return 0 if res['status'] == 'PASS' else 2

    g0 = gate0(a.tree, switches, out / 'gate0')
    res['gates']['structure'] = g0
    log(f'gate 0: {g0["status"]}')
    if g0['status'] != 'PASS':
        return done(g0['status'])

    with GpuLock():
        if not (a.quick or a.skip_timing or a.n1):
            fr = frozen_timing()
            g1 = {}
            if final:
                r0 = timing_run(rel(CFG['reference']), {}, out / 'gate1', 'ref_a')
                c = timing_run(a.tree, switches, out / 'gate1', 'cand')
                r1 = timing_run(rel(CFG['reference']), {}, out / 'gate1', 'ref_b')
                runs = [r0, c, r1]
                if any(r['status'] == 'ERROR' for r in runs):
                    res['gates']['benefit'] = {'status': 'ERROR', 'errors': [r.get('error') for r in runs]}
                    return done('ERROR')
                if not all(fingerprint_equal(r0['fingerprint'], r['fingerprint']) for r in (c, r1)):
                    res['gates']['benefit'] = {'status': 'ERROR', 'error': 'run fingerprints differ'}
                    return done('ERROR')
                ref_ms = statistics.mean([r0['step_ms'], r1['step_ms']])
                spread = abs(r0['step_ms'] - r1['step_ms']) / ref_ms
                invalid = [r['tag'] for r in runs if r['status'] == 'INVALID' or r['window_ratio'] > 1 + T['window_band']]
                invalid += [r['tag'] + ':frozen_band' for r in (r0, r1)
                            if abs(r['step_ms'] / fr['step_ms'] - 1) > T['frozen_band']]
                if spread > T['pair_band']:
                    invalid.append('pair_band')
                g1.update(reference_step_ms=[r0['step_ms'], r1['step_ms']], invalid=invalid)
            else:
                c = timing_run(a.tree, switches, out / 'gate1', 'cand')
                if c['status'] == 'ERROR':
                    res['gates']['benefit'] = {'status': 'ERROR', 'error': c.get('error')}
                    return done('ERROR')
                ref_ms, spread = fr['step_ms'], fr['spread']
                invalid = ['cand'] if c['status'] == 'INVALID' else []
                g1['notes'] = {'window_ratio': c['window_ratio'], 'frozen_window_ratio': fr['window_ratio']}
            speedup = ref_ms / c['step_ms']
            threshold = max(1.02, 1 + 2 * spread)
            g1.update(candidate_step_ms=c['step_ms'], candidate_windows=c['window_step_ms'], reference_ms=ref_ms,
                      speedup=speedup, threshold=threshold, spread=spread, peak_mem_gib=c['peak_mem_gib'],
                      gpu_util=c['gpu_util_mean_second_half'])
            res['performance'] = {'speedup': speedup, 'reference_step_ms': ref_ms, 'candidate_step_ms': c['step_ms']}
            if invalid:
                g1['status'] = 'INVALID'
                res['gates']['benefit'] = g1
                return done('INVALID')
            g1['status'] = 'PASS' if speedup > threshold else 'UNAVAILABLE'
            res['gates']['benefit'] = g1
            log(f'gate 1: {g1["status"]} speedup {speedup:.4f} (threshold {threshold:.4f})')
            if g1['status'] != 'PASS':
                return done('UNAVAILABLE', reason='no_benefit')

        bitwise = False
        if not a.n1:
            only = ['step2'] if a.quick else None
            recs = record_cases(a.tree, switches, out / 'records', only=only)
            if not all(r['ok'] for r in recs):
                res['gates']['bitwise'] = {'status': 'ERROR', 'records': recs}
                return done('ERROR')
            ref_records = rel(CFG['numerics']['ref'])
            if only:                                   # compare only the recorded case
                sub = out / 'ref_subset'
                sub.mkdir(exist_ok=True)
                for p in ref_records.glob('step2.*'):
                    if not (sub / p.name).exists():
                        os.symlink(p, sub / p.name)
                ref_records = sub
            g2 = compare_records(ref_records, out / 'records', out)
            res['gates']['bitwise'] = g2
            log(f'gate 2: {g2["status"]} ({g2["bitwise_identical"]}/{g2["checked"]} tensors bitwise)')
            if g2['status'] in ('ERROR', 'INCORRECT'):
                return done(g2['status'])
            bitwise = g2['status'] == 'BITWISE'
            if not a.keep_records:
                for p in (out / 'records').glob('*.pt'):
                    p.unlink()
            if a.quick:
                return done('PASS' if bitwise else 'NOT_BITWISE', tier='N0' if bitwise else None,
                            note='quick: gate 0 and the step2 case only')
        if bitwise:
            return done('PASS', tier='N0')

        seeds = CFG['trajectory']['reference_seeds'][:CFG['trajectory']['seeds']]
        trs = record_traj(a.tree, switches, out / 'traj', seeds)
        if not all(r['ok'] for r in trs):
            res['gates']['statistical'] = {'status': 'ERROR', 'records': trs}
            return done('ERROR')
        rep = traj_bound(out / 'traj', out)
        ok = rep.get('verdict') == 'equivalent'
        res['gates']['statistical'] = {'status': 'PASS' if ok else 'INCORRECT', 'verdict': rep.get('verdict'),
                                       'endpoints': rep.get('endpoints')}
        log(f'gate 3: {rep.get("verdict")}')
        if not ok:
            return done('INCORRECT')
        g2 = res['gates'].get('bitwise', {})
        tier = 'N2' if (g2.get('precision_mode_changes') or g2.get('dtype_changes')) else 'N1'
        return done('PASS', tier=tier)


# ---------------------------------------------------------------------------------------------- formulation

def cmd_formulate(a):
    """Formulator only: reference records, trajectories and probes (never run by the optimizing agent)."""
    ref = rel(CFG['reference'])
    out = Path(a.out)
    if a.what == 'loader':
        probe, err = loader_probe(ref, {}, out)
        if err:
            raise SystemExit(err)
        print(json.dumps({'samples': len(probe['samples']), 'batches': len(probe['batches'])}))
        return 0
    with GpuLock():
        if a.what == 'records':
            recs = record_cases(a.tree or ref, load_switches(a.config), out, variant=a.variant, only=a.only)
            print(json.dumps(recs))
            return 0 if all(r['ok'] for r in recs) else 1
        if a.what == 'traj':
            seeds = [int(s) for s in a.seeds.split(',')]
            trs = record_traj(a.tree or ref, load_switches(a.config), out, seeds, variant=a.variant)
            print(json.dumps(trs))
            return 0 if all(r['ok'] for r in trs) else 1
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--tree', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--config')
    c.add_argument('--quick', action='store_true')
    c.add_argument('--skip-timing', action='store_true')
    c.add_argument('--n1', action='store_true')
    c.add_argument('--final', action='store_true', help=argparse.SUPPRESS)
    c.add_argument('--keep-records', action='store_true')
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    pr = sub.add_parser('profile')
    pr.add_argument('--tree', required=True)
    pr.add_argument('--out', required=True)
    pr.add_argument('--config')
    pr.add_argument('--steps', type=int, default=20)
    pr.add_argument('--blocking', action='store_true')
    pr.add_argument('--stages', action='store_true')
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    b.add_argument('--freeze', action='store_true')
    f = sub.add_parser('formulate')
    f.add_argument('what', choices=('loader', 'records', 'traj'))
    f.add_argument('--out', required=True)
    f.add_argument('--tree')
    f.add_argument('--config')
    f.add_argument('--variant')
    f.add_argument('--only', nargs='*')
    f.add_argument('--seeds', default='0')
    a = p.parse_args()
    for name in ('tree', 'out', 'config'):
        if getattr(a, name, None):
            setattr(a, name, str(Path(getattr(a, name)).resolve()))
    return {'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile, 'baseline': cmd_baseline,
            'formulate': cmd_formulate}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())

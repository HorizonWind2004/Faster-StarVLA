"""starVLA (QwenOFT, LIBERO 4-in-1) adapter for the numerics/ precision judge.

The training step is NOT rewritten here. The first step() runs the tree's own entry point
(starVLA/training/train_starvla.py, through runpy as __main__, exactly like the timing run) in a worker thread; the
thread is paused every time the tree steps its learning-rate scheduler (once per optimizer step in the reference; the
learning rates are exact items, so a valid candidate does the same) and resumed by the next step(). The candidate's
loop, data pipeline, model, Accelerate/DeepSpeed setup are therefore exactly what gets recorded.

What the generic recorder (numerics.record) observes:
- loss/loss: this rank's loss of the step, the tensor the tree hands to DeepSpeedEngine.backward;
  in trajectories also loss/probe and loss/upd (see below);
- grad/, delta/, optim/: a canonical view of the trainable parameters that does not depend on how DeepSpeed
  partitions them. Around every real DeepSpeed optimizer step, each trainable parameter's full fp32 gradient, fp32
  master weight and Adam states are gathered with deepspeed.utils.safe_get_full_*; this rank keeps the r-th of
  world_size equal contiguous chunks of the flattened tensor (zero padded). The view is a torch optimizer whose
  step() runs the real DeepSpeed step. Names and shapes follow the reference's trainable parameters
  (task/trainable.json);
- exact invariants: optimizer steps and backward calls in the step, scheduler step count and learning rates, Adam
  hyper-parameters, DeepSpeed batch / accumulation / clipping / precision / ZeRO stage, the set of trainable
  parameters, dtypes of master weights / gradients / Adam states, and a digest of what the step fed the VLM
  (per row: input_ids where attention_mask is set, pixel_values cast to bf16, image_grid_thw), gathered over ranks;
- RNG fingerprints before and after every step (the recorder takes them).

Deviations from a real run, identical for reference and candidate: every rank is seeded with the case seed before
the entry point runs (upstream builds the model before it seeds, so its action head differs from run to run and from
rank to rank); cases and trajectories override the schedule (bench/cases.json, bench.json trajectory.args).
Trajectories add loss/probe (the tree's own forward, no gradient, on a held-out batch built from the tree's dataset
at fixed indices far from the training stream), loss/upd and loss/upd_vlm (the mean |change| the step makes to the
fp32 master weights of the action head and of the VLM parameters in bench.json trajectory.vlm_params: proportional
to the learning rate under Adam, zero without a gradient), and loss/updq, loss/upd_vlmq (the same over only the elements whose
|change| is above 90% of the largest: in the first updates Adam moves those by the learning rate itself, so they
resolve a 2% change of the learning rate on one seed).
"""
import hashlib
import json
import math
import os
import runpy
import sys
import tempfile
import threading
from pathlib import Path

import torch

from numerics.adapter import Adapter, Objects

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
CFG = json.loads((BENCH / 'bench.json').read_text())
ENTRY = CFG['entry']


class _Stop(BaseException):
    """Raised in the paused training thread after the last recorded step."""


def expand(value):
    out = os.path.expandvars(value)
    if '$' in out:
        raise RuntimeError(f'{value}: set ASSETS before running the bench')
    return out


def train_argv(overrides, out_dir):
    """The bench's fixed train_starvla.py arguments with {placeholders} filled and `overrides` ({flag: value}) applied."""
    paths = {k: expand(v) for k, v in CFG['assets'].items()}
    paths['out'] = str(out_dir)
    argv = [a.format(**paths) for a in CFG['train_args']]
    pairs = [[argv[i], argv[i + 1]] for i in range(0, len(argv), 2)]
    for flag, value in (overrides or {}).items():
        value = str(value).format(**paths)
        for p in pairs:
            if p[0] == flag:
                p[1] = value
                break
        else:
            pairs.append([flag, value])
    return [x for p in pairs for x in p]


class _View(torch.optim.Optimizer):
    """The recorder observes this optimizer. Its step(closure) runs the real DeepSpeed optimizer step."""

    def __init__(self, params):
        super().__init__(params, {'lr': 0.0})

    def step(self, closure=None):
        return closure()


def _view_module(trainable, world):
    root = torch.nn.Module()
    params = []
    for name, shape in trainable:
        *path, leaf = name.split('.')
        m = root
        for part in path:
            if part not in m._modules:
                m.add_module(part, torch.nn.Module())
            m = m._modules[part]
        numel = math.prod(shape)
        p = torch.nn.Parameter(torch.zeros(-(-numel // world), dtype=torch.float32), requires_grad=True)
        m.register_parameter(leaf, p)
        params.append((name, p))
    return root, params


class StarVLAAdapter(Adapter):
    name = 'starvla-qwenoft-libero'

    def __init__(self, repo, variant=None):
        self.repo = Path(repo).resolve()
        if variant not in (None, 'lr102'):
            raise ValueError(f'unknown variant {variant!r}')
        self.variant = variant
        self.case = {}
        self._thread = None
        self._error = None
        self._stop = False
        self._arrived = threading.Semaphore(0)
        self._go = threading.Semaphore(0)
        self._record_view = False
        self._engine = None
        self._sched = None
        self._losses, self._inputs, self._upd, self._probe = [], [], None, None
        self._opt_steps = 0
        self._inv, self._dtypes = {}, {}
        self._probe_batch = None
        self._probe_dataset = None

    # ------------------------------------------------------------------ build

    def configure(self, case):
        self.case = dict(case)

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError('no fp64 golden for this task (4B model); arbitration is diagnostic only')
        import torch.distributed as dist
        self.rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.local = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(self.local)
        cache = os.path.join(os.environ.get('TRITON_CACHE_DIR') or os.path.join(os.environ.get('TMPDIR', '/tmp'),
                                                                               'triton'), f'rank{self.local}')
        os.makedirs(cache, exist_ok=True)
        os.environ['TRITON_CACHE_DIR'] = cache
        overrides = dict(self.case.get('args') or {})
        overrides.setdefault('--seed', str(seed))
        self._out = tempfile.mkdtemp(prefix='starvla-out-')
        self._argv = train_argv(overrides, self._out)
        self.trainable = [(n, tuple(s)) for n, s in json.loads((ROOT / CFG['numerics']['trainable']).read_text())] \
            if (ROOT / CFG['numerics']['trainable']).exists() else None
        # The recorder asks for rank and world size right after build(); the entry point would only initialize
        # torch.distributed later (inside Accelerator). Initialize it here exactly as Accelerate does for DeepSpeed.
        from deepspeed import comm
        if not comm.is_initialized():
            comm.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
        self._install_hooks()
        del dist

    def _install_hooks(self):
        import deepspeed.runtime.engine as ds_engine
        import torch.optim.lr_scheduler as lrs
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        adapter = self

        real_backward = ds_engine.DeepSpeedEngine.backward

        def backward(engine, loss, *args, **kwargs):
            if adapter._engine is None:
                adapter._on_engine(engine)
            adapter._losses.append(loss.detach().float().clone())
            return real_backward(engine, loss, *args, **kwargs)
        ds_engine.DeepSpeedEngine.backward = backward

        real_zstep = DeepSpeedZeroOptimizer.step

        def zstep(zopt, *args, **kwargs):
            adapter._opt_steps += 1
            want_upd = adapter.case.get('upd') and not adapter._record_view
            if want_upd:
                before = adapter._masters()
            if not adapter._record_view:
                out = real_zstep(zopt, *args, **kwargs)
                if want_upd:
                    after = adapter._masters()
                    upd = {}
                    for g in after:
                        diffs = [(a - b).abs().reshape(-1) for a, b in zip(after[g], before[g])]
                        upd[g] = torch.stack([d.mean() for d in diffs]).mean()
                        # the elements Adam moved by nearly a full step: mean |change| over those above 90% of the
                        # largest. In the first updates this is the learning rate itself, whatever the seed.
                        flat = torch.cat(diffs)
                        top = flat.max()
                        upd[g + 'q'] = flat[flat >= 0.9 * top].mean() if float(top) > 0 else top
                    adapter._upd = upd
                return out
            adapter._fill('pre')
            if adapter._sched is not None:
                adapter.view_opt.param_groups[0]['lr'] = float(adapter._sched.get_last_lr()[0])
            res = []

            def closure():
                res.append(real_zstep(zopt, *args, **kwargs))
                # DeepSpeed links the Adam states to the parameters once, after the FIRST param group's first step,
                # when the other groups have no state yet; link every group here so the states can be read (this
                # only sets views used by safe_get_full_optimizer_state, nothing the training reads).
                from deepspeed.utils import lazy_init_hp_params_optimizer_state
                for i, _ in enumerate(zopt.optimizer.param_groups):
                    lazy_init_hp_params_optimizer_state(zopt.bit16_groups[i], zopt.single_partition_of_fp32_groups[i],
                                                        zopt.optimizer.state)
                adapter._fill('post')
                return res[0]
            adapter.view_opt.step(closure)
            return res[0]
        DeepSpeedZeroOptimizer.step = zstep

        from torch.utils.data import DataLoader
        real_loader_init = DataLoader.__init__

        def loader_init(loader, *args, **kwargs):
            real_loader_init(loader, *args, **kwargs)
            if adapter._probe_dataset is None and hasattr(loader.dataset, 'sample_step'):
                adapter._probe_dataset = loader.dataset
        DataLoader.__init__ = loader_init

        real_sched_step = lrs.LRScheduler.step

        def sched_step(sched, *args, **kwargs):
            out = real_sched_step(sched, *args, **kwargs)
            if sched._step_count - 1 > 0 and threading.current_thread() is adapter._thread:
                adapter._sched = sched
                adapter._at_step_end()
            return out
        lrs.LRScheduler.step = sched_step

    def _on_engine(self, engine):
        """First backward: the DeepSpeed engine exists. Hook the VLM's inputs and list the trainable parameters."""
        self._engine = engine
        names = {n: list(p.shape) for n, p in engine.module.named_parameters() if p.requires_grad}
        self._trainable_sha = hashlib.sha256(json.dumps(sorted(names.items())).encode()).hexdigest()[:32]
        if self.trainable is None:                      # formulation only: the reference's trainable parameters
            if self.rank == 0:
                (ROOT / CFG['numerics']['trainable']).write_text(json.dumps(list(names.items())))
            raise SystemExit('task/trainable.json was missing; written from this tree (formulation step)')

    def _hook_vlm_inputs(self):
        """Digest what the step feeds the VLM. Registered on the innermost module that receives input_ids and
        pixel_values (Qwen3VLModel), so a tree may or may not go through the language-model head."""
        from transformers import Qwen3VLModel
        adapter = self
        real = Qwen3VLModel.forward

        def forward(model, *args, **kwargs):
            if threading.current_thread() is adapter._thread and torch.is_grad_enabled():
                adapter._inputs.append(adapter._digest(kwargs))
            return real(model, *args, **kwargs)
        Qwen3VLModel.forward = forward

    @staticmethod
    def _digest(kwargs):
        ids, mask = kwargs.get('input_ids'), kwargs.get('attention_mask')
        pix, grid = kwargs.get('pixel_values'), kwargs.get('image_grid_thw')
        if ids is None or pix is None:
            return {'rows': ['<no input_ids/pixel_values keyword>']}
        ids_c = ids.detach().cpu()
        mask_c = mask.detach().cpu().bool() if mask is not None else torch.ones_like(ids_c, dtype=torch.bool)
        rows = [hashlib.sha256(ids_c[b][mask_c[b]].numpy().tobytes()).hexdigest()[:16] for b in range(ids_c.shape[0])]
        h = hashlib.sha256(pix.detach().to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().tobytes())
        if grid is not None:
            h.update(grid.detach().cpu().numpy().tobytes())
        return {'rows': rows, 'pixels': h.hexdigest()[:32], 'tokens': int(mask_c.sum())}

    # ------------------------------------------------------------------ hooks in the training thread

    def _shard(self, t):
        flat = t.detach().reshape(-1)
        n = flat.numel()
        per = -(-n // self.world)
        a, b = self.rank * per, min(n, (self.rank + 1) * per)
        out = torch.zeros(per, dtype=flat.dtype)
        if a < b:
            out[:b - a].copy_(flat[a:b])
        return out

    def _fill(self, when):
        from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
        params = dict(self._engine.module.named_parameters())
        for name, vp in self.view_params:
            p = params.get(name)
            if p is None or not p.requires_grad:
                continue
            w = safe_get_full_fp32_param(p)
            if w is None:
                raise RuntimeError(f'no fp32 master weight for {name}')
            self._dtypes['master'] = str(w.dtype)
            if when == 'pre':
                g = safe_get_full_grad(p)
                if g is None:
                    raise RuntimeError(f'no gradient for {name} before the optimizer step')
                self._dtypes['grad'] = str(g.dtype)
                vp.grad = self._shard(g)
                vp.data = self._shard(w)
            else:
                vp.data = self._shard(w)
                state = {}
                for key in ('exp_avg', 'exp_avg_sq'):
                    s = safe_get_full_optimizer_state(p, key)
                    if s is None:
                        raise RuntimeError(f'no optimizer state {key} for {name}')
                    self._dtypes[key] = str(s.dtype)
                    state[key] = self._shard(s)
                self.view_opt.state[vp] = state
            del w

    def _masters(self):
        """fp32 master weights behind the trajectory endpoints loss/upd (the action head) and loss/upd_vlm (the VLM
        parameters named in bench.json trajectory.vlm_params): the mean |change| a step makes to them is proportional
        to the learning rate under Adam, and zero for a parameter that receives no gradient."""
        from deepspeed.utils import safe_get_full_fp32_param
        vlm = set(CFG['trajectory']['vlm_params'])
        out = {'upd': [], 'upd_vlm': []}
        for n, p in self._engine.module.named_parameters():
            if not p.requires_grad:
                continue
            group = 'upd' if n.startswith('action_model.') else 'upd_vlm' if n in vlm else None
            if group:
                out[group].append(safe_get_full_fp32_param(p).detach().float().clone())
        if not out['upd'] or len(out['upd_vlm']) != len(vlm):
            raise RuntimeError('trajectory endpoint parameters not found in this tree')
        return out

    def _make_probe(self):
        """A held-out batch from the tree's own dataset: fixed indices far beyond the steps a trajectory trains on.
        The dataset is the one the tree's first DataLoader was built on (recorded when it is constructed; a scan of
        gc.get_objects() cannot see objects a tree has moved to the permanent generation with gc.freeze())."""
        ds = self._probe_dataset
        if ds is None:
            raise RuntimeError('no starVLA mixture dataset found for the probe batch')
        base, per = CFG['trajectory']['probe_index'], CFG['trajectory']['probe_rows']
        return [ds[base + self.rank * per + i] for i in range(per)]

    def _probe_loss(self):
        if self._probe_batch is None:
            self._probe_batch = self._make_probe()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = self._engine.module(self._probe_batch)
        return out['action_loss'].detach().float().clone()

    def _at_step_end(self):
        steps = self.case.get('probe_steps')
        g = self._sched._step_count - 1
        self._probe = None
        if steps and steps[0] <= g - 1 < steps[1]:
            self._probe = self._probe_loss()
        self._arrived.release()
        self._go.acquire()
        if self._stop:
            raise _Stop()

    def _run(self):
        torch.cuda.set_device(self.local)
        entry = str(self.repo / ENTRY)
        os.chdir(self.repo)
        sys.path.insert(0, str(self.repo))
        saved = sys.argv
        sys.argv = [entry, *self._argv]
        try:
            self._hook_vlm_inputs()
            if self.variant == 'lr102':            # control arm: every learning rate x 1.02
                import starVLA.training.trainer_utils.trainer_tools as tt
                real = tt.build_param_lr_groups

                def scaled(model, cfg):
                    groups = real(model=model, cfg=cfg)
                    for group in groups:
                        group['lr'] = group['lr'] * 1.02
                    return groups
                tt.build_param_lr_groups = scaled
            runpy.run_path(entry, run_name='__main__')
            self._error = RuntimeError('the entry point finished before the case did')
        except _Stop:
            pass
        except BaseException as error:              # report in the main thread
            self._error = error
        finally:
            sys.argv = saved
            self._arrived.release()

    # ------------------------------------------------------------------ Adapter API

    def objects(self):
        self._record_view = True
        self.view_module, self.view_params = _view_module(self.trainable or [('placeholder', (1,))], self.world)
        self.view_opt = _View([p for _, p in self.view_params])
        return Objects(models={'vla': self.view_module}, optimizers={'adamw': self.view_opt},
                       sync={'vla': 'local'}, count_forwards=False)

    def batch(self, step, rank):
        """The input is the tree's own data loader on the full LIBERO data (deterministic per index); what reached
        the VLM is an exact invariant of the step."""
        return None

    def step(self, step, batch):
        self._losses, self._inputs, self._upd, self._opt_steps = [], [], None, 0
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='trainer', daemon=True)
            self._thread.start()
        else:
            self._go.release()
        self._arrived.acquire()
        if self._error is not None:
            raise RuntimeError(f'training thread failed: {type(self._error).__name__}: {self._error}') from self._error
        import torch.distributed as dist
        engine, sched = self._engine, self._sched
        mine = {'rows': [r for d in self._inputs for r in d['rows']],
                'pixels': [d.get('pixels') for d in self._inputs],
                'tokens': sum(d.get('tokens', 0) for d in self._inputs)}
        parts = [mine]
        if dist.is_initialized() and self.world > 1:
            parts = [None] * self.world
            dist.all_gather_object(parts, mine)
        rows = sorted(r for p in parts for r in p['rows'])
        inner = getattr(engine.optimizer, 'optimizer', engine.optimizer)
        inv = {'optimizer_steps': self._opt_steps, 'backward_calls': len(self._losses),
               'scheduler_steps': sched._step_count - 1,
               'lr_after_step': [float(x) for x in sched.get_last_lr()],
               'trainable_params': self._trainable_sha,
               'vlm/rows': hashlib.sha256('\n'.join(rows).encode()).hexdigest()[:32], 'vlm/row_count': len(rows),
               'vlm/pixels': sorted(x or '' for p in parts for x in p['pixels']),
               'vlm/tokens': sum(p['tokens'] for p in parts),
               'adam': sorted({json.dumps([list(g.get('betas', ())), g.get('eps'), g.get('weight_decay')])
                               for g in inner.param_groups}),
               'deepspeed': {'micro_batch': engine.train_micro_batch_size_per_gpu(),
                             'grad_accum': engine.gradient_accumulation_steps(),
                             'train_batch': engine.train_batch_size(), 'clipping': engine.gradient_clipping(),
                             'bf16': bool(engine.bfloat16_enabled()), 'fp16': bool(engine.fp16_enabled()),
                             'zero_stage': int(engine.zero_optimization_stage())}}
        if self._record_view:
            inv['dtypes'] = dict(sorted(self._dtypes.items()))
        self._inv = inv
        out = {'loss': torch.stack(self._losses).sum() if self._losses else torch.tensor(float('nan'))}
        if self._probe is not None:
            out['probe'] = self._probe
        if self._upd is not None:
            out.update(self._upd)
        if step == int(self.case.get('steps', 1)) - 1:
            self._stop = True
            self._go.release()
            self._thread.join(timeout=300)
        return out

    def invariants(self):
        return dict(self._inv)

# GOAL: faster starVLA training (QwenOFT on LIBERO), same training

## The goal

Make one training step of starVLA's LIBERO recipe faster on the same hardware without changing what is trained.

- Upstream: https://github.com/starVLA/starVLA at `99764f5` (`targets/reference`, never changed).
- Candidate: `starVLA/` (a clone of the same commit on branch `opt`); all changes go there, one commit per change.
- Workload: `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` with the QwenOFT framework on
  Qwen3-VL-4B-Instruct, all four LIBERO suites (`libero_all`, 1,693 episodes), per-GPU batch 16, DeepSpeed ZeRO-2
  through Accelerate, bf16 autocast, every module trainable (4.50 B parameters). The exact arguments are
  `bench/bench.json` `train_args`.
- Hardware: 2 x NVIDIA B300, 32 CPU cores (16 per GPU), one node. The upstream script uses every GPU of its node with
  the same per-GPU batch; the per-GPU load here is upstream's.

## The single criterion

`python bench/bench.py check --tree starVLA --config <switch file> --out <dir>` writes `check.json` with
`status == "PASS"` and `performance.speedup >= 2.0` (target). If a profile shows the target cannot be reached within
the rules, `>= 1.3` (minimum) is acceptable for wrapping up, with the reasons in `WORK_LOG.md`.

The agent's `check` runs every gate of the final check with the same rules; only gate 1's timing differs (one
candidate run against the frozen reference timing instead of reference, candidate, reference). `check --final` is the
harness's.

## How a candidate is judged

Each gate stops the evaluation when it fails (details: the docstring of `bench/bench.py`, `bench/starvla_adapter.py`).

0. **Structure.** In `starVLA/training/train_starvla.py`, the module level, `main`, the setup functions listed in
   `bench.json` `coverage.entry_functions`, and `VLATrainer.train` outside its per-step `while` loop body must equal
   the reference AST. The data pipeline must yield bitwise the reference's samples: `dataset[i]` and the first
   batches of `build_dataloader(cfg)` are lists of dicts whose `action`, `image` and `lang` are compared (a tree may
   add fields). Failure: ERROR / INCORRECT.
1. **Benefit.** A fresh process of the real entry point, cold compile caches, 20 warmup steps, then 2 windows of 100
   steps (each window contains one of the recipe's `eval_interval` evaluations). speedup = frozen reference step time
   / candidate step time; it must exceed `max(1.02, 1 + 2 x reference spread)`. A compile inside a window is INVALID.
2. **Bitwise.** Two 2-step cases in deterministic mode (`step2`; `eval`, with an evaluation between the steps), both
   ranks. Exact items: RNG fingerprints before and after each step, optimizer steps and backward calls per step,
   scheduler step count and learning rates, Adam hyper-parameters, DeepSpeed settings, the trainable parameters, the
   dtypes of master weights / gradients / Adam states, and a digest of what the step fed the VLM (token ids per row,
   pixel values, grid). Then every tensor (loss; fp32 gradient, update and Adam state of all 729 parameters) must
   match bit for bit: PASS (N0).
3. **Statistical**, only when gate 2 is not bitwise: one 60-step trajectory in deterministic mode on the first frozen
   seed; five endpoints (training loss over two windows, a held-out probe loss, the action head's update size over two
   windows), each within 3 sigma of the reference's own seed-to-seed spread (6 seeds, frozen): PASS (N1; N2 when a
   math mode or dtype changed). The control arm (every learning rate x 1.02) is rejected by this gate.

Commands and what they cost on the reference (2 GPUs):

| Command | What | Time |
|---|---|---|
| `bench.py profile --tree T --out D [--blocking]` | cProfile of rank 0 over 20 steps after the warmup, GPU utilization | ~4 min |
| `bench.py time --tree T --out D [--config F]` | one timing run, speedup against the frozen reference | ~7 min at reference speed |
| `bench.py check --quick ...` | gate 0 and the `step2` case | ~15 min |
| `bench.py check ...` | all gates | ~40 min |
| `bench.py check --n1 ...` | gates 0 and 3 only | ~8 min |

All of them take the GPU lock themselves; never wrap them in `flock`. Any other GPU command must hold
`flock $WORK/starvla/gpu.lock`. Inside the allocation: `../b.sh <bench.py arguments>`.

## Rules

- **Training semantics do not change**: the samples and their order, per-GPU batch, loss, optimizer and its
  hyper-parameters, learning-rate schedule, gradient clipping, one optimizer step per step, DeepSpeed ZeRO-2 with fp32
  master weights and Adam states, the evaluation every `eval_interval` steps (it consumes one batch of the stream),
  logging and checkpoint contents.
- **Precision floor**: bf16 autocast as upstream. No FP8/INT8, no removal of the fp32 master weights. Local precision
  or math-mode changes are at most N2 and must pass gates 2 and 3.
- **Same code path**: the code must not read the bench's environment or behave differently when timed, recorded or
  judged. Switches are environment variables in the switch file (`{"environment": {...}}`), off by default: with no
  switch set the tree runs the upstream code path.
- **Timing only through `bench.py`.** No private timing scripts as evidence.
- `bench/`, `numerics/`, `task/`, `targets/` are read-only. No new packages; the environment is `$ASSETS/env.sh`.
- Shared machine: only the two GPUs of the allocation, only this workspace.

## Per round (at most 30 minutes, counted from taking the GPU lock)

1. Profile the current best version before changing anything (`profile`, with and without `--blocking`; the PyTorch
   profiler and nsys cannot see kernels on this GPU generation, so account by stage: data wait, input processing, VLM
   forward, backward, optimizer step, host synchronization).
2. Change one hotspot behind its own switch.
3. Screen with `time` (and `check --quick` when numerics may be touched); run the full `check` once the accumulated
   changes reach the target or before wrapping up.
4. Append the round to `WORK_LOG.md`: profile findings, the change, the `time` result, the `check` result if run.
   Failed and no-gain rounds are logged too.

Keep `starVLA/` (branch `opt`), the switch file and `FINAL.json` (`{"tree": ..., "config": ..., "check": ...}`) at the
best version there is evidence for at all times, and the summary at the end of `WORK_LOG.md` current.

## Environment

- `$ASSETS=$WORK/starvla/assets`: `env.sh` (Python 3.12, torch 2.9.1+cu129, transformers 4.57.0,
  accelerate 1.5.2, deepspeed 0.16.9, flash-attn 2.8.3.post1, cuDNN 9.25.1), `models/Qwen3-VL-4B-Instruct`,
  `data/libero` (IPEC-COMMUNITY LeRobot LIBERO sets with starVLA's `modality.json`). Offline.
- Baseline numbers: `task/baseline.md`.
- Skill cards: `skills/` of ResearchBoost-Agent (priors from other tasks; re-verify here before relying on them).

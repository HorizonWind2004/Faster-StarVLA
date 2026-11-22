# Faster-StarVLA

[English](README.md) | [简体中文](README.zh-CN.md)

**2.00× faster starVLA training steps on the same GPUs, optimized by an AI agent with [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent).**

![Faster-StarVLA: 2.00x training-step speedup on 2 GPUs](docs/faster/figures/benchmark.svg)

| Implementation | Time / training step (ms) ↓ | Speedup ↑ | Validation |
|---|---:|---:|---|
| starVLA `99764f5` | 905.4 | 1.00× | |
| Faster-StarVLA, bit-exact switches | 521.9 | 1.74× | every recorded tensor bitwise identical |
| **Faster-StarVLA, all switches** | **453.3** | **2.00×** | within the baseline's seed-to-seed noise |

Measured on **2 × NVIDIA B300** with starVLA's [LIBERO recipe](examples/simBenchmarks/LIBERO/README.md): QwenOFT on
Qwen3-VL-4B, all four LIBERO suites in one policy, per-GPU batch 16, DeepSpeed ZeRO-2, bf16, every parameter
trainable (4.50 B). Reference → candidate → reference in three fresh processes with cold compile caches, 2 windows of
100 steps each after 20 warmup steps (speedup 1.9975×). The bit-exact row is one candidate run against the
reference timing frozen before optimization (906.9 ms). Peak memory drops from 66.5 to 59.0 GiB per
GPU with all switches.

## Quick Start

Prepare the environment, weights and data as in the [upstream instructions](README.upstream.md), then add one line
to the training launcher:

```bash
source scripts/configs/faster.env            # all switches: 2.00x, not bitwise (compiled layers)
# source scripts/configs/faster_bitwise.env  # bit-exact switches only: 1.74x

bash examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh
```

Every switch is an environment variable, off unless set: with none set the trainer runs the upstream code path.
The switches apply to the **QwenOFT framework on a Qwen3-VL backbone**; the two DeepSpeed switches apply to any
starVLA trainer that uses ZeRO-2.

## Agent Optimization History

![Optimization history: 1.00x to 2.07x](docs/faster/figures/history.svg)

Single timing runs against the frozen reference timing (906.9 ms); the headline number above is the final
three-process check. The full log, including the rounds that did not help, is [docs/faster/WORK_LOG.md](docs/faster/WORK_LOG.md).

## What changed

At per-GPU batch 16 the GPUs were busy 53% of the time. The step was not short of compute: it was waiting for the
host, launching ten thousand small kernels one by one, and copying gradients around.

- **Input processing in the data-loader workers** (`STARVLA_PREP_IN_WORKERS`). The prompt construction, the
  Qwen3-VL processor and the rotary position ids (0.2 s of single-threaded CPU per step) ran in the training process
  while the GPU waited. The workers, which have the samples first and time to spare, now run the same code and hand
  over tensors. A background thread in the training process did not help: it shares the GIL with the launch-bound
  forward.
- **Vision tower** (`STARVLA_BATCHED_VISION_ATTN`, `STARVLA_VISION_POS_CACHE`). With SDPA, transformers' Qwen3-VL
  vision attention synchronizes with the host in every layer and runs one attention call per image: 24 layers × 32
  images per step. All training images have the same size, so this is one batched call per layer. The position
  tables, which depend only on the grid size, are built once instead of every step.
- **Unused work removed** (`STARVLA_SKIP_LM_HEAD`, `STARVLA_FAST_ATTN_MASK`). QwenOFT reads the last hidden state;
  the vocabulary logits of every position were computed and dropped. The attention mask was built through four
  nested `torch.vmap` calls (150 ms of host time); two broadcasted tensor ops give the same mask.
- **DeepSpeed ZeRO-2** (`STARVLA_DS_FUSED_GRAD_NORM`, `STARVLA_DS_PLAIN_ALLREDUCE`). The clipping norm cast every
  gradient to fp64 first (18 GB written and read per step); `vector_norm(g, dtype=float64)` returns the same bits
  without the copy. Gradient buckets are all-reduced in place instead of sliced, concatenated, reduced and copied
  back.
- **Compiled layers** (`STARVLA_COMPILE_LM`, `STARVLA_COMPILE_VISION`). One forward + backward of the 36 decoder
  layers is about 10,000 GPU ops, a third of the time in matrix products and the rest in launching small
  elementwise kernels. Each layer is compiled with Inductor (training calls only; evaluation stays eager). After
  compilation the Python garbage collector paused single steps for 1–2.5 s walking compiler objects;
  `gc.freeze()` removes the pauses.

The switch table with the measured effect of each one is in [docs/faster/measurement.md](docs/faster/measurement.md#switches).

## Validation

Everything was judged by a frozen evaluator built before optimization started
([docs/faster/judge/](docs/faster/judge/), [how it decides](docs/faster/measurement.md#the-judge)).

**Bit-exact switches.** With the seven non-compile switches set, two 2-step training cases in deterministic mode
(one with the recipe's periodic evaluation between the steps), on both ranks: 23,336 recorded tensors (the
loss, and the fp32 gradient, update and both Adam states of all 729 parameters) are bitwise identical to upstream,
as are the learning rates, the random-number state and a digest of what was fed to the VLM.

**All switches.** The compiled layers change floating-point rounding, so the full switch set is compared
statistically: one 60-step training run in deterministic mode against upstream on the same seed; each endpoint must
stay within 3σ of upstream's own seed-to-seed spread (6 seeds, frozen before optimization).

| Endpoint | Difference / upstream seed σ | Within 3σ |
|---|---:|:---:|
| Training loss, steps 20-40 | +0.87 | ✓ |
| Training loss, steps 40-60 | +0.95 | ✓ |
| Held-out probe loss, steps 50-60 | +0.12 | ✓ |
| Action-head update size, steps 10-60 | +0.29 | ✓ |
| VLM update size, steps 10-60 | -0.40 | ✓ |
| Action-head update size, top elements, steps 1-6 (tracks the learning rate) | +0.22 | ✓ |
| VLM update size, top elements, steps 1-6 (tracks the learning rate) | -0.31 | ✓ |

A control run with every learning rate multiplied by 1.02 is rejected by this check (+16σ on the update-size
endpoints), as are a run whose VLM receives no gradient and a run with a different clipping threshold.

**LIBERO evaluation.** A full training run with all switches (30,000 steps, seed 42, 2 x B300, global batch 32) was evaluated with starVLA's LIBERO protocol, 500 episodes per suite, next to the released checkpoint in the same pipeline. The released checkpoint was trained by the starVLA authors on 8 GPUs (global batch 128, checkpoint at step 50,000); this is a check that the accelerated code trains a working policy, not a matched-seed comparison. The 30,000 steps took 3 h 51 min of wall time including the recipe's periodic evaluations and three checkpoint saves (462 ms per step overall); at the frozen upstream step time the steps alone would take 7.6 h (not run).

| Success rate (%) | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| starVLA Qwen3-VL-OFT, published | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Released checkpoint (`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`, step 50,000), our evaluation | 98.6 | 100.0 | 99.4 | 96.6 | 98.65 |
| **Faster-StarVLA run** (all switches, 30,000 steps on 2 GPUs) | 98.8 | 98.6 | 98.4 | 93.4 | 97.30 |

## Faster evaluation

`tools/eval/` holds the batching policy server used for the evaluations above: concurrent requests from the
simulation clients become one `predict_action` call instead of queueing one by one. See
[docs/faster/measurement.md](docs/faster/measurement.md#evaluation).

## Upstream starVLA

Based on [starVLA/starVLA](https://github.com/starVLA/starVLA) at `99764f5` (MIT license). The original README is
kept as [README.upstream.md](README.upstream.md); authorship and license are unchanged.

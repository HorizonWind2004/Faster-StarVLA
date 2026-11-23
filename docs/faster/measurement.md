# Measurement and validation

[← README](../../README.md)

## Setup

| | |
|---|---|
| Upstream | [starVLA/starVLA](https://github.com/starVLA/starVLA) `99764f5` |
| Workload | `examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` with `Framework_name=QwenOFT`, `base_vlm=Qwen3-VL-4B-Instruct`, `data_mix=libero_all` (1,693 episodes of the four suites, IPEC-COMMUNITY LeRobot format), per-GPU batch 16, every module trainable (4.50 B parameters), DeepSpeed ZeRO-2 through Accelerate, bf16 autocast, evaluation every 100 steps |
| Hardware | 2 × NVIDIA B300 on one node, 32 CPU cores (16 per GPU). Upstream runs the same per-GPU load on 8 GPUs |
| Software | Python 3.12, torch 2.9.1+cu129 (B300 needs torch ≥ 2.9), cuDNN 9.25.1, transformers 4.57.0, accelerate 1.5.2, deepspeed 0.16.9. Upstream pins torch 2.6 / torchvision 0.21; everything else is as pinned |
| Shapes | per rank and step: 16 samples × 2 views of 224 × 224 (196 patches → 49 visual tokens each), about 205 tokens per sample |

The exact arguments are `judge/bench/bench.json` (`train_args`).

## The judge

The verdicts come from a judge built and frozen before any optimization
([ResearchBoost-Agent's method](https://github.com/DAGroup-PKU/ResearchBoost-Agent/blob/main/docs/judge.md)):
`judge/bench/` (timing, structure checks, the starVLA adapter), `judge/numerics/` (the generic recorder and
comparison code, an unmodified copy) and `judge/task/` (the frozen reference timing, record manifests,
trajectories and σ). `judge/GOAL.md` is the task statement.

| Gate | What it checks |
|---|---|
| 0 Structure | The entry point's setup (`main`, optimizer and scheduler construction, `prepare_training`, everything in `train()` outside the per-step loop) equals upstream's AST; the data pipeline yields bitwise the upstream samples and first batches |
| 1 Benefit | Fresh processes of the real entry point, cold compile caches, 20 warmup steps, then 2 windows of 100 steps (each contains one of the recipe's evaluations). Final check: reference → candidate → reference; the speedup must exceed `max(1.02, 1 + 2 × reference spread)`; a compile inside a window invalidates the run |
| 2 Bitwise | Two 2-step cases in deterministic mode on both ranks. Exact items: RNG fingerprints, optimizer steps, learning rates, Adam hyper-parameters, DeepSpeed settings, the trainable parameter set, dtypes, and a digest of the VLM inputs. Then the loss and the fp32 gradient, update and Adam states of all 729 parameters, bit for bit |
| 3 Statistical | Only if gate 2 is not bitwise: a 60-step trajectory in deterministic mode on a frozen seed; seven endpoints within 3σ of upstream's seed-to-seed spread (6 seeds) |

Before optimization the judge was shown to reject what it should: a changed training setup (ERROR), a changed
clipping threshold (INCORRECT at gate 2), a VLM without gradient (INCORRECT at gate 3, −729σ), every learning rate
× 1.02 (INCORRECT at gate 3, +16σ); to pass an equivalent reordering of the loss mean (not bitwise, gate 3 PASS);
and to reproduce upstream on itself bit for bit (`judge/task/formulation.json`).

One judge change was made after freezing, with a regression record (`judge/task/judge_changes.md`): the adapter
located the dataset for its probe batch by scanning `gc.get_objects()`, which cannot see objects after the
candidate's `gc.freeze()`; it now records the dataset when the DataLoader is constructed. The reference trajectory
re-recorded with the changed adapter is identical to the frozen one in all 640 values, and the control arm is
rejected with the same numbers.

The PyTorch profiler and nsys cannot see GPU kernels on B300 (CUPTI has no sm_103 support). Profiling used cProfile
with and without `CUDA_LAUNCH_BLOCKING=1`, wall time per stage with a GPU sync at each stage edge
(`bench.py profile --stages`), and isolated micro-benchmarks.

## Results

### Final check

| Run (fresh process, cold caches) | Window 1 (ms/step) | Window 2 | Mean |
|---|---:|---:|---:|
| reference | 906.0 | 903.9 | 905.0 |
| **candidate, all switches** | 453.3 | 453.2 | 453.3 |
| reference | 908.8 | 902.8 | 905.8 |

Speedup **1.9975×** (threshold 1.02; the two reference runs differ by 0.09%). Gate 2: not bitwise (3,227 of 23,336 tensors identical, all exact items hold). Gate 3: PASS, tier N1. A first complete final check measured 2.0010× and stopped at gate 3 with a judge error (see the judge change above).

Bit-exact switch set (`scripts/configs/faster_bitwise.env`), agent-style check against the frozen reference timing: 521.9 ms, **1.74×**, gate 2: 23,336 of 23,336 tensors bitwise identical, PASS (N0).

### Where the time was, where it is

Milliseconds per step with a GPU sync at every stage edge (the syncs remove the overlap between host and GPU, so
the columns add up to slightly more or less than a timed step):

| Configuration | forward | backward | optimizer step | rest of the loop |
|---|---:|---:|---:|---:|
| upstream | 415 | 319 | 179 | 7 |
| inputs in workers, no LM head, direct mask | 204 | 316 | 170 | 10 |
| + vision attention, position cache, rotary ids, grad norm | 96 | 266 | 161 | 8 |
| + compiled decoder layers | 88 | 218 | 161 | 5 |

What is left is DeepSpeed ZeRO-2 itself: per rank and step it copies the 9 GB of bf16 gradients into reduction
buckets, all-reduces them, flattens and casts this rank's half to fp32, clips, runs fused AdamW over 2.25 B fp32
elements (about 107 ms on its own), casts back and all-gathers the updated half; and the per-parameter Python hooks
that drive it. Changing that needs a different ZeRO implementation, not a switch.

### What changed, in detail

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

### Switches

| Step | Switches added | ms / step | Speedup | Peak GiB |
|---|---|---:|---:|---:|
| upstream | starVLA 99764f5 | 906.9 | 1.00× | 66.5 |
| + inputs in workers, no LM head, direct mask | `STARVLA_PREP_IN_WORKERS`, `STARVLA_SKIP_LM_HEAD`, `STARVLA_FAST_ATTN_MASK` | 677.8 | 1.34× | 66.6 |
| + batched vision attention | `STARVLA_BATCHED_VISION_ATTN` | 588.8 | 1.54× | 66.2 |
| + vision position cache | `STARVLA_VISION_POS_CACHE` | 541.5 | 1.67× | 66.2 |
| + rotary ids in workers, fused grad norm | `STARVLA_PREP_IN_WORKERS (position ids)`, `STARVLA_DS_FUSED_GRAD_NORM` | 528.7 | 1.72× | 66.2 |
| + compiled decoder layers | `STARVLA_COMPILE_LM` | 493.0 | 1.84× | 59.9 |
| + in-place all-reduce | `STARVLA_DS_PLAIN_ALLREDUCE` | 472.4 | 1.92× | 59.9 |
| + compiled vision blocks, gc.freeze | `STARVLA_COMPILE_VISION` | 437.3 | 2.07× | 59.0 |

`STARVLA_DS_REDUCE_TO_OWNER` (DeepSpeed `use_multi_rank_bucket_allreduce = false`) is implemented and off: 686 ms
against 678 ms without it. An earlier `STARVLA_PREP_AHEAD` (processor in a background thread of the training
process) reached only 1.18× and was replaced by the worker version.

Flash-attention 2 instead of SDPA (upstream hard-codes SDPA) was timed once with the bit-exact switches that still
apply: 630.1 ms (1.44×) against 521.9 ms for the SDPA path with the batched vision attention and the direct mask. It
cannot be combined with the compiled layers (Dynamo cannot trace the flash-attn extension call). Not included.

## Evaluation

`tools/eval/run_eval.sh` starts one batching policy server per GPU (`tools/eval/batched_server.py`) and splits the (task, episode) pairs of each suite across simulation clients. starVLA's own server answers one request at a time, so clients queue behind each other. In October 2026, on upstream `985c223` (same evaluation files) and one B300, `libero_spatial` with 16 clients ran at about 1,800 episodes/h with the upstream server and 5,500 / 6,700 / 7,300 episodes/h with the batching server at 16 / 32 / 48 clients, with the same success rate (98-99% on the same 100 episodes). Batched and single-request inference can differ in the last bits of an action. The evaluations on this page used 2 GPUs and 28 clients: each 2,000-episode run took 17-18 min of simulation after 1.3 min of server start (6,700-7,000 episodes/h).

A full training run with all switches (30,000 steps, seed 42, 2 x B300, global batch 32) was evaluated with starVLA's LIBERO protocol, 500 episodes per suite, next to the released checkpoint in the same pipeline. The released checkpoint was trained by the starVLA authors on 8 GPUs (global batch 128, checkpoint at step 50,000); this is a check that the accelerated code trains a working policy, not a matched-seed comparison. The 30,000 steps took 3 h 51 min of wall time including the recipe's periodic evaluations and three checkpoint saves (462 ms per step overall); at the frozen upstream step time the steps alone would take 7.6 h (not run).

| Success rate (%) | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| starVLA Qwen3-VL-OFT, published | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Released checkpoint (`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`, step 50,000), our evaluation | 98.6 | 100.0 | 99.4 | 96.6 | 98.65 |
| **Faster-StarVLA run** (all switches, 30,000 steps on 2 GPUs) | 98.8 | 98.6 | 98.4 | 93.4 | 97.30 |

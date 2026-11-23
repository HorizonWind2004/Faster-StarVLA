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
| **candidate, all fast paths** | 453.3 | 453.2 | 453.3 |
| reference | 908.8 | 902.8 | 905.8 |

Speedup **1.9975×** (threshold 1.02; the two reference runs differ by 0.09%). Gate 2: not bitwise (3,227 of 23,336 tensors identical, all exact items hold). Gate 3: PASS, tier N1. A first complete final check measured 2.0010× and stopped at gate 3 with a judge error (see the judge change above).

Exact set (the seven paths of `STARVLA_FAST=exact`), agent-style check against the frozen reference timing: 521.9 ms, **1.74×**, gate 2: 23,336 of 23,336 tensors bitwise identical, PASS (N0).

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

- **Input processing in the data-loader workers** (`worker_prep`). The prompt construction, the
  Qwen3-VL processor and the rotary position ids (0.2 s of single-threaded CPU per step) ran in the training process
  while the GPU waited. The workers, which have the samples first and time to spare, now run the same code and hand
  over tensors. A background thread in the training process did not help: it shares the GIL with the launch-bound
  forward.
- **Vision tower** (`vision_attn`, `vision_pos`). With SDPA, transformers' Qwen3-VL
  vision attention synchronizes with the host in every layer and runs one attention call per image: 24 layers × 32
  images per step. All training images have the same size, so this is one batched call per layer. The position
  tables, which depend only on the grid size, are built once instead of every step.
- **Unused work removed** (`hidden_only`, `attn_mask`). QwenOFT reads the last hidden state;
  the vocabulary logits of every position were computed and dropped. The attention mask was built through four
  nested `torch.vmap` calls (150 ms of host time); two broadcasted tensor ops give the same mask.
- **DeepSpeed ZeRO-2** (`grad_norm`, `allreduce`). The clipping norm cast every
  gradient to fp64 first (18 GB written and read per step); `vector_norm(g, dtype=float64)` returns the same bits
  without the copy. Gradient buckets are all-reduced in place instead of sliced, concatenated, reduced and copied
  back.
- **Compiled layers** (`compile_lm`, `compile_vision`). One forward + backward of the 36 decoder
  layers is about 10,000 GPU ops, a third of the time in matrix products and the rest in launching small
  elementwise kernels. Each layer is compiled with Inductor (training calls only; evaluation stays eager). After
  compilation the Python garbage collector paused single steps for 1–2.5 s walking compiler objects;
  `gc.freeze()` removes the pauses.

### Step by step

| Step | Fast paths added | ms / step | Speedup | Peak GiB |
|---|---|---:|---:|---:|
| upstream | starVLA 99764f5 | 906.9 | 1.00× | 66.5 |
| + inputs in workers, no LM head, direct mask | `worker_prep`, `hidden_only`, `attn_mask` | 677.8 | 1.34× | 66.6 |
| + batched vision attention | `vision_attn` | 588.8 | 1.54× | 66.2 |
| + vision position cache | `vision_pos` | 541.5 | 1.67× | 66.2 |
| + rotary ids in workers, fused grad norm | `worker_prep`, `grad_norm` | 528.7 | 1.72× | 66.2 |
| + compiled decoder layers | `compile_lm` | 493.0 | 1.84× | 59.9 |
| + in-place all-reduce | `allreduce` | 472.4 | 1.92× | 59.9 |
| + compiled vision blocks, gc.freeze | `compile_vision` | 437.3 | 2.07× | 59.0 |

`reduce_to_owner` (DeepSpeed `use_multi_rank_bucket_allreduce = false`) is implemented and in no level: 686 ms
against 678 ms without it. An earlier version of `worker_prep` that ran the processor in a background thread of the
training process reached only 1.18× and was replaced by the worker version.

Flash-attention 2 instead of SDPA (upstream hard-codes SDPA) was timed once with the bit-exact switches that still
apply: 630.1 ms (1.44×) against 521.9 ms for the SDPA path with the batched vision attention and the direct mask. It
cannot be combined with the compiled layers (Dynamo cannot trace the flash-attn extension call). Not included.

## One setting for every recipe

The optimization rounds above were run with one environment variable per fast path. They were then moved into one
package, `starVLA/fast/`, behind one setting:

| `STARVLA_FAST` | Fast paths |
|---|---|
| `exact` | `attn_mask`, `hidden_only`, `vision_attn`, `vision_pos`, `worker_prep`, `grad_norm`, `allreduce` |
| `all` | `exact` + `vision_attn_relaxed`, `compile_lm`, `compile_vision` |
| `exact,-worker_prep,compile_lm` | a level and/or names; `-name` removes one |
| unset, `off`, `0` | none: the baseline code paths |

Each fast path is a module with `apply(target) -> bool`; `False` means "does not apply to this model", and the path is
skipped and listed as such in the log. The callers are `get_vlm_model` (every backbone interface), the three trainers
(the DeepSpeed engine) and `build_dataloader`. The variables of the first release are still honoured and select the
same paths:

| Fast path | First-release variable |
|---|---|
| `worker_prep` | `STARVLA_PREP_IN_WORKERS` |
| `hidden_only` | `STARVLA_SKIP_LM_HEAD`, `STARVLA_INFER_SKIP_LM_HEAD` |
| `attn_mask` | `STARVLA_FAST_ATTN_MASK` |
| `vision_attn` | `STARVLA_BATCHED_VISION_ATTN` |
| `vision_pos` | `STARVLA_VISION_POS_CACHE` |
| `grad_norm`, `allreduce`, `reduce_to_owner` | `STARVLA_DS_FUSED_GRAD_NORM`, `STARVLA_DS_PLAIN_ALLREDUCE`, `STARVLA_DS_REDUCE_TO_OWNER` |
| `compile_lm`, `compile_vision` | `STARVLA_COMPILE_LM`, `STARVLA_COMPILE_VISION` |

### The judge on the unified code

The judge was run in three sessions (three batch jobs, each with 2 GPUs and 32 cores of one node; nodes A and B are
the same hardware model, 8 × B300 and 2 × 64 cores). The headline recipe in each of them, the baseline being the
mean of the two reference runs of that session's three-process check:

| Session | Node | Baseline (ms/step) | `exact` | `all` | Three-process check of `all` |
|:---:|:---:|---:|---:|---:|---|
| 1: 22-23 Nov | A | 905.4 (905.0 / 905.8) | 521.9 (1.74×) | 453.3 (2.00×) | PASS |
| 2: 23 Nov, afternoon | B | 795.5 (787.6 / 803.4) | 510.1 (1.56×) | 425.3 (1.87×) | INVALID: baseline outside the frozen band |
| 3: 23 Nov, evening | A | 815.9 (823.8 / 808.1) | 501.9 (1.63×) | 421.5 (1.94×) | INVALID: baseline outside the frozen band |

The baseline ran 10 - 12% faster in sessions 2 and 3 than in session 1, in every part of the step-time distribution
(median 896 ms, then 777 - 819 ms), with the same code, environment, allocation and command line. Node A was
rebooted between sessions 1 and 3; what made it slower before was not determined. The accelerated code changed less
(`all` 6 - 7%, `exact` 2 - 4%), so the speedup measured in the later sessions is lower. The baseline spends more of its
step on the host than the accelerated code does, and it moved the most.

The judge's reference timing was frozen in session 1. An agent-style check compares one candidate run with that
frozen timing, so its speedup figure only means something in session 1; the table gives the later ones against their
own session's baseline instead. The three-process check measures its own references and refuses a verdict (INVALID)
when they are further from the frozen timing than the band fixed before optimization, which is what happened in
sessions 2 and 3. The judge and its frozen timing were left as they are. The bitwise gate does not depend on timing.

| Check | Session | Reference (ms/step) | Candidate | Speedup | Numerics | Verdict |
|---|:---:|---:|---:|---:|---|---|
| first release, exact set (per-path variables, before the refactor) | 1 | 906.9 (frozen) | 521.9 | 1.74× | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| first release, all (per-path variables, before the refactor) | 1 | 905.4 | 453.3 | 2.00× | seven endpoints within 0.95σ | PASS (N1) |
| unified, exact (STARVLA_FAST=exact, commit 7bf6a48) | 2 | 906.9 (frozen) | 510.1 | 1.56× against this session's baseline (795.5) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| unified, all (STARVLA_FAST=all, commit 7bf6a48) | 2 | 795.5 | 425.3 | 1.87× | not reached | INVALID: references outside the frozen band |
| unified final tree, exact (STARVLA_FAST=exact, final commit) | 3 | 906.9 (frozen) | 501.9 | 1.63× against this session's baseline (815.9) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| unified final tree, all (STARVLA_FAST=all, final commit) | 3 | 815.9 | 421.5 | 1.94× | not reached | INVALID: references outside the frozen band |

LIBERO with the released checkpoint and the policy servers on the unified code (`STARVLA_FAST=exact`): 97.85% over 2,000 episodes (Spatial 98.0 / Object 99.8 / Goal 99.0 / Long 94.6). The same evaluation harness with the first release's inference variables gave 98.25%; the sampling error of 2,000 episodes at this success rate is about 0.3 points. The harness is a faster LIBERO runner that is not part of this repository.

### Other frameworks and backbones: step time

Same data, batch size and timing procedure as the judge's benefit gate (fresh process, 20 warmup steps, two windows
of 100 steps), one run per cell, all in session 2; only `framework.name`, `framework.qwenvl.base_vlm` and, where stated,
the attention implementation differ from the recipe. The first row is the headline recipe, from the judge's own
processes in session 2.

| Framework | Backbone | Attention | Baseline (ms/step) | `exact` | `all` | `exact` checked bit for bit | Peak GiB: baseline / exact / all |
|---|---|---|---:|---:|---:|---|---|
| QwenOFT | Qwen3-VL-4B | SDPA | 795.5 | 510.1 (1.56×) | 425.3 (1.87×) | judge: 23,336 tensors; 728 gradients + actions ✓ | — / — / — |
| QwenGR00T | Qwen3-VL-4B | SDPA | 903.0 | 783.9 (1.15×) | 722.9 (1.25×) | 960 gradients + actions ✓ | 73.0 / 72.5 / 65.5 |
| QwenPI | Qwen3-VL-4B | SDPA | 1291.9 | 1177.4 (1.10×) | 1133.3 (1.14×) | 1,225 gradients + actions ✓ | 114.7 / 114.4 / 107.5 |
| QwenOFT | Qwen2.5-VL-3B | SDPA | 1021.7 | 694.1 (1.47×) | 554.4 (1.84×) | 840 gradients + actions ✓ | 64.1 / 63.6 / 55.3 |
| QwenOFT | Qwen2.5-VL-3B | flash-attention | 775.0 | 738.7 (1.05×) | same as `exact` | 840 gradients + actions ✓ | 63.2 / 63.2 / — |
| QwenGR00T | Qwen2.5-VL-3B | flash-attention | 857.0 | 830.0 (1.03×) | same as `exact` | 1,072 gradients + actions ✓ | 70.6 / 70.6 / — |

The baseline is the upstream code with the recipe's settings; speedups are against the baseline of the same row. The last column is a comparison of `exact` with the baseline on real weights: one training step (loss and every gradient) and one `predict_action` call.

This table is from a later session on a second node, in which the baseline of the headline recipe ran at 796 ms instead of 905 ms and `STARVLA_FAST=all` at 425 ms instead of 453 ms ([sessions](#the-judge-on-the-unified-code)). Every row is compared with a baseline from the same job.

The gain is largest on QwenOFT + Qwen3-VL because `worker_prep` has a builder only for that pair; elsewhere the processor still runs in the training process. No fast path touches the action heads of QwenGR00T and QwenPI.

The Qwen2.5-VL recipes load the backbone with flash-attention, where the SDPA-specific paths and the compiled layers do not apply and are skipped, so `all` selects the same paths as `exact` there. Loading the same backbone with SDPA (`framework.qwenvl.attn_implementation=sdpa`) and `STARVLA_FAST=all` is the fastest configuration measured for it: 554 ms against 775 ms for the flash-attention baseline (1.40×).

On Qwen2.5-VL with SDPA (bf16, B300), the bitwise probe rejects batching for the four full-attention vision layers (32 images × 256 patches, head dimension 80), so `exact` batches only the 28 window-attention layers; `all` batches both.

### Other frameworks and backbones: bitwise checks on real weights

`tests/test_fast_paths.py` checks the exact backbone paths on tiny random-weight Qwen3-VL and Qwen2.5-VL models on
the CPU. On the GPU, with the real weights in bf16, two scripts (`docs/faster/coverage/`) compare the baseline and
`STARVLA_FAST=exact`, each after confirming that the baseline reproduces itself:

- backbone: one forward + backward of 16 samples × 2 views; the last hidden state and every gradient;
- framework: one training step of the whole framework (loss and every gradient) and one `predict_action` call
  (the returned actions).

| Check | Framework | Backbone | Attention | dtype | Fast paths on | Not applicable | `vision_attn` probe | Compared | Result |
|---|---|---|---|---|---|---|---|---|:---:|
| `bb_q25` | (backbone only) | Qwen2.5-VL-3B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | 32 x 256: kept; 128 x 64: batched | hidden state, 824 gradients | identical |
| `bb_q25_relaxed_fp32` | (backbone only) | Qwen2.5-VL-3B | SDPA | fp32 | attn_mask, hidden_only, vision_attn, vision_pos, vision_attn_relaxed | — | — | hidden state, 824 gradients | identical |
| `bb_q3` | (backbone only) | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | 32 x 256: batched | hidden state, 712 gradients | identical |
| `fw_gr00t_q25` | QwenGR00T | Qwen2.5-VL-3B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | — | loss, 1,072 gradients, actions | identical |
| `fw_gr00t_q25_fa2` | QwenGR00T | Qwen2.5-VL-3B | flash-attention | bf16 | hidden_only, vision_pos | attn_mask, vision_attn | — | loss, 1,072 gradients, actions | identical |
| `fw_gr00t_q3` | QwenGR00T | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | — | loss, 960 gradients, actions | identical |
| `fw_oft_q25` | QwenOFT | Qwen2.5-VL-3B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | — | loss, 840 gradients, actions | identical |
| `fw_oft_q25_fa2` | QwenOFT | Qwen2.5-VL-3B | flash-attention | bf16 | hidden_only, vision_pos | attn_mask, vision_attn | — | loss, 840 gradients, actions | identical |
| `fw_oft_q3` | QwenOFT | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | — | loss, 728 gradients, actions | identical |
| `fw_pi_q3` | QwenPI | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos | — | — | loss, 1,225 gradients, actions | identical |

Two things these checks established:

- **Batched vision attention is exact only for some shapes.** One attention call on [32 images, 16 heads, 256 patches]
  is bitwise identical to 32 single calls at head dimension 64 (Qwen3-VL-4B) but differs in the last bit at head
  dimension 72 or 80 (the four full-attention layers of Qwen2.5-VL-3B) in bf16 on B300; windows of 64 patches are
  identical at all three head dimensions, and fp32 is identical in every case. `vision_attn` therefore runs both forms once
  on random tensors of each shape, dtype and device it meets, forward and backward, and batches a shape only if they
  agree bit for bit. `vision_attn_relaxed` (level `all`) batches every uniform shape.
- **Upstream's QwenGR00T and QwenPI gradients are not reproducible run to run**: their action heads call PyTorch's
  fused scaled-dot-product-attention kernels, whose backward is not deterministic. The framework checks for these two
  were made with those two kernels disabled for the baseline and the candidate alike.

With flash-attention (the default of the Qwen2.5-VL recipes; the Qwen3-VL interface always uses SDPA), `attn_mask`,
`vision_attn` and the compiled layers do not apply and are skipped: flash-attention builds no SDPA mask, already makes
one variable-length call per vision layer, and its extension call cannot be traced by the compiler. Before this was handled, `STARVLA_FAST=all` on a flash-attention backbone failed in the first training step, in the compiled vision blocks. With the paths skipped, the two flash-attention recipes run with `STARVLA_FAST=all` at 747.2 ms (QwenOFT) and 849.3 ms (QwenGR00T) in session 3; the session 2 rows of the table above are not comparable with these.

## Evaluation

`tools/eval/run_eval.sh` starts one batching policy server per GPU (`tools/eval/batched_server.py`) and splits the (task, episode) pairs of each suite across simulation clients. starVLA's own server answers one request at a time, so clients queue behind each other. In October 2026, on upstream `985c223` (same evaluation files) and one B300, `libero_spatial` with 16 clients ran at about 1,800 episodes/h with the upstream server and 5,500 / 6,700 / 7,300 episodes/h with the batching server at 16 / 32 / 48 clients, with the same success rate (98-99% on the same 100 episodes). Batched and single-request inference can differ in the last bits of an action. The evaluations on this page used 2 GPUs and 28 clients: each 2,000-episode run took 17-18 min of simulation after 1.3 min of server start (6,700-7,000 episodes/h).

A full training run with all fast paths (30,000 steps, seed 42, 2 x B300, global batch 32) was evaluated with starVLA's LIBERO protocol, 500 episodes per suite, next to the released checkpoint in the same pipeline. The released checkpoint was trained by the starVLA authors on 8 GPUs (global batch 128, checkpoint at step 50,000); this is a check that the accelerated code trains a working policy, not a matched-seed comparison. The 30,000 steps took 3 h 51 min of wall time including the recipe's periodic evaluations and three checkpoint saves (462 ms per step overall); at the frozen upstream step time the steps alone would take 7.6 h (not run).

| Success rate (%) | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| starVLA Qwen3-VL-OFT, published | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Released checkpoint (`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`, step 50,000), our evaluation | 98.6 | 100.0 | 99.4 | 96.6 | 98.65 |
| **Faster-StarVLA run** (all fast paths, 30,000 steps on 2 GPUs) | 98.8 | 98.6 | 98.4 | 93.4 | 97.30 |

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

Exact set (the seven paths of `STARVLA_FAST=exact` that apply to this recipe), agent-style check against the frozen reference timing: 521.9 ms, **1.74×**, gate 2: 23,336 of 23,336 tensors bitwise identical, PASS (N0).

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
| `exact` | `attn_mask`, `hidden_only`, `vision_attn`, `vision_pos`, `worker_prep`, `worker_examples`, `grad_norm`, `allreduce` |
| `all` | `exact` + `sdpa`, `vision_attn_relaxed`, `compile_lm`, `compile_vision`, `compile_head`, `compile_world_model`, `text_longest`, `condition_once`, `unused_views`, `openpi_one_pass`, `openpi_compile`, `adamw_compiled`: the same operations at the same precision, in another order or from another kernel |
| `max` | `all` + `tf32`, `openpi_sdpa`: the precision of an operation changes (on the Qwen frameworks `max` is `all,tf32`) |
| by name only | `reduce_to_owner` (slower on two ranks), `grad_cast` and `unit_clip` (DeepSpeed's step: no measurable difference, and `unit_clip` reads the clipping factor on the host every step) |
| `exact,-worker_prep,compile_lm` | a level and/or names; `-name` removes one |
| unset, `off`, `0` | none: the baseline code paths |

Each fast path is a function `apply(target) -> bool`; `False` means "does not apply to this model", and the path is
skipped and listed as such in the log. The callers are four hooks, none of them inside a framework or a backbone
interface: `get_vlm_model` (every backbone interface), `build_framework` (the framework as a whole: action head,
PI0 / PI05), the three trainers (the DeepSpeed engine) and `build_dataloader` (the paths that move work into the
data-loader workers). The variables of the first release are still honoured and select the same paths:

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

The three-process check was run in four sessions (batch jobs with 2 GPUs and 32 cores of one node; nodes A, B and C
are the same hardware model, 8 × B300 and 2 × 64 cores). The headline recipe in each of them, the baseline being the
mean of the two reference runs of that session's check:

| Session | Node | Baseline (ms/step) | `exact` | `all` | Three-process check of `all` |
|:---:|:---:|---:|---:|---:|---|
| 1: 22-23 Nov | A | 905.4 (905.0 / 905.8) | 521.9 (1.74×) | 453.3 (2.00×) | PASS |
| 2: 23 Nov, afternoon | B | 795.5 (787.6 / 803.4) | 510.1 (1.56×) | 425.3 (1.87×) | INVALID: baseline outside the frozen band |
| 3: 23 Nov, evening | A | 815.9 (823.8 / 808.1) | 501.9 (1.63×) | 421.5 (1.94×) | INVALID: baseline outside the frozen band |
| 8: 24 Nov | C | 811.3 (806.4 / 816.2) | 490.5 (1.65×) | 413.5 (1.96×) | INVALID: baseline outside the frozen band |

The baseline ran 10 - 12% faster in the later sessions than in session 1, in every part of the step-time distribution
(median 896 ms in session 1, 777 - 819 ms in sessions 2 and 3), with the same code, environment, allocation and
command line. Node A was rebooted between sessions 1 and 3; what made it slower before was not determined, and the
same two levels appeared on node C from one job to the next (see the step-time section below). The accelerated code
changed less (`all` 6 - 9%, `exact` 2 - 6%), so the speedup measured in the later sessions is lower. The baseline spends
more of its step on the host than the accelerated code does, and it moved the most.

The judge's reference timing was frozen in session 1. An agent-style check compares one candidate run with that
frozen timing, so its speedup figure only means something in session 1; the table gives the later ones against their
own session's baseline where that session measured one. The three-process check measures its own references and
refuses a verdict (INVALID) when they are further from the frozen timing than the band fixed before optimization,
which is what happened in sessions 2, 3 and 8; it stops there, before the bitwise and statistical gates. Those gates
do not depend on timing, so for the later code they come from agent-style checks. The judge and its frozen timing
were left as they are.

| Check | Session | Reference (ms/step) | Candidate | Speedup | Numerics | Verdict |
|---|:---:|---:|---:|---:|---|---|
| first release, exact set (per-path variables, before the refactor) | 1 | 906.9 (frozen) | 521.9 | 1.74× | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| first release, all (per-path variables, before the refactor) | 1 | 905.4 | 453.3 | 2.00× | seven endpoints within 0.95σ | PASS (N1) |
| unified, exact (STARVLA_FAST=exact, commit 7bf6a48) | 2 | 906.9 (frozen) | 510.1 | 1.56× against this session's baseline (795.5) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| unified, all (STARVLA_FAST=all, commit 7bf6a48) | 2 | 795.5 | 425.3 | 1.87× | not reached | INVALID: references outside the frozen band |
| unified final tree, exact (STARVLA_FAST=exact, final commit) | 3 | 906.9 (frozen) | 501.9 | 1.63× against this session's baseline (815.9) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| unified final tree, all (STARVLA_FAST=all, final commit) | 3 | 815.9 | 421.5 | 1.94× | not reached | INVALID: references outside the frozen band |
| worker_prep for every framework, exact (STARVLA_FAST=exact, commit b14fd6b) | 4 | 906.9 (frozen) | 535.4 | 1.69× against this session's baseline (904.5) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| all + tf32 (STARVLA_FAST=all,tf32, commit 8332201) | 6 | 906.9 (frozen) | 446.4 | 2.04× against this session's baseline (912.9) | seven endpoints within 1.41σ | PASS (N2) |
| final tree, exact (STARVLA_FAST=exact, commit e5f7e15) | 8 | 906.9 (frozen) | 490.5 | 1.65× against this session's baseline (811.3) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| final tree, all (STARVLA_FAST=all, commit e5f7e15, three-process check) | 8 | 811.3 | 413.5 | 1.96× | not reached | INVALID: references outside the frozen band |
| commit 8427140, all (STARVLA_FAST=all, against the frozen timing) | 10 | 906.9 (frozen) | 426.4 | no baseline was timed in this session | seven endpoints within 0.95σ | PASS (N1) |
| commit 8427140, exact (STARVLA_FAST=exact) | 10 | 906.9 (frozen) | 498.0 | no baseline was timed in this session | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |
| released commit, all (STARVLA_FAST=all (adamw_compiled, compile_head), against the frozen timing) | 12 | 906.9 (frozen) | 340.3 | 2.36× against this session's baseline (801.6) | seven endpoints within 1.77σ | PASS (N1) |
| released commit, exact (STARVLA_FAST=exact) | 12 | 906.9 (frozen) | 500.3 | 1.60× against this session's baseline (801.6) | 23,336 of 23,336 tensors bitwise identical | PASS (N0) |

LIBERO with the released checkpoint and the policy servers on the unified code (`STARVLA_FAST=exact`): 97.85% over 2,000 episodes (Spatial 98.0 / Object 99.8 / Goal 99.0 / Long 94.6). The same evaluation harness with the first release's inference variables gave 98.25%; the sampling error of 2,000 episodes at this success rate is about 0.3 points. The harness is a faster LIBERO runner that is not part of this repository.

### Other frameworks and backbones: step time

Same data, batch size and timing procedure as the judge's benefit gate (fresh process, 20 warmup steps, two windows
of 100 steps on 2 GPUs), one run per cell. Only the framework, the backbone and, where stated, the attention
implementation differ from the headline recipe; QwenAdapter, QwenDiscreteDiffusion, LangForce and QwenDual use the
framework settings of their own configs (`docs/faster/coverage/configs/`), PI0 and PI05 the openpi LIBERO recipe with
the converted base checkpoints. Each row is the latest session that measured that combination, with the baseline of
the same job:

| Framework | Backbone | Attention in the recipe | Baseline (ms/step) | `exact` | `all` | `max` | `exact` checked bit for bit | Session | Peak GiB: baseline / exact / all |
|---|---|---|---:|---:|---:|---:|---|:---:|---|
| QwenOFT | Qwen3-VL-4B | SDPA | 801.6 | 489.8 (1.64×) | 292.2 (2.74×) | 305.5 (2.62×) | judge: 23,336 tensors; 728 gradients + actions ✓ | 12 | 66.5 / 66.2 / 59.0 |
| QwenGR00T | Qwen3-VL-4B | SDPA | 880.4 | 605.2 (1.45×) | 399.1 (2.21×) | 392.6 (2.24×) | 960 gradients + actions ✓ | 12 | 73.0 / 72.5 / 61.6 |
| QwenPI | Qwen3-VL-4B | SDPA | 1293.8 | 1110.8 (1.16×) | 811.7 (1.59×) | 530.5 (2.44×) | 1,225 gradients + actions ✓ | 12 | 114.7 / 114.4 / 105.0 |
| QwenPI_v3 | Qwen3-VL-4B | SDPA | 1638.3 | 1389.5 (1.18×) | 849.8 (1.93×) | 519.7 (3.15×) | 1,376 gradients + actions ✓ | 12 | 122.6 / 122.2 / 99.5 |
| QwenDiscreteDiffusion | Qwen3-VL-4B | SDPA | 2512.9 | 2295.7 (1.09×) | 1884.9 (1.33×) | 566.9 (4.43×) | 1,220 gradients + actions ✓ | 12 | 143.1 / 142.7 / 127.6 |
| QwenAdapter | Qwen3-VL-4B | SDPA | 1052.3 | 767.9 (1.37×) | 553.0 (1.90×) | 555.8 (1.89×) | 1,226 gradients + actions ✓ | 12 | 88.2 / 87.8 / 78.1 |
| LangForce | Qwen3-VL-4B | SDPA | 1439.6 | 905.7 (1.59×) | 686.5 (2.10×) | 673.7 (2.14×) | 961 gradients + actions ✓ | 12 | 104.4 / 103.6 / 85.8 |
| QwenDual | Qwen3-VL-4B | SDPA | 1201.1 | 909.2 (1.32×) | 546.6 (2.20×) | 521.0 (2.31×) | 1,135 of 1,136 gradients + actions (the other 1 differs between two baseline runs) ✓ | 12 | 87.9 / 87.5 / 66.8 |
| QwenOFT | Qwen2.5-VL-3B | SDPA | 1013.8 | 539.9 (1.88×) | 317.3 (3.20×) | — | 840 gradients + actions ✓ | 12 | 64.1 / 63.6 / 55.3 |
| QwenOFT | Qwen2.5-VL-3B | flash-attention | 877.6 | 645.6 (1.36×) | 466.5 (1.88×) | — | 840 gradients + actions ✓ | 6 | 63.2 / 63.2 / 55.3 |
| QwenGR00T | Qwen2.5-VL-3B | flash-attention | 861.9 | 692.9 (1.24×) | 405.3 (2.13×) | 405.2 (2.13×) | 1,072 gradients + actions ✓ | 12 | 70.6 / 70.6 / 59.2 |
| PI0 | PaliGemma (openpi) | eager | 639.2 | 591.5 (1.08×) | 290.3 (2.20×) | 288.9 (2.21×) | — | 12 | 102.5 / 102.5 / 63.0 |
| PI05 | PaliGemma (openpi) | eager | 670.2 | 626.8 (1.07×) | 308.3 (2.17×) | 315.9 (2.12×) | — | 12 | 113.2 / 113.2 / 70.0 |
| QwenGR00T | Qwen3.5-0.8B | flash-attention | 599.9 | 443.7 (1.35×) | 407.3 (1.47×) | 406.7 (1.48×) | — | 13 | 28.2 / 27.9 / 25.8 |
| QwenOFT | Qwen3.5-0.8B | flash-attention | 521.9 | 361.0 (1.45×) | 326.3 (1.60×) | 321.5 (1.62×) | — | 13 | 21.7 / 21.2 / 20.9 |
| Gemma4PI | Gemma-4-E2B (batch 4) | eager | 1080.1 | 965.0 (1.12×) | 594.3 (1.82×) | 581.3 (1.86×) | — | 13 | 164.3 / 164.3 / 82.0 |
| WanGR00T | Wan2.2-TI2V-5B (world model) | SDPA | 1584.0 | 1590.2 (1.00×) | 1111.7 (1.42×) | 1138.2 (1.39×) | — | 13 | 155.2 / 155.2 / 139.1 |

The baseline is the upstream code with the recipe's settings; a speedup is against the baseline of the same row, which was measured in the same batch job. The last column of the README's table is a comparison of `exact` with the baseline on real weights: one training step (loss and every gradient) and one `predict_action` call.

Every row is compared with a baseline from its own batch job. Between jobs on the same node the baseline of one recipe moved by up to 10%, so milliseconds of different jobs are not comparable; the ratios within a row are ([sessions](#other-frameworks-and-backbones-step-time)).

`exact` gains where the VLM's input processing and the backbone are a large part of the step (LangForce builds two prompts per step: 1.59×) and little on PI0 / PI05 (1.08×), whose step is the PaliGemma forward and backward itself.

The flow-matching heads run in fp32. `condition_once` (in `all`) projects the conditioning that a framework repeats for several noise levels once per sample. `max` also lets the head's fp32 matrix products use TF32; where every block of the head projects VLM hidden states, those products are most of what is left (QwenPI 812 → 531 ms; QwenPI_v3 850 → 520 ms; QwenDiscreteDiffusion 1885 → 567 ms). TF32 is a precision setting rather than a reordering, so it is a level of its own. The judge accepts it on the headline recipe as statistically equivalent; the same test on QwenPI (`all` on 6 seeds and `max` on 4 seeds) finds every endpoint within the judge's limit of 3σ, the largest being 1.6σ (VLM update size with `max`) ([details](#the-statistical-gate-on-qwenpi)), and the loss curves of the other frameworks over the 220 timed steps are indistinguishable from the baseline's ([numbers](#tf32)).

The Qwen2.5-VL recipes load the backbone with flash-attention. `exact` keeps it and applies the paths that do not depend on the attention kernel. `all` first switches the backbone to PyTorch's scaled-dot-product attention (`sdpa`), after which the SDPA paths and the compiled layers apply: QwenOFT 878 → 646 (1.36×) → 467 (1.88×) ms.

On Qwen2.5-VL with SDPA (bf16, B300), the bitwise probe rejects batching for the four full-attention vision layers (32 images × 256 patches, head dimension 80), so `exact` batches only the 28 window-attention layers; `all` batches both.

PI0 and PI05 bring their own PaliGemma instead of a `get_vlm_model` backbone. `exact` is `worker_examples` and the two DeepSpeed paths there; `all` adds `unused_views` (LIBERO has two of the three camera views these models expect, so the prefix shrinks from 816 to 560 tokens) and `openpi_compile`; `max` adds `openpi_sdpa`. The whole model already runs in bf16, so TF32 changes nothing there.

Sessions 4 to 8 are five batch jobs on one node (2 GPUs and 32 cores each). The baseline is not constant across
them: the same recipe, code and command line gave, for example, QwenGR00T + Qwen3-VL-4B at 1001, 998, 905, 903, 880 ms in sessions 4, 5, 7, 11, 12; QwenPI + Qwen3-VL-4B at 1445, 1356, 1292, 1303, 1294 ms in sessions 4, 5, 7, 11, 12; QwenOFT + Qwen3-VL-4B at 904, 913, 802, 786, 802 ms in sessions 4, 6, 8, 11, 12. Every combination measured in
every session is in `docs/faster/results/unified/coverage.json` (`timing_by_session`).

The headline recipe with `all` ran at 425 ms in session 4 and at 464 ms in session 6 (baselines 904 and 913 ms), with the same fast paths applied to that recipe by both trees. In session 8 the final tree and the tree of session 4 were run alternately in one job: final tree 420.8 and 414.6 ms, session 4 tree 426.2 and 427.1 ms, baseline 801.8 ms at the start of the series and 807.0 ms at its end.

PI0 and PI05 in session 9, `all` and `max` alternating in one job (what `max` adds there is `openpi_sdpa`; TF32 has nothing to act on in a model that runs in bf16). PI0: baseline 636.9 ms, `exact` 591.8, `all` 453.6 and 452.2, `max` 443.3 and 439.8 (peak memory 102.5 / 63.0 / 57.2 GiB for baseline / `all` / `max`). PI05: baseline 666.5 ms, `exact` 626.9, `all` 468.6 and 475.0, `max` 465.1 and 463.8 (peak memory 113.2 / 70.0 / 63.2 GiB for baseline / `all` / `max`).

### TF32

The action heads of the flow-matching frameworks run in fp32, and PyTorch's default keeps fp32 matrix products at
full precision. `tf32` (level `max`, or `STARVLA_FAST=all,tf32`) sets `torch.backends.cuda.matmul.allow_tf32` and the
cuDNN equivalent after the framework is built. In the judge's terms this is tier N2, a change of precision, which
its rules allow if the statistical gate passes. The judge on the headline recipe: PASS, tier N2: 3,233 of 23,336 tensors bitwise identical, seven endpoints within 1.41σ of upstream's seed-to-seed spread, 446.4 ms against the frozen 906.9 ms (session 6). Training loss of the timed runs: mean
± standard error over steps 100 to 220 on rank 0. Runs do not draw the same batches, so the columns of a row differ
by chance as well; `exact` is bitwise identical to the baseline and shows how much:

| Framework | Backbone | Session | Baseline | `exact` | `all` | `max` |
|---|---|:---:|---:|---:|---:|---:|
| QwenOFT | Qwen3-VL-4B | 12 | 0.2538 ± 0.0019 | 0.2529 ± 0.0018 | 0.2528 ± 0.0019 | 0.2527 ± 0.0018 |
| QwenGR00T | Qwen3-VL-4B | 12 | 0.9936 ± 0.0088 | 0.9567 ± 0.01 | 1.004 ± 0.0086 | 0.9997 ± 0.0083 |
| QwenPI | Qwen3-VL-4B | 12 | 521.6 ± 34 | 493.2 ± 33 | 472.6 ± 33 | 554.5 ± 35 |
| QwenPI_v3 | Qwen3-VL-4B | 12 | 0.57 ± 0.0065 | 0.5753 ± 0.0067 | 0.5802 ± 0.0072 | 0.5785 ± 0.007 |
| QwenDiscreteDiffusion | Qwen3-VL-4B | 12 | 4.456 ± 0.015 | 4.452 ± 0.015 | 4.463 ± 0.015 | 4.463 ± 0.015 |
| QwenAdapter | Qwen3-VL-4B | 12 | 0.2472 ± 0.0019 | 0.252 ± 0.0018 | 0.2512 ± 0.0018 | 0.2515 ± 0.0019 |
| LangForce | Qwen3-VL-4B | 12 | 0.6437 ± 0.019 | 0.6523 ± 0.02 | 0.652 ± 0.019 | 0.632 ± 0.02 |
| QwenDual | Qwen3-VL-4B | 12 | 0.972 ± 0.0093 | 0.9893 ± 0.0093 | 0.9993 ± 0.0081 | 0.9828 ± 0.0093 |
| QwenGR00T | Qwen2.5-VL-3B | 12 | 0.8021 ± 0.017 | 0.808 ± 0.018 | 0.8046 ± 0.018 | 0.8041 ± 0.017 |
| PI0 | PaliGemma (openpi) | 12 | 0.008892 ± 0.00043 | 0.008892 ± 0.00043 | 0.00889 ± 0.00043 | 0.008895 ± 0.00043 |
| PI05 | PaliGemma (openpi) | 12 | 0.07843 ± 0.0012 | 0.07843 ± 0.0012 | 0.07841 ± 0.0012 | 0.07841 ± 0.0012 |
| QwenGR00T | Qwen3.5-0.8B | 13 | 0.7574 ± 0.016 | 0.751 ± 0.016 | 0.733 ± 0.016 | 0.7546 ± 0.017 |
| QwenOFT | Qwen3.5-0.8B | 13 | 0.256 ± 0.0019 | 0.2575 ± 0.0019 | 0.2546 ± 0.0019 | 0.2564 ± 0.0019 |
| Gemma4PI | Gemma-4-E2B (batch 4) | 13 | 0.9892 ± 0.019 | 0.9961 ± 0.018 | 0.9801 ± 0.019 | 1.008 ± 0.018 |
| WanGR00T | Wan2.2-TI2V-5B (world model) | 13 | 0.7208 ± 0.017 | 0.7066 ± 0.016 | 0.7047 ± 0.016 | 0.706 ± 0.016 |

### Where the time goes, and the paths added after it (session 11)

**The optimizer.** One rank's ZeRO-2 partition of the headline recipe is a flat fp32 tensor of 2.25 billion elements. Each piece of DeepSpeed's step on a tensor of that size, on one NVIDIA B300 SXM6 AC (ms):

| Operation | ms |
|---|---:|
| one pass that reads two tensors and writes one (p.add_(g)) | 3.9 |
| flatten the bf16 gradients | 5.3 |
| cast the gradients to fp32 | 5.5 |
| clip (g.mul_(c)) | 2.6 |
| copy the weights back to bf16 | 1.9 |
| AdamW, fused | 104.9 |
| AdamW, foreach | 33.3 |
| AdamW, compiled | 9.4 |

Torch's fused AdamW kernel needs about ten times the time of one pass over the four tensors it reads and writes. The update written out and compiled with Inductor (`adamw_compiled`) runs at that pass's speed. After three steps it differs from the fused kernel in 1.4% of the elements, by rounding, as torch's own foreach implementation does (1.2%).

**Where a step goes** with `STARVLA_FAST=all` before `adamw_compiled` (ms per step; the stages are synchronized at their edges, which removes the overlap between them, so the sum is not the step time):

| Framework, `all` | Forward | Backward | Optimizer step |
|---|---:|---:|---:|
| QwenOFT + Qwen3-VL-4B | 71 | 152 | 154 |
| QwenGR00T + Qwen3-VL-4B | 106 | 216 | 168 |
| QwenAdapter + Qwen3-VL-4B | 123 | 288 | 218 |
| PI0 + PaliGemma (openpi) | 91 | 161 | 135 |

**Each new path on and off**, alternating in one job (the rest of `all` on in both; `adamw_compiled` was not yet part of `all` in this job):

| Path | Framework | With (ms/step) | Without | Baseline |
|---|---|---:|---:|---:|
| `adamw_compiled` | QwenOFT + Qwen3-VL-4B | 299.9 / 310.3 | 422.6 / 410.8 | 786.3 |
| `grad_cast` + `unit_clip` | QwenOFT + Qwen3-VL-4B | 496.1 / 491.8 | 491.4 / 498.6 | 786.3 |
| `openpi_one_pass` | PI0 + PaliGemma (openpi) | 378.8 / 378.6 | 448.7 / 447.1 | 649.7 |
| `compile_head` | QwenAdapter + Qwen3-VL-4B | 623.0 / 626.6 | 688.1 / 681.6 | 1040.0 |
| `compile_head` | QwenPI + Qwen3-VL-4B | 963.7 | 977.9 | 1302.9 |
| `compile_head` | QwenGR00T + Qwen3-VL-4B | 495.0 (one run failed, see below) | 492.1 / 498.7 | 902.9 |

`grad_cast` and `unit_clip` made no difference beyond run-to-run noise and are selectable by name only. Three of the runs with `compile_head` on (QwenGR00T, QwenPI) failed in the first step inside Inductor's tiling analysis, which does not support dynamic shapes and fails intermittently; the head is now compiled with that analysis off, and with eager's random draws for its dropout (session 12 checks both).

**Compilation at start.** PI0 with `all` reached its first step after 445 s with empty compiler caches and after 165 s with the caches of that first run. Inductor keeps its cache in a directory that persists between runs on one machine (`TORCHINDUCTOR_CACHE_DIR`, by default under the system's temporary directory); point it at shared storage to reuse it across machines.

### The statistical gate on QwenPI

The judge's statistical gate is frozen on the headline recipe, whose action head is a small MLP. `coverage/stat_gate.py` runs the same comparison with the judge's own recorder, trajectory case (60 training steps in deterministic mode, the recipe's periodic evaluation included) and endpoints, for QwenPI, whose head is an fp32 DiT: upstream on 6 seeds gives the seed-to-seed standard deviation σ of every endpoint, and a candidate is compared with upstream on the same seeds. The judge's criterion is that the mean same-seed difference stays within 3σ on every endpoint.

| Endpoint | Upstream, 6 seeds: mean (σ) | `exact`, seed 0: same-seed difference | upstream again, seeds 0, 1: same-seed difference | `all`, 6 seeds: mean same-seed difference (unpaired: difference of means, permutation p) | `max`, 4 seeds: mean same-seed difference (unpaired: difference of means, permutation p) |
|---|---:|---:|---:|---:|---:|
| training loss, steps 20-40 | 10.94 (3.9) | -3.26σ **beyond 3σ** | -1.11σ | -0.71σ (-0.71σ, p = 0.25) | -1.09σ (-0.63σ, p = 0.40) |
| training loss, steps 40-60 | 1.467 (0.24) | -1.11σ | -1.16σ | -0.11σ (-0.11σ, p = 0.85) | -0.43σ (-0.00σ, p = 1.00) |
| loss on a held-out batch, steps 50-60 | 1.236 (0.069) | -1.76σ | -0.64σ | -0.04σ (-0.04σ, p = 0.92) | -0.08σ (+0.47σ, p = 0.60) |
| action head update size, steps 10-60 | 6.734e-06 (1.5e-07) | +1.65σ | +1.79σ | -0.79σ (-0.79σ, p = 0.35) | +0.31σ (-0.09σ, p = 0.93) |
| VLM update size, steps 10-60 | 8.68e-07 (2.2e-08) | -0.99σ | +0.77σ | -0.09σ (-0.09σ, p = 0.89) | -1.61σ (-1.44σ, p = 0.15) |
| action head update size, steps 1-6 | 2.806e-05 (1.1e-08) | -5.97σ **beyond 3σ** | -0.10σ | +0.06σ (+0.06σ, p = 0.94) | +0.95σ (+1.14σ, p = 0.26) |
| VLM update size, steps 1-6 | 2.807e-06 (2.1e-09) | -6.98σ **beyond 3σ** | -1.28σ | -0.35σ (-0.35σ, p = 0.51) | -0.52σ (-0.35σ, p = 0.62) |

- **Upstream does not reproduce itself on a seed.** A second run of upstream on seeds 0, 1 differs from the first by -2.8σ to +2.1σ on single endpoints and seeds (training loss of steps 20 to 40 on seed 0: -2.8σ; action head update size: +1.8σ and +1.8σ). Upstream is as far from itself as the candidates are from it. The cause is known from the bitwise checks: the fused scaled-dot-product-attention kernels in the action head have a non-deterministic backward, and the first steps of this framework have losses in the thousands that change by a factor of several from step to step, so a last-bit difference grows to a different trajectory within a few steps.
- **`exact` departs from upstream at the same step as upstream's own second run.** On seed 0 all three runs have the same loss in the first two steps, to the last digit; in the third step upstream's second run and `exact` both differ from the first run in the fourth digit, and in the fifth step the three losses are 4,144 (first run), 10,320 (second run) and 24,398 (`exact`). On one seed `exact` is therefore "different" by the same-seed criterion, although every step of it is bitwise identical to upstream's when the kernels are deterministic (the bitwise checks below).
- **With the seeds pooled, the judge's criterion is met.** `all` (6 seeds): equivalent, -0.79σ to +0.06σ; `max` (4 seeds): equivalent, -1.61σ to +0.95σ. The three loss endpoints show no difference in either comparison. The largest values: with `all`, action head update size, steps 10-60, at -0.79σ, -1.7% of the mean (σ is 2.2%), unpaired p = 0.35; with `max`, VLM update size, steps 10-60, at -1.61σ, -3.7% of the mean (σ is 2.6%), unpaired p = 0.15; upstream's own second run reaches 1.79σ. The p-values are not corrected for the 14 comparisons made. `all` and `max` were run on the final tree (session 12: `all` with `adamw_compiled` and `compile_head`), against the same upstream runs as the other columns (session 10).
- A single seed, which is what the judge runs on the headline recipe, is not enough here: the first `all` run on seed 0 had one endpoint at -3.2σ (the loss of steps 20 to 40, lower than upstream's, whose seed 0 is the highest of its six).

### How far a rounding-level path moves one training step

`all` contains paths whose results are not bitwise identical. For each of them, one training step on real weights
(bf16, batch 16, deterministic mode, same batch and noise) was compared with the same step on the default code: the
relative difference of the loss, and the relative difference and cosine of all gradients, by module. Two runs of
the default code agree exactly (the noise floor is zero), so every number below is caused by the path.

| Change | Framework | Backbone / default attention | Loss | All gradients: relative difference | cosine | Vision tower | Language model (world model for Wan) | Action head |
|---|---|---|---:|---:|---:|---|---|---|
| `compile_lm` | QwenGR00T | Qwen2.5-VL-3B / SDPA | 7.4e-06 | 0.2 | 0.980 | 0.32 (cos 0.949) | 0.22 (cos 0.980) | 0.045 (cos 0.999) |
| `compile_lm` | QwenOFT | Qwen2.5-VL-3B / SDPA | 1.0e-02 | 0.25 | 0.969 | 0.21 (cos 0.979) | 0.28 (cos 0.959) | 0.12 (cos 0.994) |
| `compile_vision` | QwenGR00T | Qwen2.5-VL-3B / SDPA | 7.7e-05 | 1.3 | 0.370 | 2.0 (cos -0.044) | 0.16 (cos 0.986) | 0.052 (cos 0.999) |
| `compile_vision` | QwenOFT | Qwen2.5-VL-3B / SDPA | 1.3e-03 | 0.046 | 0.999 | 0.56 (cos 0.841) | 0.048 (cos 0.999) | 0.032 (cos 1.000) |
| (for comparison) SDPA → eager attention | QwenGR00T | Qwen2.5-VL-3B / SDPA | 1.4e-04 | 4.3 | 0.076 | 6.6 (cos -0.088) | 0.23 (cos 0.975) | 0.083 (cos 0.997) |
| (for comparison) SDPA → eager attention | QwenGR00T | Qwen3.5-0.8B / SDPA | 6.0e-05 | 1.0 | 0.570 | 1.5 (cos 0.203) | 0.17 (cos 0.986) | 0.078 (cos 0.997) |
| (for comparison) SDPA → eager attention | QwenOFT | Qwen2.5-VL-3B / SDPA | 4.5e-04 | 0.11 | 0.995 | 0.92 (cos 0.554) | 0.1 (cos 0.996) | 0.072 (cos 0.997) |
| `sdpa` (eager → SDPA) | Gemma4PI | gemma-4-E2B-it / eager | 8.6e-04 | 0.53 | 0.973 | 1.4 (cos 0.434) | 0.53 (cos 0.977) | 0.048 (cos 0.999) |
| `sdpa` (flash-attention → SDPA) | QwenGR00T | Qwen2.5-VL-3B / flash-attention | 5.4e-05 | 0.87 | 0.531 | 1.0 (cos 0.242) | 0.15 (cos 0.988) | 0.041 (cos 0.999) |
| `sdpa` (flash-attention → SDPA) | QwenOFT | Qwen2.5-VL-3B / flash-attention | 1.7e-03 | 0.079 | 0.997 | 1.3 (cos 0.287) | 0.051 (cos 0.999) | 0.046 (cos 0.999) |
| `vision_attn_relaxed` | QwenGR00T | Qwen2.5-VL-3B / SDPA | 0.0e+00 | 0 | 1.000 | 0 (cos 1.000) | 0 (cos 1.000) | 0 (cos 1.000) |
| `vision_attn_relaxed` | QwenOFT | Qwen2.5-VL-3B / SDPA | 0.0e+00 | 0 | 1.000 | 0 (cos 1.000) | 0 (cos 1.000) | 0 (cos 1.000) |
| `condition_once` | QwenGR00T | Qwen3-VL-4B / SDPA | 0.0e+00 | 0.017 | 1.000 | 0.023 (cos 1.000) | 0.0097 (cos 1.000) | 9.8e-07 (cos 1.000) |
| `condition_once` | QwenPI | Qwen3-VL-4B / SDPA | 0.0e+00 | 0.0042 | 1.000 | 0.0099 (cos 1.000) | 0.0045 (cos 1.000) | 1.4e-06 (cos 1.000) |
| `unused_views` | PI0 | PaliGemma (openpi) / eager | 1.0e-03 | 0.03 | 1.000 | 0.031 (cos 1.000) | 0.032 (cos 0.999) | 0.023 (cos 1.000) |
| `unused_views` | PI05 | PaliGemma (openpi) / eager | 9.9e-04 | 0.025 | 1.000 | 0.04 (cos 0.999) | 0.023 (cos 1.000) | 0.021 (cos 1.000) |
| `openpi_compile` | PI0 | PaliGemma (openpi) / eager | 2.4e-03 | 0.08 | 0.997 | 0.12 (cos 0.993) | 0.066 (cos 0.998) | 0.051 (cos 0.999) |
| `openpi_compile` | PI05 | PaliGemma (openpi) / eager | 1.2e-03 | 0.068 | 0.998 | 0.13 (cos 0.992) | 0.059 (cos 0.998) | 0.049 (cos 0.999) |
| `openpi_sdpa` | PI0 | PaliGemma (openpi) / eager | 1.1e-04 | 0.072 | 0.997 | 0.12 (cos 0.993) | 0.053 (cos 0.999) | 0.022 (cos 1.000) |
| `openpi_sdpa` | PI05 | PaliGemma (openpi) / eager | 2.3e-04 | 0.059 | 0.998 | 0.14 (cos 0.990) | 0.041 (cos 0.999) | 0.034 (cos 1.000) |
| `sdpa,compile_lm,compile_vision` | Gemma4PI | gemma-4-E2B-it / eager | 1.3e-03 | 0.3 | 0.979 | 1.0 (cos 0.538) | 0.3 (cos 0.980) | 0.053 (cos 0.999) |
| `condition_once,compile_head` | Gemma4PI | gemma-4-E2B-it / eager | 1.2e-07 | 0.011 | 1.000 | 0.028 (cos 1.000) | 0.011 (cos 1.000) | 7.6e-07 (cos 1.000) |
| `compile_lm,compile_vision` | MiniCPMRobotManip | MiniCPM-V 4.6 (RobotManip checkpoint) / eager | 2.8e-04 | 0.0079 | 1.000 | 0.013 (cos 1.000) | 0.0078 (cos 1.000) | 0.004 (cos 1.000) |
| `hidden_only,vision_pos` | MiniCPMRobotManip | MiniCPM-V 4.6 (RobotManip checkpoint) / eager | 0.0e+00 | 0 | 1.000 | 0 (cos 1.000) | 0 (cos 1.000) | 0 (cos 1.000) |
| `openpi_one_pass` | PI0 | PaliGemma (openpi) / eager | 0.0e+00 | 0.0015 | 1.000 | 0.0028 (cos 1.000) | 0.00083 (cos 1.000) | 0 (cos 1.000) |
| `openpi_one_pass` | PI05 | PaliGemma (openpi) / eager | 0.0e+00 | 0.00099 | 1.000 | 0.0028 (cos 1.000) | 0.00049 (cos 1.000) | 0 (cos 1.000) |
| `compile_head` | QwenAdapter | Qwen3-VL-4B / SDPA | 7.0e-04 | 0.091 | 0.997 | 0.13 (cos 0.998) | 0.061 (cos 0.998) | 0.016 (cos 1.000) |
| `compile_head` | QwenDiscreteDiffusion | Qwen3-VL-4B / SDPA | 1.9e-06 | 0.066 | 0.998 | 0.042 (cos 0.999) | 0.065 (cos 0.998) | 0.073 (cos 0.997) |
| `compile_head` | QwenGR00T | Qwen3-VL-4B / SDPA | 1.8e-07 | 0.018 | 1.000 | 0.026 (cos 1.000) | 0.0092 (cos 1.000) | 0.0039 (cos 1.000) |
| `compile_lm,compile_vision` | QwenGR00T (backbone loss) | Florence-2-large / eager | 1.9e-02 | 0.73 | 0.770 | 1.1 (cos 0.554) | 0.09 (cos 0.996) | — |
| `compile_lm,compile_vision` | QwenGR00T | Qwen3.5-0.8B / SDPA | 1.0e-04 | 0.83 | 0.634 | 1.0 (cos 0.411) | 0.2 (cos 0.981) | 0.099 (cos 0.995) |
| `hidden_only,vision_attn,vision_pos` | QwenGR00T | Qwen3.5-0.8B / SDPA | 0.0e+00 | 0 | 1.000 | 0 (cos 1.000) | 0 (cos 1.000) | 0 (cos 1.000) |
| `compile_head` | QwenPI | Qwen3-VL-4B / SDPA | 4.7e-06 | 0.034 | 0.999 | 0.031 (cos 1.000) | 0.032 (cos 1.000) | 0.034 (cos 0.999) |
| `compile_world_model` | WanGR00T | Wan2.2-TI2V-5B-Diffusers / SDPA | 1.0e-06 | 0.018 | 1.000 | — | 0.035 (cos 0.999) | 0.014 (cos 1.000) |
| `condition_once,compile_head` | WanGR00T | Wan2.2-TI2V-5B-Diffusers / SDPA | 0.0e+00 | 0.0013 | 1.000 | — | 0.003 (cos 1.000) | 0.00075 (cos 1.000) |

Qwen3.5's `compile_lm` row was measured before `compile_lm` stopped applying to language models with linear-attention layers; at the `all` level only its vision tower is compiled now.

(backbone loss): upstream's training step of that recipe fails, so the step is the backbone's language-model loss and no action head is measured.

- **One bf16 training step is far more sensitive to rounding than the loss is.** Compiling the decoder layers, which the judge accepts on the headline recipe, moves the language model's gradient by 28% (QwenOFT) and 22% (QwenGR00T), and the loss of that step by 1% and 0.0007%.
- **The vision tower's one-step gradient is mostly rounding noise.** It is what remains after backpropagation through the whole language model and 32 vision layers in bf16. Compiling the vision blocks leaves it with cosine 0.84 (QwenOFT) and -0.04 (QwenGR00T) to the default one; switching the same backbone from SDPA to eager attention, a change that is not ours and computes the same function, gives 0.55 and -0.09.
- **`sdpa` is in the same range**: language model 5.1% and 15%, vision tower cosine 0.29 and 0.24. The low cosine of all gradients together on QwenGR00T is the vision tower, whose gradient is the largest part of the total there. The judge is frozen on the headline recipe and cannot be run on the flash-attention recipes; the mean loss of their timed runs (steps 100 to 220) was QwenOFT, session 6: 0.2567 (baseline), 0.2569 (`exact`, bitwise identical to the baseline, so the difference is which batches a run draws), 0.2571 (`all`); QwenGR00T, session 6: 0.7987 (baseline), 0.7753 (`exact`, bitwise identical to the baseline, so the difference is which batches a run draws), 0.8057 (`all`); QwenGR00T, session 8: 0.8066 (baseline), 0.8082 (`exact`, bitwise identical to the baseline, so the difference is which batches a run draws), 0.8021 (`all`); QwenGR00T, session 12: 0.8021 (baseline), 0.8080 (`exact`, bitwise identical to the baseline, so the difference is which batches a run draws), 0.8046 (`all`).
- **`condition_once` does not change the loss at all** and moves gradients by 0.5% to 2.3%: only the order in which the copies' gradients are summed differs.
- **`vision_attn_relaxed` is bitwise identical in this check**, because deterministic mode uses the unfused SDPA kernel, for which one batched call equals the per-image calls. The last-bit difference it is named for appears with the fused kernels that training uses (see the probe below).
- **The PI0 / PI05 paths**: `openpi_compile` moves the gradients by 8.0% and 6.8%, `openpi_sdpa` by 7.2% and 5.9%, `unused_views` by 3.0% and 2.5%; these checks load the converted base checkpoints, so every module has trained weights. With `openpi_compile`, five parameters get a gradient that they do not get in the default step, and it is exactly zero: the MLP, second norm and attention output projection of the last layer of the prefix stream, whose output the loss does not use. Compiled code returns zeros there and eager autograd returns nothing; DeepSpeed sets missing gradients to zero itself, so training is unchanged.
- The action head is randomly initialised in every check (only the backbone has pretrained weights, except for PI0 / PI05), so absolute values differ between rows; each row compares two steps of the same process.

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
| `fw_adapter_q3` | QwenAdapter | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 1,226 gradients, actions | identical |
| `fw_dd_q3` | QwenDiscreteDiffusion | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 1,220 gradients, actions | identical |
| `fw_dual_q3` | QwenDual | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 1,136 gradients, actions | identical except 1 gradient that upstream does not reproduce run to run (`dino_encoder.body.pos_embed`) |
| `fw_gr00t_q25` | QwenGR00T | Qwen2.5-VL-3B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 1,072 gradients, actions | identical |
| `fw_gr00t_q25_fa2` | QwenGR00T | Qwen2.5-VL-3B | flash-attention | bf16 | hidden_only, vision_pos, worker_prep | attn_mask, vision_attn | — | loss, 1,072 gradients, actions | identical |
| `fw_gr00t_q3` | QwenGR00T | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 960 gradients, actions | identical |
| `fw_langforce_q3` | LangForce | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 961 gradients, actions | identical |
| `fw_oft_q25` | QwenOFT | Qwen2.5-VL-3B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 840 gradients, actions | identical |
| `fw_oft_q25_fa2` | QwenOFT | Qwen2.5-VL-3B | flash-attention | bf16 | hidden_only, vision_pos, worker_prep | attn_mask, vision_attn | — | loss, 840 gradients, actions | identical |
| `fw_oft_q3` | QwenOFT | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 728 gradients, actions | identical |
| `fw_pi_q25_fa2` | QwenPI | Qwen2.5-VL-3B | flash-attention | bf16 | hidden_only, vision_pos, worker_prep | attn_mask, vision_attn | — | loss, 1,335 gradients, actions | identical |
| `fw_pi_q3` | QwenPI | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 1,225 gradients, actions | identical |
| `fw_piv3_q3` | QwenPI_v3 | Qwen3-VL-4B | SDPA | bf16 | attn_mask, hidden_only, vision_attn, vision_pos, worker_prep | — | — | loss, 1,376 gradients, actions | identical |

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

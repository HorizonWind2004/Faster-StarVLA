# WORK_LOG: starVLA QwenOFT LIBERO training on 2 x B300

Reference 906.9 ms/step (frozen). Target 2.0x, minimum 1.3x. All timings below are `bench.py time` unless noted.

## Round 0: profile of the reference (formulation profile, see task/baseline.md)

Per step, main thread: processor 190 ms (CPU, single thread, GPU idle), attention-mask vmap 154 ms (host), vision
tower 114 ms, DeepSpeed grad norm 89 ms, synchronize waits 117 ms, data hand-over 30 ms. GPU utilization 53%.
Classification: launch/host class first (three host stalls with the GPU idle), then DeepSpeed's optimizer path.
The data stage is not a bottleneck (4 workers, 58 ms per sample on the CPU, `data_times` ~1 ms).

## Round 1: host stalls in forward (prefetch thread, LM head, attention mask) -> 1.18x

- STARVLA_PREP_AHEAD (prompt + processor in a background thread, one batch ahead), STARVLA_SKIP_LM_HEAD (backbone
  without the unused vocabulary logits; first version took the backbone's own hidden_states[-1], which is the normed
  state and changed the loss on a CPU check: fixed to the last decoder layer's output, bitwise), STARVLA_FAST_ATTN_MASK
  (causal + padding mask with two tensor ops instead of transformers' nested vmaps; 52 cases identical).
- time: 766.6 ms, 1.183x. Much less than the 190 + 154 ms the profile suggested.
- Added `profile --stages` (sync at the DeepSpeed forward / backward / step edges). Reference: forward 415, backward
  319, optimizer step 179, loop 7 ms. Candidate: forward 242, backward 337, optimizer step 169. The prefetch thread
  shares the GIL with the launch-bound forward, and the mask's host time overlapped with queued vision-tower work.

## Round 2: processor in the data-loader workers; DeepSpeed reduce-to-owner -> 1.34x

- STARVLA_PREP_IN_WORKERS replaces the thread: the collate_fn in the workers runs the same prompt construction and
  processor and attaches the tensors to the batch (batch[0]["_prepared"]). CPU check: tensors, loss and all gradients
  bitwise.
- time (PREP_IN_WORKERS + SKIP_LM_HEAD + FAST_ATTN_MASK): 677.8 ms, 1.338x.
- STARVLA_DS_REDUCE_TO_OWNER (DeepSpeed use_multi_rank_bucket_allreduce = false): 686.1 ms, no gain. Left off.
- Stages: forward 204, backward 316, optimizer step 170. Blocking profile of forward: vision tower 148 ms, of which
  the 24 blocks 111 ms; scaled_dot_product_attention is called 804 times per step.

## Round 3: vision tower attention -> 1.54x

- With SDPA, transformers' vision attention reads the chunk lengths back to the host in every layer and runs one
  attention call per image (24 layers x 32 images). STARVLA_BATCHED_VISION_ATTN: one batched call per layer when all
  images have the same number of patches. Real vision tower on B300: outputs and all 315 gradients bitwise, forward +
  backward 283.9 -> 167.3 ms.
- time: 588.8 ms, 1.540x. Stages: forward 155, backward 244, optimizer step 164.

## Round 4: vision tower position indices -> 1.67x

- Vision forward (sync per part): 53.7 ms, of which fast_pos_embed_interpolate 17.4 and rot_pos_emb 10.1 are per-image
  host loops over values that depend only on grid_thw. STARVLA_VISION_POS_CACHE memoizes them per grid; the lookup in
  the trainable position embedding still runs every step. Bitwise (outputs, 315 gradients); 167.5 -> 126.1 ms.
- time: 541.5 ms, 1.675x.
- Also moved Qwen3VLModel.get_rope_index (15 ms host loop per step) into the workers' collate (same function, passed
  as position_ids). CPU check bitwise.

## Round 5: DeepSpeed clipping norm -> 1.71x

- DeepSpeed's get_grad_norm_direct casts every gradient of the rank's partition to fp64 before the norm (18 GB written
  and read per step). vector_norm(g, dtype=float64) gives the same bits (33 shapes on B300). STARVLA_DS_FUSED_GRAD_NORM
  installs the same loop with the fused cast on the ZeRO optimizer.
- time: 528.7 ms, 1.715x (529.0 before: within noise; stages: optimizer step 164 -> 161 ms). Kept: bitwise, and it
  removes 18 GB of transient fp64 allocations.

## Round 6: decoder layers compiled -> 1.84x (not bitwise)

- Isolated LM at 16 x 205 tokens: forward 47 ms (host = wall: launch-bound), forward + backward 238 ms. Blocking op
  profile: aten::mm 72 ms in 756 calls, aten::mul 68 ms in 1,843 calls, copy_ 34 ms in 1,163 calls: about 10,000
  small ops per step, a third of the time in matrix products.
- STARVLA_COMPILE_LM: torch.compile of each decoder layer's forward (dynamic shapes, gradient-enabled calls only so
  the periodic evaluation never compiles). Isolated: 242.7 -> 160.1 ms, 36 graphs in 88 s, no recompilation for other
  sequence lengths. Rounding differs: compiled vs eager 6.5e-4 in fp32 on random tokens; in bf16 the difference is
  the size of bf16's own distance to fp32.
- time: 493.0 ms, 1.839x; peak memory 66.2 -> 59.9 GiB. check --n1: PASS (N1), endpoints within 1.74 sigma, the
  learning-rate endpoints at +0.27 / -0.34 sigma.

## Round 7: DeepSpeed plain all-reduce -> 1.92x

- Stages at 493 ms: forward 89, backward 218, optimizer step 161. With reduce_scatter DeepSpeed slices every bucket
  per parameter and owner, concatenates the slices (a 9 GB copy per step), all-reduces the copy and copies the owned
  part back. STARVLA_DS_PLAIN_ALLREDUCE (reduce_scatter = false) all-reduces the bucket in place; on two ranks
  (a + b) * 0.5 equals a * 0.5 + b * 0.5 in bf16.
- time: 472.4 ms, 1.920x.

## Round 8: vision blocks compiled, garbage-collector pauses -> 2.07x

- STARVLA_COMPILE_VISION (24 blocks; the batched attention now takes the chunk layout from grid_thw once per forward,
  no host reads per layer): isolated vision forward + backward 126.9 -> 110.4 ms. time: 463.4 ms, but the two windows
  differ by 6.6% (478 / 449).
- Per-step times of every compiled configuration show 1-2.5 s single steps at fixed step numbers (same steps in two
  runs of one configuration), none in uncompiled runs; TORCH_LOGS=recompiles logs nothing. Full garbage collections
  walking the objects compilation leaves behind. gc.freeze() four training forwards in: outliers gone (only the two
  evaluation steps remain, +0.25-0.45 s each as in the reference).
- time: 437.3 ms, 2.074x (windows 443.7 / 430.8).


## Round 9: final check, validation run, LIBERO

- Final check (reference -> candidate -> reference, fresh processes, cold caches), `switches/final.json`:
  905.0 / 905.8 ms -> 453.3 ms, **1.9975x**, PASS, tier N1. Gate 2 not bitwise (3,227 of 23,336 tensors identical,
  no exact item fails); gate 3: all seven endpoints within 1 sigma. Peak memory 66.5 -> 59.0 GiB.
  The target in GOAL.md was 2.0x: the final check is 0.0025 below it. Single timing runs of the same switch set
  measured 437-453 ms (2.00-2.07x).
- A first final check measured 2.0010x and stopped at gate 3 with a judge ERROR: the adapter looked for the dataset in
  gc.get_objects(), which gc.freeze() empties. Judge change 1 (task/judge_changes.md): the adapter records the dataset
  when the DataLoader is built. Regression: the reference trajectory re-recorded with the changed adapter equals the
  frozen one in all 640 values; the lr x 1.02 control arm is rejected with the same numbers.
- Bit-exact switch set (`switches/bitwise.json`, the seven non-compile switches), full check against the frozen
  reference timing: 521.9 ms, 1.738x, gate 2 BITWISE (23,336 of 23,336 tensors), PASS, tier N0.
- Validation run with all switches: 30,000 steps, seed 42, 2 x B300, global batch 32, 3 h 51 min of wall time
  (462 ms per step including the periodic evaluations and three checkpoint saves). The epoch boundary near step 12,684
  passed without recompilation or a stall.
- LIBERO, 500 episodes per suite (spatial / object / goal / long / average), batching server, 28 clients on 2 GPUs:
  this run at 30,000 steps 98.8 / 98.6 / 98.4 / 93.4 / 97.30; the released checkpoint (step 50,000, trained by the
  authors on 8 GPUs, global batch 128) in the same pipeline 98.6 / 100.0 / 99.4 / 96.6 / 98.65; upstream's published
  30K numbers for this recipe 97.8 / 98.6 / 96.2 / 93.8 / 96.6. Each 2,000-episode evaluation took 17-18 min of
  simulation.

## Round 10: flash-attention 2 (not kept)

- Upstream hard-codes `attn_implementation = "sdpa"`. One timing run with a temporary switch that loads the model with
  `flash_attention_2` (flash-attn 2.8.3.post1) plus the bit-exact switches that still apply (inputs in workers, no LM
  head, vision position cache, both DeepSpeed switches): 630.1 ms, 1.44x, peak 65.7 GiB, no NaN, the same mean loss
  over steps 100-220 (0.254). The SDPA path with the batched vision attention and the direct mask is at 521.9 ms.
  transformers' flash path unpads every sequence and calls the variable-length kernel; at 16 x 205 tokens attention
  is not where the time is.
- With the compiled layers it does not run: Dynamo cannot trace `flash_attn._flash_attn_varlen_forward`
  (`Expected a value of type 'int' for argument 'max_seqlen_q' but instead found type 'FakeTensor'`).
- Dropped; the switch is not in the tree. Run files: runs/fa2, runs/fa2_compile.

## Round 11: one setting for every recipe, and what it does elsewhere

- The nine per-path variables became one package, `starVLA/fast/`, behind `STARVLA_FAST=off|exact|all|name,-name`.
  Each fast path is a module with `apply(target) -> bool` that inspects the model it is given; the callers are
  `get_vlm_model`, the three trainers and `build_dataloader`. The backbone interfaces carry no switch any more. The
  old variables still select the same paths. CPU tests on tiny random-weight Qwen3-VL and Qwen2.5-VL models
  (`tests/test_fast_paths.py`): selection syntax, 62 / 68 gradients, hidden states and logits bitwise identical.
- Real weights, bf16, B300, baseline against `exact` (runs/unified/bitwise4): 10 checks, all identical. Backbone only
  on Qwen3-VL-4B (712 gradients) and Qwen2.5-VL-3B (824); one training step and one `predict_action` call of QwenOFT,
  QwenGR00T and QwenPI on Qwen3-VL (728 / 960 / 1,225 gradients), QwenOFT and QwenGR00T on Qwen2.5-VL with SDPA
  (840 / 1,072) and with flash-attention (840 / 1,072).
- Found on the way: one batched attention call on [32, 16, 256, d] is bitwise identical to 32 single calls at head
  dimension 64 but not at 72 or 80 in bf16 on B300 (64-token windows and fp32 are identical at all three). The first
  version of `vision_attn` would have changed Qwen2.5-VL's gradients in the last bit. It now probes each shape, dtype
  and device once, forward and backward, and batches only what agrees; `vision_attn_relaxed` (level `all`) batches the
  rest.
- Found on the way: upstream's QwenGR00T and QwenPI gradients differ run to run (PyTorch's fused SDPA backward in the
  action heads). Their checks were made with the flash and memory-efficient SDPA kernels disabled on both sides.
- Found on the way: `STARVLA_FAST=all` on a backbone loaded with flash-attention (the default of the Qwen2.5-VL
  recipes) failed in the first training step, in the compiled vision blocks (the round 10 Dynamo error). `attn_mask`,
  `vision_attn(_relaxed)` and both compiled paths now return "not applicable" there. Both flash-attention recipes then
  ran with `all` (747.2 / 849.3 ms, session 3).
- Step time on other combinations, one run per cell, each against a baseline from the same job (session 2):
  QwenGR00T + Qwen3-VL 903.0 -> 783.9 (`exact`, 1.15x) -> 722.9 (`all`, 1.25x); QwenPI + Qwen3-VL 1291.9 -> 1177.4
  (1.10x) -> 1133.3 (1.14x); QwenOFT + Qwen2.5-VL with SDPA 1021.7 -> 694.1 (1.47x) -> 554.4 (1.84x); with
  flash-attention 775.0 -> 738.7 (1.05x); QwenGR00T + Qwen2.5-VL with flash-attention 857.0 -> 830.0 (1.03x).
  `worker_prep` has a builder only for QwenOFT + Qwen3-VL, and no path touches the action heads.
- The judge on the refactored code. `exact`: 23,336 of 23,336 tensors bitwise identical, PASS N0, in both sessions.
  `all`, three-process check: INVALID in both sessions, because the baseline itself ran outside the frozen band:

  | Session | Node | Baseline | `exact` | `all` |
  |---|---|---:|---:|---:|
  | 1 (rounds 0-9; the timing was frozen here) | A | 905.4 | 521.9 (1.74x) | 453.3 (2.00x) |
  | 2 | B | 795.5 | 510.1 (1.56x) | 425.3 (1.87x) |
  | 3 | A, after a reboot | 815.9 | 501.9 (1.63x) | 421.5 (1.94x) |

  Same reference code, environment, allocation (2 GPUs, 32 cores) and command line; the whole step-time distribution
  of the baseline moved (median 896 -> 777-819 ms). Why node A was slower in session 1 was not determined. The agent-
  style speedups the judge printed in sessions 2 and 3 (1.78x, 1.81x for `exact`) divide by the frozen timing of
  session 1 and overstate; the table uses each session's own baseline. The judge and its frozen timing were not
  changed.
- LIBERO through the unified code (released checkpoint, policy servers with `STARVLA_FAST=exact`, 2,000 episodes):
  97.85% (98.0 / 99.8 / 99.0 / 94.6); the first release's inference variables gave 98.25% in the same harness.

## Round 12: every framework, not one

Round 11 put the fast paths behind one setting, but `worker_prep` still had a builder for one framework and one
backbone, and nothing touched an action head. This round is about the other recipes. Sessions 4 to 8 are five batch
jobs on one node (2 GPUs, 32 cores); every speedup is against a baseline measured in the same job.

- `worker_prep` no longer contains framework code. A worker runs the backbone interface's own `build_qwenvl_inputs` on
  a stand-in that has the processor and the config but no model; `apply` checks once that the stand-in returns what
  the interface returns. In the training process the interface returns a worker's result only for a call with the
  images of that batch (the same objects, or copies with the same pixels) and the very prompts the worker used. A
  framework that edits the prompt registers the edit in one function (`@prompt`: QwenOFT, QwenAdapter, QwenPI_v3,
  LangForce, which makes two calls per step); a wrong or missing registration costs the speedup, not correctness.
  CPU test through whole frameworks (`tests/test_fast_frameworks.py`, small random backbone behind the real
  processor): QwenOFT, QwenGR00T, QwenPI, QwenPI_v3, QwenAdapter, QwenDiscreteDiffusion, LangForce, QwenDual and
  CosmosGR00T on Qwen3-VL, the first eight on Qwen2.5-VL: every backbone input comes from the workers, loss and all
  gradients identical.
- Real weights, bf16, B300, training batch from a DataLoader with workers, `exact` against the default code (sessions
  4 and 6): QwenOFT / QwenGR00T / QwenPI on Qwen3-VL and on Qwen2.5-VL with SDPA and with flash-attention, QwenPI_v3
  (1,376 gradients), QwenAdapter (1,226), QwenDiscreteDiffusion (1,220), LangForce (961): all identical. QwenDual:
  1,135 of 1,136 identical; `dino_encoder.body.pos_embed` differs between two runs of the default code itself.
  Judge on that tree, `exact`: 23,336 of 23,336 tensors, PASS N0.
- `sdpa` (level `all`): a backbone loaded with flash-attention (the Qwen2.5-VL recipes) is switched to PyTorch SDPA
  first, after which `attn_mask`, the batched vision attention and the compiled layers apply. Before this, `all`
  equalled `exact` on those recipes. Session 6: QwenOFT + Qwen2.5-VL 877.6 -> 645.6 (`exact`, 1.36x) -> 466.5 ms
  (`all`, 1.88x; peak memory 63.2 -> 55.3 GiB); QwenGR00T 960.9 -> 754.5 (1.27x) -> 569.1 ms (1.69x).
- `condition_once` (level `all`): the flow-matching frameworks repeat the VLM's hidden states for several noise levels
  per sample (8 for the GR00T head, 2 in QwenPI) and every cross-attention layer of the DiT projects all copies to
  keys and values in fp32. The copies are detected in each call (two tensor comparisons), the projections run on the
  first copy and the result is repeated. Session 7: QwenGR00T + Qwen3-VL 905.0 -> 603.5 (`exact`, 1.50x) -> 498.2 ms
  (`all`, 1.82x; it was 1.70x in sessions 4 and 5 without the path); QwenPI 1291.5 -> 1110.2 (1.16x) -> 976.2 ms
  (1.32x). One-step distance: the loss is unchanged, gradients move by 0.2% (QwenPI) and 1.2% (QwenGR00T).
- `tf32` (by name only): TF32 for fp32 matrix products. What is left of a DiT-head framework after `all` is mostly the
  head's fp32 projections of VLM hidden states (profile of QwenPI with `all`: 0.2 s of the 0.39 s forward is
  `F.linear` in the head). Session 6: QwenPI_v3 1328.7 -> 620.9 ms, QwenDual 852.2 -> 605.4; session 7: QwenPI 976.2 ->
  664.3; no gain on QwenGR00T once `condition_once` is on (498.2 / 500.4) or on LangForce (719.2 / 720.9). It lowers
  precision, so it is not part of `all`. Judge on the headline recipe with `all,tf32`: PASS, tier N2, 446.4 ms (2.03x
  against the frozen timing), gate 3 within 1.41 sigma. Mean loss of steps 100-220 shows no difference from the
  baseline in any framework.
- The other frameworks on Qwen3-VL, session 6 (before `condition_once`): QwenPI_v3 1627.3 -> 1389.9 (1.17x) -> 1328.7
  (1.22x) -> 620.9 ms with TF32 (2.62x); QwenAdapter 999.2 -> 767.8 (1.30x) -> 668.4 (1.49x); LangForce 1414.9 -> 894.8
  (1.58x) -> 719.2 (1.97x); QwenDual 1209.1 -> 914.6 (1.32x) -> 852.2 (1.42x) -> 605.4 with TF32 (2.00x).
  QwenDiscreteDiffusion failed in all four runs including the baseline: my generated config asked for 16 action steps
  and the LIBERO data gives 8 (`make_configs.py` fixed; measured in session 8).
- PI0 and PI05 (openpi recipe on LIBERO, converted base checkpoints) do not use a `get_vlm_model` backbone, so a
  fourth hook was added (`build_framework -> fast.accelerate_framework`). `unused_views`: LIBERO has two of the three
  camera views these models take; the third is an all-black image that goes through SigLIP and stays in the sequence
  as 256 masked tokens. A view that no sample of the batch has is left out (prefix 816 -> 560 tokens). Session 7:
  PI0 636.5 -> 626.9 (`exact`: the two DeepSpeed paths, 1.02x) -> 500.7 ms (`all`, 1.27x; 102.5 -> 77.0 GiB); PI05 668.0
  -> 662.1 -> 532.5 ms (1.25x). The profile of the baseline then gave two more paths: `worker_examples` (exact; the
  framework's `_prepare_examples`, 33 ms per step in the training process, runs in the workers on a stand-in) and
  `openpi_sdpa` (level `all`; fused attention instead of matmul, fp32 softmax, matmul in SigLIP and the joint layers).
- How far does a rounding-level path move one training step (real weights, bf16, deterministic mode, same batch and
  noise; two runs of the default code agree exactly; session 8)? Compiling the decoder layers, which the judge
  accepts, moves the language model's gradient by 22% (QwenGR00T) and 28% (QwenOFT), the loss by 0.001% and 1%. The
  vision tower's
  one-step gradient is mostly rounding noise: compiling the vision blocks leaves it with cosine 0.84 (QwenOFT) and
  -0.04 (QwenGR00T) to the default one, and switching the backbone from SDPA to eager attention, which is not our
  change and computes the same function, with 0.55 and -0.09. Flash-attention -> SDPA is in the same range (language
  model 5% and 16%, vision tower cosine 0.29 and 0.24). `openpi_sdpa` moves PI0's gradients by 7% (cosine 0.998),
  `unused_views` by 3%. So a single step's gradient cannot tell a rounding-level
  change from a wrong one at this precision; the loss, the 60-step statistical gate and the 220-step loss curves can.
- Found on the way: the same baseline moved by 10% between jobs on the same node (QwenGR00T 1000.8, 998.5, 905.0 ms in
  sessions 4, 5, 7), and the headline recipe with `all` ran at 424.5 ms in session 4 and 464.3 ms in session 6 with
  the same paths applied. Only numbers of one job are compared anywhere.
- Found on the way: my first distance script concatenated all gradients on the GPU and cast them to fp64, which does
  not fit for 3.9 billion elements; it now accumulates the dot products in chunks. It also ran the step without the
  trainer's bf16 autocast, which PI05 needs. The session 8 numbers are from the corrected script.
- Session 8, final tree of that morning (`e5f7e15`), everything in one job. Judge: `exact` 23,336 of 23,336 tensors,
  PASS N0 (490.5 ms); the three-process check of `all` INVALID again (its own references 806 / 816 ms, outside the
  frozen band; candidate 413.5 ms, 1.96x). Headline recipe, this tree and the session 4 tree alternating: 420.8 /
  414.6 ms against 426.2 / 427.1 ms, baseline 801.8 ms before and 807.0 ms after: the 464 ms of session 6 was the
  job, not the code. With `condition_once`: QwenPI_v3 1679.6 -> 1430.6 -> 1012.3 ms (`all`, 1.66x; 1.22x before) ->
  702.3 with TF32; QwenDual 1308.0 -> 966.7 -> 680.6 (1.92x; 1.42x before) -> 665.0; LangForce 1577.5 -> 955.9 -> 780.8
  (2.02x); QwenGR00T + Qwen2.5-VL flash-attention 956.3 -> 755.6 -> 554.7 (1.72x). QwenDiscreteDiffusion, first valid
  measurement: 2507.8 -> 2295.9 (1.09x) -> 2062.1 (1.22x) -> 739.4 ms with TF32 (3.39x).
- Found on the way: `worker_examples` passed its CPU test and then reported "not applicable" in training. The trainer
  wraps the config in an access-tracking object, and the stand-in for the workers kept only plain OmegaConf values.
  It now unwraps that object; the test builds PI0 / PI05 on a wrapped config, as the trainer does. Session 9: PI0
  636.9 -> 591.8 ms with `exact` (1.08x; 0.99x in session 8).
- Levels follow the judge's tiers: `exact` is N0, `all` adds N1 (another order or kernel, compilation), and a third
  level `max` adds N2, a change of precision: `tf32`, and `openpi_sdpa`, whose softmax runs in bf16 instead of fp32
  and which had been in `all` for one morning. On the Qwen frameworks `max` is what was measured as `all,tf32`.
- `openpi_compile` (level `all`): the SigLIP layers and the framework's joint layer function compiled for training
  calls. Dynamo traces each as one graph without breaks. Session 9: PI0 `all` 453.6 ms (1.40x; 1.27x in session 7
  with `unused_views` alone); the first step takes 7 min to compile. The first distance check of it failed in my
  script: compiled code returns zeros for parameters the loss does not use (the prefix stream's last MLP), eager
  autograd returns None, and the script compared concatenated vectors. DeepSpeed sets None gradients to zero
  itself, so training is the same; the script now compares by parameter name.
- Session 9, PI0 and PI05 on the tree with `worker_examples` fixed and `openpi_compile`, `all` and `max` alternating
  in one job: PI0 636.9 -> 591.8 (`exact`, 1.08x) -> 453.6 / 452.2 (`all`, 1.40x) -> 443.3 / 439.8 ms (`max`, 1.44x);
  PI05 666.5 -> 626.9 (1.06x) -> 468.6 / 475.0 (1.40-1.42x) -> 465.1 / 463.8 ms (1.43-1.44x). Peak memory of PI0: 102.5
  GiB upstream, 63.0 with `all`, 57.2 with `max`. `openpi_sdpa` is worth 1.5-2.5% and 9% of the memory; the "3.5%
  slower on PI05" of session 8 was a comparison across jobs.
- Session 10, distance of `openpi_compile` with the corrected script: gradients move by 8.0% (PI0) and 6.8% (PI05),
  cosine 0.997 / 0.998. Five parameters have a gradient only in the compiled step, all exactly zero: the MLP, the
  second norm and the attention output projection of the prefix stream's last layer, which the loss does not use.
- Session 10, the judge on the final commit (`8427140`): `all` against the frozen timing PASS, tier N1 (3,227 of
  23,336 tensors bitwise, the statistical gate equivalent; 426.4 ms); `exact` PASS, tier N0 (23,336 of 23,336; 498.0
  ms). No baseline was timed in this job.
- The judge's statistical gate on a framework with an fp32 head (`coverage/stat_gate.py`: the judge's recorder,
  trajectory case and endpoints with `--framework.name QwenPI`; sessions 10 to 12). Upstream on six seeds for the
  spread. `all` on seed 0 alone: "different", one endpoint of seven at -3.2 sigma (loss of steps 20-40: 4.5 against
  upstream's 17.2 on that seed, the highest of its six). Then `exact` on seed 0: also "different" (-3.3 sigma on the
  same endpoint), although its first two steps are bitwise identical to upstream's on both ranks; by step 4 the loss
  is 24,398 against 4,144. Control, upstream a second time on seeds 0 and 1: single differences from -2.8 to +2.1
  sigma. Upstream QwenPI does not reproduce itself on a seed (non-deterministic backward of the fused SDPA kernels in
  the head, and losses in the thousands in the first steps), so one same-seed comparison says nothing here.
  With the seeds pooled, the judge's criterion (mean same-seed difference within 3 sigma on every endpoint) holds:
  `all` on six seeds -1.31 to +1.69 sigma, `max` on four seeds -0.73 to +2.80 sigma. The largest is the action head's
  update size over steps 10-60 (+3.7% with `all`, +5.2% with `max`; upstream's own second run: +1.79 sigma). The loss
  endpoints show nothing.
- Found on the way: my first version of that script stopped at the first "different" verdict (the comparison tool
  exits non-zero), so `max` did not run in session 10.

## Round 13: the table, cleared (24 Nov, evening)

Code first, then one probe job (session 11, job 73035), then one verification job on the final tree (session 12).

- Probes, `STARVLA_FAST=all`, stages synchronized at their edges (forward / backward / optimizer step, ms): QwenOFT
  71 / 152 / 154; QwenGR00T 106 / 216 / 168; QwenAdapter 123 / 288 / 218; PI0 91 / 161 / 135. The optimizer step is
  28-40% of a step in every framework.
- Optimizer microbenchmark on one B300, a flat fp32 tensor of 2.25 billion elements (one rank's partition of the
  headline recipe): torch's fused AdamW 104.9 ms, foreach 33.3 ms, the same update written out and compiled with
  Inductor 9.4 ms (one pass over the four tensors: 3.9 ms per two-in-one-out pass). `adamw_compiled` (level `all`)
  replaces the fused kernel for plain decoupled AdamW on fp32 CUDA tensors; torch's optimizer keeps its state. In
  training, headline recipe, alternating in one job: `all` 422.6 / 410.8 ms, `all` + `adamw_compiled` 299.9 /
  310.3 ms, baseline 786.3 ms.
- `openpi_one_pass` (level `all`): PI0 / PI05 camera views through SigLIP in one call. PI0 378.8 / 378.6 ms against
  448.7 / 447.1 without (baseline 649.7). One-step distance: gradients 0.15% (PI0) and 0.10% (PI05), loss unchanged.
- `compile_head` (level `all`): the DiT head blocks and the adapter head's residual blocks compiled. QwenAdapter 623.0 /
  626.6 ms against 688.1 / 681.6; QwenPI 963.7 against 977.9; QwenGR00T no difference. Two problems found by the
  probes, both fixed before verification: (1) the GR00T DiT uses dropout 0.2, and compiled dropout drew its masks from
  Inductor's own random stream (one-step gradients with cosine about 0 against the default step); the head is now
  compiled with `fallback_random`, which keeps eager's kernels and generator (CPU test: masks, gradients and the
  generator state identical over several shapes). (2) Three runs with the head compiled failed in the first step in
  Inductor's memory-coalescing analysis (a sympy assertion; sympy decides is_constant at random sample points, so it
  is intermittent); Inductor's config notes that the analysis does not support dynamic shapes. Both settings are
  passed as torch.compile options of the head blocks only, so the backbone's compiled layers are unchanged.
- `grad_cast` (cast while flattening the partition's gradients) and `unit_clip` (no multiply by a clipping factor of
  exactly 1): bitwise identical, but `exact` with and without them 496.1 / 491.8 against 491.4 / 498.6 ms, and
  `unit_clip` reads the factor on the host, a synchronization in every step. Both are selectable by name only.
- Other backbones (not run): `attn_mask` checks its mask builder against the installed transformers before
  registering it; `hidden_only` and every compiled layer fall back to the default forward when their first call
  fails; `compile_lm` / `compile_vision` know the layer paths of other Hugging Face multimodal wrappers;
  `worker_prep` retries without position ids; `get_world_model` calls the backbone hook and `compile_world_model`
  compiles the blocks of the Wan and Cosmos-Predict2 backbones.
- Compilation at start: PI0 with `all` reached its first step after 445 s with empty compiler caches, 165 s with the
  caches of that run.

### Session 12 (job 73045, tree d8544d8; released as 4e23836)

- `compile_head` with eager's random draws: one step's gradients differ from eager by 1.8% (GR00T), 3.4% (PI),
  6.6% (DiscreteDiffusion), 9.1% (Adapter), cosine ≥ 0.997 (before the fix: cosine ≈ 0); no compile fallback.
- Judge on the final tree: `exact` PASS N0, 23,336/23,336 tensors bitwise, 500.3 ms (1.81× the frozen 906.9 ms);
  `all` (now with `adamw_compiled` and `compile_head`) PASS N1, gate 3 equivalent, endpoints within 1.77σ,
  340.3 ms (2.66×; 2.13× on 8427140 without the two paths).
- Table (13 recipes × ref / exact / all / max) and the QwenPI statistical gate for the new `all` / `max`: running.

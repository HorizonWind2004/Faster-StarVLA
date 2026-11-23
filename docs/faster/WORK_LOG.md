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

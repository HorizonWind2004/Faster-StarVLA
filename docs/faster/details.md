# Faster-StarVLA: details

[Back to the README](../../README.md) · [简体中文](details.zh-CN.md)

These figures are from the session in which the baseline timing was frozen. In three later sessions the baseline itself ran faster (796 to 816 ms instead of 905 ms), and the same comparison gave 1.87× to 1.96× for `all` and 1.56× to 1.65× for `exact` ([sessions](measurement.md#the-judge-on-the-unified-code)).

## How it works

### Three levels

The levels differ in what they allow to change, and each was judged by the standard that fits it:

| Level | What it adds | Compared with upstream | Judge tier | Headline recipe |
|---|---|---|:---:|---:|
| `exact` | Work moved off the critical path or not done at all: input processing in the data-loader workers, no unused vocabulary logits, mask and position tables without per-call overhead, batched attention calls only where a probe proves them bitwise identical, leaner DeepSpeed bookkeeping | **Bitwise identical** in deterministic mode: loss, every gradient, update and optimizer state (and, where checked on real weights, the actions of `predict_action`) | N0 | 1.74× |
| `all` | `exact` + changes that keep the precision of every operation but may change the order of floating-point rounding: Inductor-compiled layers, SDPA instead of flash-attention, batched attention for every shape, conditioning projected once, a compiled AdamW update | **Statistically equivalent**: every training endpoint within 3σ of upstream's seed-to-seed spread | N1 | 2.00× |
| `max` | `all` + precision settings: TF32 for fp32 matrix products (the fp32 action heads), bf16 softmax in PI0 / PI05 attention | Statistically equivalent, with some operations at lower precision | N2 | no further gain (little fp32 work) |

Deterministic mode is the setting in which upstream reproduces itself bit for bit; outside it, two runs of upstream
generally differ in the last bits already (GPU reductions in varying order), so a bitwise comparison means something
only there. The judge's tiers are the ones
of [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent): N0 bitwise, N1 reordered at the same
precision, N2 a dtype or math mode changed.

### Where each fast path acts

All of it lives in one package, [`starVLA/fast/`](../../starVLA/fast/__init__.py), called from four places that belong to no
framework: where a backbone is built (`get_vlm_model`), where a framework is built (`build_framework`), where the
trainer gets its DeepSpeed engine, and where the data loader is built. No framework and no backbone interface
contains a switch or a line of it. A fast path is a module with `apply(target) -> bool` and one line in
[`FAST_PATHS`](../../starVLA/fast/__init__.py); it returns `False` where it does not fit, and the log lists it as not applicable.

## Every fast path

| Fast path | Level | Acts on | What it does, and where it applies |
|---|---|---|---|
| `worker_prep` | exact | data loader | the prompt construction, the VLM processor and the rotary position ids run in the data-loader workers instead of the training process; every framework that hands the batch's images and prompts to its backbone interface (`build_qwenvl_inputs`), on the Qwen2.5-VL, Qwen3-VL and Cosmos-Reason2 interfaces; a framework that edits the prompt registers the edit in one function |
| `worker_prep_framework` | exact | data loader | the same for a framework that builds its backbone inputs itself (MiniCPM-RobotManip: its MiniCPM-V processor call), probed against the framework's own builder before it is used |
| `video_prep` | exact | data loader | a world-model backbone's per-sample image resize and normalization (Wan, Cosmos-Predict2) run in the data-loader workers; the processor in the workers is probed against the one in the training process |
| `worker_examples` | exact | data loader | PI0 / PI05: the framework's own sample preparation (image resize and padding, state and action padding) runs in the data-loader workers |
| `vision_attn` | exact | vision tower | one batched attention call per layer instead of one call per image with a host synchronization in every layer; Qwen2-VL / Qwen2.5-VL / Qwen3-VL vision towers with SDPA attention, for the image shapes where a probe on the running GPU shows the batched call is bitwise identical |
| `vision_pos` | exact | vision tower | position tables built once per grid size instead of in every forward; Qwen2-VL / Qwen2.5-VL / Qwen3-VL vision towers |
| `attn_mask` | exact | language model | the causal + padding mask from two broadcasted tensor ops instead of four nested `torch.vmap` calls; any backbone whose transformers model builds its mask with `masking_utils` (Qwen, Gemma, Llama, Phi families), with SDPA attention |
| `hidden_only` | exact | language model | the unused vocabulary logits are not computed; any backbone with an LM head, in calls that ask for hidden states without labels |
| `grad_norm` | exact | optimizer step | the clipping norm without an fp64 copy of every gradient (18 GB written and read per step on the headline recipe); any framework trained with DeepSpeed ZeRO-2 |
| `allreduce` | exact | optimizer step | gradient buckets all-reduced in place; any framework trained with DeepSpeed ZeRO-2 |
| `sdpa` | all | whole backbone | a backbone loaded with flash-attention (or with eager attention only because a sub-model has no SDPA: Gemma 4) is switched to PyTorch's scaled-dot-product attention, the same attention from another kernel, which the SDPA paths and the compiled layers need. A vision tower that off flash-attention would run one attention call per image slice or window, and that `vision_attn` does not batch (MiniCPM-V 4.6), keeps flash-attention |
| `vision_attn_relaxed` | all | vision tower | the batched vision attention for every uniform image shape (last-bit differences) |
| `compile_vision`, `compile_lm` | all | vision tower, language model | the vision blocks and decoder layers compiled with Inductor for training calls, with `gc.freeze()` against the garbage-collector pauses that compilation causes; backbones with `model.visual.blocks` / `model.language_model.layers` (the Qwen-VL families) and the corresponding places of other Hugging Face multimodal wrappers, with SDPA attention; a layer the compiler cannot handle stays eager |
| `compile_world_model` | all | world model | the transformer blocks of a video world-model backbone (Wan, Cosmos-Predict2) compiled with Inductor for training calls |
| `text_longest` | all | world model | Wan: the 4.7 B UMT5 text encoder on the batch's longest instruction instead of 512 padded tokens (upstream zeroes the positions past the instruction afterwards); only while the encoder has no active dropout: evaluation and inference. In training upstream keeps UMT5's dropout on, and the path leaves the encoding as it is |
| `frozen_no_grad` | all | optimizer | Wan: the UMT5 text encoder and the VAE (6.5 B parameters) run under `torch.no_grad` only but are left trainable upstream, so ZeRO-2 keeps fp32 copies and Adam moments for them and updates them with zero gradients, which leaves them bitwise unchanged; they are frozen before the optimizer is built. Applies only when every call into them is under `no_grad` and lr × weight decay rounds away in fp32; the clipping norm differs in rounding, and optimizer checkpoints are not interchangeable with runs without it |
| `condition_once` | all | action head | action heads built on starVLA's cross-attention DiT (GR00T, layer-wise flow matching, discrete diffusion heads): conditioning that the framework repeated for several noise levels is projected once per sample |
| `compile_head` | all | action head | the blocks of the action head (the DiT heads of GR00T, the layer-wise flow-matching and discrete-diffusion heads, LangForce, QwenDual; the adapter head's residual blocks) compiled with Inductor, with eager's random draws for dropout |
| `openpi_one_pass` | all | vision tower | PI0 / PI05: the camera views of a batch go through SigLIP in one call instead of one call per view |
| `unused_views` | all | vision tower | PI0 / PI05: a camera view that no sample of the batch has is left out instead of being encoded as a black image |
| `openpi_compile` | all | whole backbone | PI0 / PI05: the SigLIP layers and the joint PaliGemma / action-expert layers compiled with Inductor for training calls |
| `adamw_compiled` | all | optimizer step | every framework trained with DeepSpeed ZeRO-2 and torch's fused AdamW: the update of each rank's flat partition as one compiled kernel (torch's fused kernel took 104.9 ms on a 2.25-billion element partition on B300, the compiled one 9.4 ms) |
| `tf32` | max | action head | TF32 for fp32 matrix products, which is what the fp32 action heads spend their time on. A precision setting, so not part of `all` |
| `openpi_sdpa` | max | whole backbone | PI0 / PI05: fused scaled-dot-product attention in the SigLIP tower and the joint PaliGemma / action-expert layers instead of matmul, fp32 softmax, matmul; the softmax then runs in bf16 |
| `unit_clip` | by name | optimizer step | no pass over the gradient to multiply it by a clipping factor of exactly 1; it reads the factor on the host, a synchronization in every step, to save one pass of 2.6 ms |
| `grad_cast` | by name | optimizer step | the partition's gradients cast to fp32 while they are flattened, in one pass instead of two; no difference beyond run-to-run noise on the headline recipe |
| `reduce_to_owner` | by name | optimizer step | gradient buckets reduced to the owning rank only; measured slower than the default on two ranks |

"By name" paths are in no level; select them explicitly (`STARVLA_FAST=all,unit_clip`). The first release's per-path
variables (`STARVLA_PREP_IN_WORKERS=1` and so on) are still honoured.

## Other frameworks and backbones

starVLA is a framework of many policies and backbones. The headline numbers are from one recipe (QwenOFT on
Qwen3-VL-4B). The same setting on the other frameworks, same hardware, data and batch size, each row against the
baseline of the same job (23 recipes, 18 frameworks, 9 backbones):

| Framework | Backbone | Attention in the recipe | Baseline (ms/step) | `exact` | `all` | `max` | `exact` checked bit for bit |
|---|---|---|---:|---:|---:|---:|---|
| QwenOFT | Qwen3-VL-4B | SDPA | 801.6 | 489.8 (1.64×) | 292.2 (2.74×) | 305.5 (2.62×) | judge: 23,336 tensors; 728 gradients + actions ✓ |
| QwenGR00T | Qwen3-VL-4B | SDPA | 880.4 | 605.2 (1.45×) | 399.1 (2.21×) | 392.6 (2.24×) | 960 gradients + actions ✓ |
| QwenPI | Qwen3-VL-4B | SDPA | 1293.8 | 1110.8 (1.16×) | 811.7 (1.59×) | 530.5 (2.44×) | 1,225 gradients + actions ✓ |
| QwenPI_v3 | Qwen3-VL-4B | SDPA | 1638.3 | 1389.5 (1.18×) | 849.8 (1.93×) | 519.7 (3.15×) | 1,376 gradients + actions ✓ |
| QwenDiscreteDiffusion | Qwen3-VL-4B | SDPA | 2512.9 | 2295.7 (1.09×) | 1884.9 (1.33×) | 566.9 (4.43×) | 1,220 gradients + actions ✓ |
| QwenAdapter | Qwen3-VL-4B | SDPA | 1052.3 | 767.9 (1.37×) | 553.0 (1.90×) | 555.8 (1.89×) | 1,226 gradients + actions ✓ |
| LangForce | Qwen3-VL-4B | SDPA | 1439.6 | 905.7 (1.59×) | 686.5 (2.10×) | 673.7 (2.14×) | 961 gradients + actions ✓ |
| QwenDual | Qwen3-VL-4B | SDPA | 1201.1 | 909.2 (1.32×) | 546.6 (2.20×) | 521.0 (2.31×) | 1,135 of 1,136 gradients + actions (the other 1 differs between two baseline runs) ✓ |
| QwenOFT | Qwen2.5-VL-3B | SDPA | 1032.8 | 540.9 (1.91×) | 328.6 (3.14×) | 326.0 (3.17×) | 840 gradients + actions ✓ |
| QwenOFT | Qwen2.5-VL-3B | flash-attention | 785.9 | 588.3 (1.34×) | 329.2 (2.39×) | 325.0 (2.42×) | 840 gradients + actions ✓ |
| QwenGR00T | Qwen2.5-VL-3B | flash-attention | 861.9 | 692.9 (1.24×) | 405.3 (2.13×) | 405.2 (2.13×) | 1,072 gradients + actions ✓ |
| PI0 | PaliGemma (openpi) | eager | 639.2 | 591.5 (1.08×) | 290.3 (2.20×) | 288.9 (2.21×) | — |
| PI05 | PaliGemma (openpi) | eager | 670.2 | 626.8 (1.07×) | 308.3 (2.17×) | 315.9 (2.12×) | — |
| QwenGR00T | Qwen3.5-0.8B | flash-attention | 599.9 | 443.7 (1.35×) | 407.3 (1.47×) | 406.7 (1.48×) | — |
| QwenOFT | Qwen3.5-0.8B | flash-attention | 521.9 | 361.0 (1.45×) | 326.3 (1.60×) | 321.5 (1.62×) | — |
| Gemma4PI | Gemma-4-E2B (batch 4) | eager | 1080.1 | 965.0 (1.12×) | 594.3 (1.82×) | 581.3 (1.86×) | — |
| Gemma4GR00T | Gemma-4-E2B (batch 4) | eager | 1062.8 | 875.7 (1.21×) | 524.7 (2.03×) | 525.3 (2.02×) | — |
| MiniCPMGR00T | MiniCPM-V 4.6 | flash-attention | 867.9 | 570.4 (1.52×) | 519.2 (1.67×) | 527.1 (1.65×) | — |
| MiniCPMPI | MiniCPM-V 4.6 | flash-attention | 825.3 | 595.2 (1.39×) | 551.1 (1.50×) | 550.8 (1.50×) | — |
| MiniCPMRobotManip | MiniCPM-V 4.6 (RobotManip checkpoint; 10-D shapes from LIBERO) | eager | 1819.3 | 1532.9 (1.19×) | 498.4 (3.65×) | 499.7 (3.64×) | — |
| WanPI | Wan2.2-TI2V-5B (world model; head depth 30, upstream fails with 36) | SDPA | 2102.5 | 1755.7 (1.20×) | 958.9 (2.19×) | 725.3 (2.90×) | — |
| WanOFT | Wan2.2-TI2V-5B (world model) | SDPA | 1479.8 | 1136.2 (1.30×) | 540.2 (2.74×) | 538.0 (2.75×) | — |
| WanGR00T | Wan2.2-TI2V-5B (world model) | SDPA | 1611.8 | 1238.3 (1.30×) | 631.5 (2.55×) | 622.9 (2.59×) | — |

Every row is compared with a baseline from its own batch job. Between jobs on the same node the baseline of one recipe moved by up to 10%, so milliseconds of different jobs are not comparable; the ratios within a row are ([sessions](measurement.md#other-frameworks-and-backbones-step-time)).

`exact` gains where the VLM's input processing and the backbone are a large part of the step (LangForce builds two prompts per step: 1.59×) and little on PI0 / PI05 (1.08×), whose step is the PaliGemma forward and backward itself.

The flow-matching heads run in fp32. `condition_once` (in `all`) projects the conditioning that a framework repeats for several noise levels once per sample. `max` also lets the head's fp32 matrix products use TF32; where every block of the head projects VLM hidden states, those products are most of what is left (QwenPI 812 → 531 ms; QwenPI_v3 850 → 520 ms; QwenDiscreteDiffusion 1885 → 567 ms). TF32 is a precision setting rather than a reordering, so it is a level of its own. The judge accepts it on the headline recipe as statistically equivalent; the same test on QwenPI (`all` on 6 seeds and `max` on 4 seeds) finds every endpoint within the judge's limit of 3σ, the largest being 1.6σ (VLM update size with `max`) ([details](measurement.md#the-statistical-gate-on-qwenpi)), and the loss curves of the other frameworks over the 220 timed steps are indistinguishable from the baseline's ([numbers](measurement.md#tf32)).

The Qwen2.5-VL recipes load the backbone with flash-attention. `exact` keeps it and applies the paths that do not depend on the attention kernel. `all` first switches the backbone to PyTorch's scaled-dot-product attention (`sdpa`), after which the SDPA paths and the compiled layers apply: QwenOFT 786 → 588 (1.34×) → 329 (2.39×) ms.

On Qwen2.5-VL with SDPA (bf16, B300), the bitwise probe rejects batching for the four full-attention vision layers (32 images × 256 patches, head dimension 80), so `exact` batches only the 28 window-attention layers; `all` batches both.

PI0 and PI05 bring their own PaliGemma instead of a `get_vlm_model` backbone. `exact` is `worker_examples` and the two DeepSpeed paths there; `all` adds `unused_views` (LIBERO has two of the three camera views these models expect, so the prefix shrinks from 816 to 560 tokens) and `openpi_compile`; `max` adds `openpi_sdpa`. The whole model already runs in bf16, so TF32 changes nothing there.

New backbones (session 13): Wan2.2-TI2V-5B (WanGR00T, WanOFT, WanPI), Qwen3.5-0.8B, Gemma-4-E2B and MiniCPM-V 4.6
(MiniCPMGR00T, MiniCPMPI, MiniCPM-RobotManip) are in the table above. On the real Qwen3.5-0.8B the `exact` paths leave
loss and gradients bitwise identical. What made the difference on them:

- **Wan: sub-models left trainable by mistake (`frozen_no_grad`).** starVLA's Wan interface runs its UMT5 text encoder
  and VAE (6.5 B parameters) under `torch.no_grad` only, but the lines that freeze them are commented out, so ZeRO-2
  keeps fp32 copies and Adam moments for them and updates them with zero gradients every step. With LIBERO's learning
  rates and weight decay that update leaves the weights bitwise unchanged, so freezing them changes nothing but the
  cost: about 100 ms per step and 42–52 GiB of peak memory on every Wan recipe (WanPI `all` 1.99× → 2.19×, `max`
  2.53× → 2.90×). The same check flags ABot-M0's VGGT-1B spatial model (not timed here). A frozen sub-model that is
  ever run with gradients raises instead of silently training less than upstream.
- **MiniCPM-V 4.6: per-chunk vision attention.** Off flash-attention its vision tower reads the chunk lengths back to
  the host in every layer and makes one attention call per image slice and per 2 × 2 window (about 9,000 calls per
  step). The `all` level runs that tower on flash-attention (kept there when the recipe loads it, moved there from SDPA
  otherwise), one call per layer: MiniCPM-RobotManip 1,533 → 498 ms per step. `compile_vision` leaves such towers alone.
- **Qwen3.5 and Gemma 4.** `compile_lm` leaves a language model with linear-attention layers (Qwen3.5) alone, since
  compiling it made the first step take 15 minutes and the step slower; and a batch without padding gets the same causal
  mask as any other batch, so compiled layers are not compiled again in the middle of training (Gemma 4 at batch 4 hit
  one at step 161). Gemma 4 runs at batch 4 because its eager baseline does not fit 16.

Upstream failures met on the way, timed with the smallest change that makes the reference run: WanPI sets a head depth
of 30 that the layer-wise head no longer reads (it builds 36 layers and fails with `IndexError`; timed with
`diffusion_model_cfg.num_layers=30`); MiniCPM-RobotManip's released action head is fp32 while starVLA's DeepSpeed bf16
trainer stores it in bf16 (a timing-only patch casts the head's inputs, in both trees), and it trains on 10-D
end-effector data that is not here (actions and states are zero-padded to its shapes, so its loss values are not
meaningful); starVLA's Florence-2 interface takes one camera view where the LIBERO recipe gives two (distance checked
on the backbone's loss only). `text_longest` (Wan) is for evaluation and inference. Qwen3.5, Gemma 4 and MiniCPM-V 4.6
need transformers 5; upstream pins 4.57. The new recipes are timed against their own baseline in one job; on the
real models, the rounding-distance check covers their compile and head paths (WanGR00T, MiniCPM-RobotManip, Gemma4PI);
`frozen_no_grad` and the flash-attention vision tower are covered by the CPU tests (and the bitwise argument above);
on the real models (judge job 73722, tree 8ac4405 against upstream 99764f5): WanGR00T `exact` is bitwise (17,204 / 17,204 tensors after two steps and after evaluation); WanGR00T `all,-frozen_no_grad` and MiniCPMGR00T `all` pass the statistical gate (largest bias 1.91 sigma and 0.85 sigma; the lr x 1.02 control is rejected at 17.6 / 22.4 sigma). WanGR00T `all` reaches 3.05 sigma on one endpoint (size of the VLM update) on seed 0, less than upstream against itself on the same endpoint (4.78 sigma), and pooled over four seeds it is within 0.90 sigma on every endpoint: equivalent, but that endpoint's one-seed noise is larger than the six-seed sigma says. MiniCPMGR00T upstream does not reproduce itself bit for bit (3,175 / 16,436 tensors), so the bitwise gate cannot decide its `exact` level. Not measured: Molmo2, VILA and the real Cosmos-Reason2 weights (its
interface is exercised by the CPU test on the Qwen3-VL architecture); QwenFast (it hands label text to the interface,
which `worker_prep` leaves to the training process), InternVLA-M1 and ABot-M0; Cosmos-Predict2 and the vision-only
frameworks; and the co-training and VLM trainers (they call the same hooks).

## Validation

Everything was judged by a frozen evaluator built before optimization started
([judge/](judge/), [how it decides](measurement.md#the-judge)). It runs four gates in order, and its
verdict carries a tier:

| Gate | What it checks |
|---|---|
| 0 Structure | the training entry point's setup equals upstream's syntax tree; the data pipeline yields bitwise the upstream samples and first batches |
| 1 Benefit | fresh processes of the real entry point, cold compile caches, 20 warm-up steps, then two windows of 100 steps; reference → candidate → reference, and the speedup must exceed `max(1.02, 1 + 2 × reference spread)` |
| 2 Bitwise | two 2-step cases in deterministic mode on both ranks: the exact items (random-number state, learning rates, optimizer and DeepSpeed settings, trainable parameters, dtypes, a digest of the VLM inputs), then the loss and the fp32 gradient, update and Adam states of every parameter, bit for bit. All equal: **PASS N0** |
| 3 Statistical | only if gate 2 is not bitwise: a 60-step trajectory on a frozen seed; seven endpoints within 3σ of upstream's seed-to-seed spread (6 seeds): **PASS N1**, or **N2** if a dtype or math mode (TF32, reduced-precision reductions, SDPA backend) changed |

Before optimization the judge was shown to reject what it should (a changed training setup, a changed clipping
threshold, a VLM without gradient, every learning rate × 1.02) and to pass an equivalent reordering.

**`STARVLA_FAST=exact`.** With the seven exact fast paths that apply to this recipe on, two 2-step training cases in deterministic mode
(one with the recipe's periodic evaluation between the steps), on both ranks: 23,336 recorded tensors (the
loss, and the fp32 gradient, update and both Adam states of all 729 parameters) are bitwise identical to upstream,
as are the learning rates, the random-number state and a digest of what was fed to the VLM.

**`STARVLA_FAST=all`.** The compiled layers change floating-point rounding, so the full set is compared
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

**On the current code.** The checks above were made on the first release, before the fast paths were moved behind `STARVLA_FAST` and extended to the other frameworks. On the current code the judge finds `STARVLA_FAST=exact` bitwise identical again: 23,336 of 23,336 tensors, in each of the 6 later sessions that ran it, the last one on the released commit. The three-process check of `STARVLA_FAST=all` gave no verdict in the later sessions: the baseline ran faster than the frozen timing by more than the allowed band, and the judge then refuses to conclude (INVALID). Its own measurements were 1.87× to 1.96×. The agent-style check, which compares one candidate run with the frozen timing and then runs the bitwise and the statistical gate, passes for `all` on the released commit (tier N1, endpoints within 1.77σ; 340 ms per step, 2.66× the frozen 906.9 ms) and, on an earlier commit, for `all,tf32`, which is what `max` selects on this recipe (tier N2, within 1.41σ). [Details](measurement.md#the-judge-on-the-unified-code).

**Other frameworks, bit for bit.** Beyond the judge's recipe, `exact` was compared with the baseline on real weights for
every framework in the table above that has its weights here: one training step (loss and every gradient) and one
`predict_action` call, in deterministic mode (the last column of the table; [all checks](measurement.md#other-frameworks-and-backbones-bitwise-checks-on-real-weights)).
How far a rounding-level path moves one training step, next to what switching SDPA to eager attention does, is
[measured there too](measurement.md#how-far-a-rounding-level-path-moves-one-training-step).

**LIBERO evaluation.** A full training run with all fast paths (30,000 steps, seed 42, 2 x B300, global batch 32) was evaluated with starVLA's LIBERO protocol, 500 episodes per suite, next to the released checkpoint in the same pipeline. The released checkpoint was trained by the starVLA authors on 8 GPUs (global batch 128, checkpoint at step 50,000); this is a check that the accelerated code trains a working policy, not a matched-seed comparison. The 30,000 steps took 3 h 51 min of wall time including the recipe's periodic evaluations and three checkpoint saves (462 ms per step overall); at the frozen upstream step time the steps alone would take 7.6 h (not run).

| Success rate (%) | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| starVLA Qwen3-VL-OFT, published | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Released checkpoint (`StarVLA/Qwen3-VL-OFT-LIBERO-4in1`, step 50,000), our evaluation | 98.6 | 100.0 | 99.4 | 96.6 | 98.65 |
| **Faster-StarVLA run** (all fast paths, 30,000 steps on 2 GPUs) | 98.8 | 98.6 | 98.4 | 93.4 | 97.30 |

## FAQ

**Does it change my results?** With `exact`, no: in deterministic mode every recorded tensor of a training step is
bitwise identical, and so are the actions of `predict_action` wherever that was checked on real weights. With `all`, the arithmetic is the same at the same
precision, but compiled kernels round in a different order; the judge compares such runs statistically and finds them
within upstream's own seed-to-seed spread, and a full LIBERO training run with them gives a working policy (above).
`max` lowers the precision of some operations (TF32, bf16 softmax) and was accepted as tier N2.

**How do I turn one path off, or add one?** List it after the level: `STARVLA_FAST=all,-compile_lm` removes one,
`STARVLA_FAST=exact,compile_head` adds one. Unknown names are an error, not ignored. The start-up log prints, per hook,
the paths that were applied and the ones that do not apply to this model.

**Does it help inference?** Set the same variable for the policy server. The compiled layers, `condition_once` and the
DeepSpeed paths act on training only (gradient-enabled calls); in `predict_action` the others apply, for example no LM
head, the direct mask, batched vision attention, the position cache, `sdpa` and `text_longest`. `tools/eval/` adds a policy server
that batches concurrent simulator requests ([below](#faster-evaluation)).

**Is the first step slower?** With `all`, the first training steps compile the layers with Inductor. PI0 reached its
first step after 445 s with empty compiler caches and after 165 s with the caches of an earlier run; the
cache persists between runs (`TORCHINDUCTOR_CACHE_DIR`). The timings above start after 20 warm-up steps, and a
compilation inside a timed window invalidates the run.

**Memory?** Lower with `all`: peak 66.5 → 59.0 GiB per GPU on the headline recipe, most of it from the round
that compiled the decoder layers. Per recipe in [the measurement details](measurement.md#other-frameworks-and-backbones-step-time).

**Which backbones need transformers 5?** Qwen3.5 and Gemma 4 (upstream pins 4.57). The fast paths work with both
versions; the state of each new backbone is in [Other frameworks and backbones](#other-frameworks-and-backbones).

**Other GPUs?** Everything here was measured on B300 (torch 2.9.1, CUDA 12.9). `vision_attn` decides per device, with
a bitwise probe on the running GPU, which shapes it batches. Neither the speedups nor the bitwise identity were checked
on other GPUs; [judge/](judge/) says how to rerun the judge elsewhere.

## Faster evaluation

`tools/eval/` holds the batching policy server used for the evaluations above: concurrent requests from the
simulation clients become one `predict_action` call instead of queueing one by one. See
[measurement.md](measurement.md#evaluation).

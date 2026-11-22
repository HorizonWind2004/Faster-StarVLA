"""Opt-in fast paths of the trainer. Every switch is an environment variable, off unless set to a non-zero value;
with none set the trainer runs the upstream code path.

STARVLA_PREP_IN_WORKERS   QwenOFT: build the batch's VLM inputs (prompt construction + Qwen3-VL processor, 0.19 s of
                          single-threaded CPU per step) in the data-loader workers instead of in the training
                          process, where the GPU waits for them. Same code on the same samples: same tensors.
STARVLA_SKIP_LM_HEAD      QwenOFT training reads the VLM's last hidden state only: call the backbone without the
                          language-model head, whose logits (batch x tokens x 151,936) are never used. Same hidden
                          state (the last decoder layer's output), same gradients.
STARVLA_FAST_ATTN_MASK    build the causal + padding attention mask with two broadcasted tensor ops instead of
                          transformers' four nested torch.vmap calls (starVLA/model/modules/vlm/fast_masks.py).
                          Same boolean mask.
STARVLA_BATCHED_VISION_ATTN  Qwen3-VL vision tower with SDPA: one batched attention call per layer instead of one call
                          per image plus a host read of the chunk lengths in every layer, when all images have the
                          same number of patches (starVLA/model/modules/vlm/fast_vision_attn.py). Same outputs and
                          gradients, bit for bit.
STARVLA_VISION_POS_CACHE  Qwen3-VL vision tower: build the rotary position table and the position-embedding
                          interpolation indices once per distinct grid instead of in every forward with a per-image
                          host loop (starVLA/model/modules/vlm/fast_vision_pos.py). Same tensors; the lookup in the
                          trainable position embedding still runs every step.
STARVLA_DS_REDUCE_TO_OWNER  DeepSpeed ZeRO-2: reduce every gradient bucket to the rank that owns those parameters
                          (zero_optimization.use_multi_rank_bucket_allreduce = false) instead of all-reducing it to
                          every rank and discarding the copy on the ranks that do not own it. Each owner receives
                          the same sum; the transfer volume is halved on two ranks.
"""
import os


def switch(name):
    return os.environ.get(name, "0") not in ("", "0", "false", "False")


def apply_deepspeed_switches(model):
    """Runtime DeepSpeed settings selected by switches; `model` is the DeepSpeed engine Accelerate returned."""
    zero_optimizer = getattr(model, "optimizer", None)
    if switch("STARVLA_DS_REDUCE_TO_OWNER") and hasattr(zero_optimizer, "use_multi_rank_bucket_allreduce"):
        zero_optimizer.use_multi_rank_bucket_allreduce = False

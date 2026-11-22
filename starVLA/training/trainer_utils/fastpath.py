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
STARVLA_COMPILE_LM        torch.compile (Inductor) of each Qwen3-VL decoder layer's training forward: the elementwise
                          chains of forward and backward (RMSNorm, rotary embedding, gating, casts) become a few fused
                          kernels per layer. Changes floating-point rounding, not the function (not bitwise).
STARVLA_DS_REDUCE_TO_OWNER  DeepSpeed ZeRO-2: reduce every gradient bucket to the rank that owns those parameters
                          (zero_optimization.use_multi_rank_bucket_allreduce = false) instead of all-reducing it to
                          every rank and discarding the copy on the ranks that do not own it. Each owner receives
                          the same sum; the transfer volume is halved on two ranks.
STARVLA_DS_FUSED_GRAD_NORM  DeepSpeed ZeRO-2 clipping norm: vector_norm(g, dtype=float64) instead of
                          vector_norm(g.double()), per parameter as before. Same value bit for bit, without
                          materializing every gradient in fp64.
"""
import os


def switch(name):
    return os.environ.get(name, "0") not in ("", "0", "false", "False")


def _fused_grad_norm(zero_optimizer):
    """DeepSpeedZeroOptimizer.get_grad_norm_direct (0.16.9) with the fp64 cast fused into the norm kernel.

    DeepSpeed computes the clipping norm per parameter as vector_norm(g.double()): every gradient of this rank's
    partition (2.25 B elements) is first materialized in fp64, 18 GB written and read back per step.
    vector_norm(g, dtype=float64) accumulates in fp64 without the copy and returns the same bits (33 shapes compared
    on B300: identical; 68 ms -> 2 ms). Everything else is DeepSpeed's code; the inf-norm branch is delegated."""
    import types
    from math import inf

    import torch
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.utils import is_model_parallel_parameter, mask_nan_or_inf_with_val_inplace
    from deepspeed.runtime.zero import stage_1_and_2 as zero

    original = type(zero_optimizer).get_grad_norm_direct

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        if norm_type == inf:
            return original(self, gradients, params, norm_type=norm_type)
        all_norms = []
        for g, p in zip(gradients, params):
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, zero.PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                all_norms.append(
                    torch.linalg.vector_norm(g.data, ord=norm_type, dtype=torch.float64).to(
                        get_accelerator().current_device_name()
                    )
                )
        if len(all_norms) > 0:
            total_norm = torch.stack(all_norms).square().sum().float()
        else:
            total_norm = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        # Sum across all model parallel Device.
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)

        total_norm = total_norm.pow(1.0 / norm_type)

        mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)

        return total_norm

    zero_optimizer.get_grad_norm_direct = types.MethodType(get_grad_norm_direct, zero_optimizer)


def apply_deepspeed_switches(model):
    """Runtime DeepSpeed settings selected by switches; `model` is the DeepSpeed engine Accelerate returned."""
    zero_optimizer = getattr(model, "optimizer", None)
    if switch("STARVLA_DS_REDUCE_TO_OWNER") and hasattr(zero_optimizer, "use_multi_rank_bucket_allreduce"):
        zero_optimizer.use_multi_rank_bucket_allreduce = False
    if switch("STARVLA_DS_FUSED_GRAD_NORM") and type(zero_optimizer).__name__ == "DeepSpeedZeroOptimizer":
        _fused_grad_norm(zero_optimizer)

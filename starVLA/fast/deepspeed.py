"""Fast paths of the DeepSpeed ZeRO-2 optimizer; they act on the engine and are independent of the model.

fused_grad_norm    clipping norm: vector_norm(g, dtype=float64) instead of vector_norm(g.double()), per parameter as
                   before. Same value bit for bit, without materializing every gradient in fp64.
cast_while_flattening
                   step() builds the fp32 gradient of this rank's partition as flatten(gradients).to(float32): one
                   pass over the partition to concatenate the bf16 gradients, a second to cast the copy. Each
                   gradient is copied to its place in an fp32 buffer instead; the copy casts. Same values, one pass.
skip_unit_clip     step() multiplies the fp32 gradient by 1 / clip in every step, also when the norm is below the
                   clipping threshold and the factor is exactly 1: a pass over the whole partition that changes
                   nothing (x * 1.0 is x for every finite x and for infinities). That pass is skipped.
plain_allreduce    all-reduce each gradient bucket in place (zero_optimization.reduce_scatter = false) instead of
                   slicing it per parameter and owner, concatenating the slices, all-reducing the copy and copying
                   the owned part back. On two ranks (a + b) * 0.5 and a * 0.5 + b * 0.5 are the same bf16 value.
reduce_to_owner    reduce every gradient bucket to the rank that owns those parameters
                   (zero_optimization.use_multi_rank_bucket_allreduce = false) instead of all-reducing it to every
                   rank and discarding the copy on the ranks that do not own it. Measured slower on two ranks.
"""


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


def _zero_optimizer(engine):
    optimizer = getattr(engine, "optimizer", None)
    return optimizer if type(optimizer).__name__ == "DeepSpeedZeroOptimizer" else None


def fused_grad_norm(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None:
        return False
    _fused_grad_norm(optimizer)
    return True


def _cast_while_flattening(zero_optimizer):
    """DeepSpeedZeroOptimizer.step (0.16.9) calls self.flatten in two places, both followed by .to(dtype of the fp32
    partition). During step(), self.flatten returns an object whose .to() fills the fp32 buffer directly."""
    import types

    import torch

    flatten, original = zero_optimizer.flatten, type(zero_optimizer).step

    class Flattened:
        def __init__(self, tensors):
            self.tensors = list(tensors)

        def to(self, dtype, *args, **kwargs):
            tensors = self.tensors
            if args or kwargs or not tensors or not isinstance(dtype, torch.dtype) or dtype == tensors[0].dtype:
                return flatten(tensors).to(dtype, *args, **kwargs)
            flat = torch.empty(sum(t.numel() for t in tensors), dtype=dtype, device=tensors[0].device)
            offset = 0
            for tensor in tensors:
                flat.narrow(0, offset, tensor.numel()).copy_(tensor.reshape(-1))
                offset += tensor.numel()
            return flat

    def step(self, closure=None):
        self.flatten = Flattened
        try:
            return original(self, closure)
        finally:
            self.flatten = flatten

    zero_optimizer.step = types.MethodType(step, zero_optimizer)


def cast_while_flattening(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None or not callable(getattr(optimizer, "flatten", None)) or getattr(optimizer, "cpu_offload", False):
        return False
    if not getattr(optimizer.step, "_starvla_cast_while_flattening", False):
        _cast_while_flattening(optimizer)
        optimizer.step.__func__._starvla_cast_while_flattening = True
    return True


def _skip_unit_clip(zero_optimizer):
    """DeepSpeedZeroOptimizer.unscale_and_clip_grads (0.16.9): the factor is computed as there; the multiplication
    is DeepSpeed's own whenever the factor is not exactly 1."""
    import types

    import torch

    original = type(zero_optimizer).unscale_and_clip_grads

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            clip = torch.clamp(clip, min=1.0)
            combined_scale = clip * self.loss_scale
        if float(1. / combined_scale) == 1.0:
            return
        original(self, grad_groups_flat, total_norm)

    unscale_and_clip_grads._starvla_skip_unit_clip = True
    zero_optimizer.unscale_and_clip_grads = types.MethodType(unscale_and_clip_grads, zero_optimizer)


def skip_unit_clip(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None or not all(hasattr(optimizer, name) for name in ("unscale_and_clip_grads", "loss_scale", "clip_grad")):
        return False
    if not getattr(optimizer.unscale_and_clip_grads, "_starvla_skip_unit_clip", False):
        _skip_unit_clip(optimizer)
    return True


def plain_allreduce(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None or not hasattr(optimizer, "reduce_scatter"):
        return False
    optimizer.reduce_scatter = False
    return True


def reduce_to_owner(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None or not hasattr(optimizer, "use_multi_rank_bucket_allreduce"):
        return False
    optimizer.use_multi_rank_bucket_allreduce = False
    return True

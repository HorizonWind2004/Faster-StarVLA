"""Fast paths of the DeepSpeed ZeRO-2 optimizer; they act on the engine and are independent of the model.

cast_while_flattening
                   step() builds the fp32 gradient of this rank's partition as flatten(gradients).to(float32): one
                   pass over the partition to concatenate the bf16 gradients, a second to cast the copy. Each
                   gradient is copied to its place in an fp32 buffer instead; the copy casts. Same values, one pass.
skip_unit_clip     step() multiplies the fp32 gradient by 1 / clip in every step, also when the norm is below the
                   clipping threshold and the factor is exactly 1: a pass over the whole partition that changes
                   nothing (x * 1.0 is x for every finite x and for infinities). That pass is skipped. Knowing the
                   factor means reading it on the host, a synchronization that the bf16 step otherwise does not
                   have, so this is selectable by name only.
plain_allreduce    all-reduce each gradient bucket in place (zero_optimization.reduce_scatter = false) instead of
                   slicing it per parameter and owner, concatenating the slices, all-reducing the copy and copying
                   the owned part back. The default path halves each rank's gradient before the sum and this one
                   the sum: on one or two ranks a / 2 + b / 2 and (a + b) / 2 are the same bf16 / fp32 value,
                   except where a half is below the smallest normal number (|g| < 2.4e-38: the halves round, the
                   sum does not) or the sum overflows. With fp16 gradients a + b overflows from 6.6e4 on, so the
                   path declines there; with more than two ranks the two orders round differently and it declines
                   as well.

A clipping norm "without the fp64 copy" (vector_norm(g, dtype=float64) instead of vector_norm(g.double())) was a fast
path here until it was measured in isolation: torch makes the same fp64 copy inside vector_norm, both forms take
12.8 ms and 16.8 GiB of extra peak memory at 2.25 B elements on a B300. It saved nothing and was removed.
reduce_to_owner    reduce every gradient bucket to the rank that owns those parameters
                   (zero_optimization.use_multi_rank_bucket_allreduce = false) instead of all-reducing it to every
                   rank and discarding the copy on the ranks that do not own it. Measured slower on two ranks.
"""


def _zero_optimizer(engine):
    optimizer = getattr(engine, "optimizer", None)
    return optimizer if type(optimizer).__name__ == "DeepSpeedZeroOptimizer" else None


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
    import torch
    from deepspeed import comm as dist

    # bitwise identical to the default path on one or two ranks, for bf16 / fp32 gradients (see the module docstring)
    if dist.get_world_size(group=getattr(optimizer, "dp_process_group", None)) > 2:
        return False
    if torch.float16 in (getattr(optimizer, "communication_data_type", None), getattr(optimizer, "dtype", None)):
        return False
    optimizer.reduce_scatter = False
    return True


def reduce_to_owner(engine):
    optimizer = _zero_optimizer(engine)
    if optimizer is None or not hasattr(optimizer, "use_multi_rank_bucket_allreduce"):
        return False
    optimizer.use_multi_rank_bucket_allreduce = False
    return True

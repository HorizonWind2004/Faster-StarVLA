"""The AdamW update of large flat parameters as one compiled kernel (by name only: STARVLA_FAST=all,adamw_compiled).

Under ZeRO-2 each rank's optimizer holds one flat fp32 tensor per parameter group (2.25 billion elements for the
headline recipe on two ranks), and torch's fused AdamW kernel is what updates it. This path writes the same update
out in tensor operations and lets Inductor compile them into a single pass over the weights, the gradient and the
two moments. Torch's optimizer keeps its state (step counts, moments), so checkpoints are unchanged; only the
function that torch.optim calls for fused Adam is replaced, and anything but plain decoupled AdamW on fp32 CUDA
tensors goes to torch's own kernel as before.

The arithmetic is the same formula in a different order of operations, so results differ in rounding. Whether it is
faster than torch's kernel is a measurement (coverage/optimizer_probe.py); it is not part of any level.
"""
import torch

_update = None


def _adamw(p, g, m, v, decay, beta1, beta2, step_size, bias2_sqrt, eps):
    p.mul_(decay)
    m.mul_(beta1).add_(g * (1 - beta1))
    v.mul_(beta2).add_(g * g * (1 - beta2))
    p.sub_(m / (v.sqrt() / bias2_sqrt + eps) * step_size)


def compiled_update():
    global _update
    if _update is None:
        _update = torch.compile(_adamw, dynamic=True)
    return _update


def apply(engine):
    """Engine hook. False unless the engine steps a torch AdamW with fused=True (what starVLA's trainers build)."""
    from torch.optim import adam

    optimizer = getattr(getattr(engine, "optimizer", None), "optimizer", None)
    if not isinstance(optimizer, torch.optim.AdamW) or not optimizer.defaults.get("fused") or not torch.cuda.is_available():
        return False
    original = adam._fused_adam
    if getattr(original, "_starvla_compiled", False):
        return True

    def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, *, amsgrad,
                   has_complex, beta1, beta2, lr, weight_decay, eps, maximize, capturable, differentiable,
                   decoupled_weight_decay):
        plain = (params and decoupled_weight_decay and not (amsgrad or maximize or differentiable or has_complex)
                 and grad_scale is None and found_inf is None and not isinstance(lr, torch.Tensor)
                 and all(p.is_cuda and p.dtype == torch.float32 and g.dtype == torch.float32 for p, g in zip(params, grads)))
        if not plain:
            return original(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf,
                            amsgrad=amsgrad, has_complex=has_complex, beta1=beta1, beta2=beta2, lr=lr,
                            weight_decay=weight_decay, eps=eps, maximize=maximize, capturable=capturable,
                            differentiable=differentiable, decoupled_weight_decay=decoupled_weight_decay)
        update = compiled_update()
        for p, g, m, v, step in zip(params, grads, exp_avgs, exp_avg_sqs, state_steps):
            step += 1
            t = step.double()  # stays on the device: no host synchronization
            update(p, g, m, v, 1 - lr * weight_decay, beta1, beta2, lr / (1 - beta1 ** t), (1 - beta2 ** t).sqrt(), eps)

    fused_adam._starvla_compiled = True
    adam._fused_adam = fused_adam
    return True

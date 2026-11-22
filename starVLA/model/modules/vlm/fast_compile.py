"""Regional torch.compile of the Qwen3-VL decoder layers for training.

At the training shape (16 x 205 tokens) one forward + backward of the 36 decoder layers runs about 10,000 small GPU
ops: per layer and direction a handful of matrix products and dozens of elementwise ops (RMSNorm in fp32, rotary
embedding, SiLU gating, casts, broadcasts). The matrix products take a third of the time; the rest is the latency of
launching the small ops one by one. Compiling each layer's forward with Inductor fuses the elementwise chains of the
forward and of the backward into a few kernels each.

Only gradient-enabled calls use the compiled forward; evaluation (inference_mode / no_grad) keeps the eager one, so
nothing is compiled when the trainer's periodic evaluation first runs. Module structure and parameter names are
unchanged (the layer's `forward` attribute is replaced, the layer is not wrapped).
"""
import torch

_installed = False


def install(qwen_model, dynamic=True):
    """qwen_model: Qwen3VLForConditionalGeneration."""
    global _installed
    if _installed:
        return
    import torch._dynamo

    layers = qwen_model.model.language_model.layers
    # one cache entry per layer instance on the shared code object
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 4 * len(layers)))
    for layer in layers:
        eager = layer.forward
        compiled = torch.compile(eager, dynamic=dynamic)

        def forward(*args, _eager=eager, _compiled=compiled, **kwargs):
            if torch.is_grad_enabled():
                return _compiled(*args, **kwargs)
            return _eager(*args, **kwargs)

        layer.forward = forward
    _installed = True

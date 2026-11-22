"""Regional torch.compile of the Qwen3-VL decoder layers and vision blocks for training.

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

_installed = set()


def _compile_forwards(layers, dynamic):
    import torch._dynamo

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


def install(qwen_model, dynamic=True):
    """Compile the decoder layers of a Qwen3VLForConditionalGeneration."""
    if "lm" not in _installed:
        _compile_forwards(qwen_model.model.language_model.layers, dynamic)
        _installed.add("lm")


def install_vision(qwen_model, dynamic=True):
    """Compile the vision blocks of a Qwen3VLForConditionalGeneration (24 blocks; about 60 ops each per direction).
    Meant to be used with the batched vision attention (fast_vision_attn), whose per-layer code has no host reads."""
    if "vision" not in _installed:
        _compile_forwards(qwen_model.model.visual.blocks, dynamic)
        _installed.add("vision")

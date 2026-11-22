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
import gc

import torch

_installed = set()
_gc = {"layers": 0, "calls": 0, "frozen": False}
_FREEZE_AFTER_STEPS = 4  # training forwards before the freeze


def _compile_forwards(layers, dynamic):
    import torch._dynamo

    # one cache entry per layer instance on the shared code object
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 4 * len(layers)))
    _gc["layers"] += len(layers)
    for layer in layers:
        eager = layer.forward
        compiled = torch.compile(eager, dynamic=dynamic)

        def forward(*args, _eager=eager, _compiled=compiled, **kwargs):
            if torch.is_grad_enabled():
                out = _compiled(*args, **kwargs)
                if not _gc["frozen"]:
                    _freeze_compiler_objects()
                return out
            return _eager(*args, **kwargs)

        layer.forward = forward


def _freeze_compiler_objects():
    """Compiling leaves millions of long-lived Python objects behind (FX graphs, guards, the symbolic-shape
    environment). Every full garbage collection then walks all of them: 1-2 s pauses in single training steps,
    at fixed step numbers. Once the graphs exist (a few steps in; forward and backward are both compiled within the
    first step), move everything allocated so far to the permanent generation, which the collector does not scan."""
    _gc["calls"] += 1
    if _gc["calls"] >= _FREEZE_AFTER_STEPS * _gc["layers"]:
        gc.collect()
        gc.freeze()
        _gc["frozen"] = True


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

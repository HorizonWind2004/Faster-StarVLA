"""Regional torch.compile of the backbone's decoder layers and vision blocks for training.

Measured on Qwen3-VL-4B: at the training shape (16 x 205 tokens) one forward + backward of the 36 decoder layers runs about 10,000 small GPU
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

_gc = {"layers": 0, "calls": 0, "frozen": False}
_FREEZE_AFTER_STEPS = 4  # training forwards before the freeze


def _raise_cache_limits(entries):
    import torch._dynamo

    # one cache entry per layer instance (or per specialisation) on a shared code object
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 4 * entries))
    _gc["layers"] += entries


def _compiled_for_training(eager, dynamic):
    compiled = torch.compile(eager, dynamic=dynamic)

    def call(*args, **kwargs):
        if torch.is_grad_enabled():
            out = compiled(*args, **kwargs)
            if not _gc["frozen"]:
                _freeze_compiler_objects()
            return out
        return eager(*args, **kwargs)

    return call


def _compile_forwards(layers, dynamic):
    _raise_cache_limits(len(layers))
    for layer in layers:
        layer.forward = _compiled_for_training(layer.forward, dynamic)


def compile_method(owner, name, variants, dynamic=True):
    """Compile `owner.<name>` for training calls: a method that Dynamo specialises on a Python argument (a layer
    index), `variants` times."""
    _raise_cache_limits(variants)
    setattr(owner, name, _compiled_for_training(getattr(owner, name), dynamic))


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


def _modules(vlm, path):
    target = vlm
    for name in path.split("."):
        target = getattr(target, name, None)
        if target is None:
            return None
    return target


def _apply(vlm, path, dynamic=True):
    from starVLA.fast import attention_implementation

    layers = _modules(vlm, path)
    if layers is None or len(layers) == 0:
        return False
    # Dynamo cannot trace the flash-attn extension call: those layers stay eager
    owner = _modules(vlm, path.rsplit(".", 1)[0])
    if attention_implementation(owner, getattr(vlm, "model", None)) == "flash_attention_2":
        return False
    if not getattr(layers, "_starvla_compiled", False):
        _compile_forwards(layers, dynamic)
        layers._starvla_compiled = True
    return True


def apply_lm(vlm):
    """Compile the decoder layers of the backbone's language model (`model.model.language_model.layers`)."""
    return _apply(vlm, "model.model.language_model.layers")


def apply_vision(vlm):
    """Compile the blocks of the backbone's vision tower (`model.model.visual.blocks`). Meant to be used with
    vision_attn, whose per-layer code has no host reads."""
    return _apply(vlm, "model.model.visual.blocks")

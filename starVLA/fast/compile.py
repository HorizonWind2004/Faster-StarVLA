"""Regional torch.compile of the backbone's decoder layers and vision blocks, and of the action head's blocks, for training.

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

_gc = {"layers": 0, "calls": 0, "frozen": False, "reported": False}
_FREEZE_AFTER_STEPS = 4  # training forwards before the freeze


def _raise_cache_limits(entries):
    import torch._dynamo

    # one cache entry per layer instance (or per specialisation) on a shared code object
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 4 * entries))
    _gc["layers"] += entries


def _compiled_for_training(eager, dynamic):
    # Random operations (the dropout of the DiT action heads) keep eager's kernels and generator: without this,
    # compiled dropout draws its masks from Inductor's own random stream, so a compiled step uses other masks than
    # the default step. Only random operations are affected.
    import torch._inductor.config

    torch._inductor.config.fallback_random = True
    compiled = torch.compile(eager, dynamic=dynamic)
    state = {"works": None}  # None until the first compiled call has returned

    def call(*args, **kwargs):
        if not torch.is_grad_enabled() or state["works"] is False:
            return eager(*args, **kwargs)
        try:
            out = compiled(*args, **kwargs)
        except Exception as error:
            if state["works"]:  # it has worked before: this is the caller's error
                raise
            state["works"] = False  # a module the compiler cannot handle stays eager
            if not _gc["reported"]:
                _gc["reported"] = True
                print(f"[starVLA.fast] compile: a layer could not be compiled and runs as before "
                      f"({type(error).__name__}: {str(error)[:200]})", flush=True)
            return eager(*args, **kwargs)
        state["works"] = True
        if not _gc["frozen"]:
            _freeze_compiler_objects()
        return out

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


def _apply(vlm, paths, dynamic=True):
    """Compile the layers at the first of `paths` (dotted attribute paths from the interface) that exists."""
    from starVLA.fast import attention_implementation

    for path in paths:
        layers = _modules(vlm, path)
        if isinstance(layers, torch.nn.ModuleList) and len(layers):
            break
    else:
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
    """Compile the decoder layers of the backbone's language model: `model.model.language_model.layers` in the
    Qwen-VL and Gemma families; the other places are where other Hugging Face multimodal wrappers keep them."""
    return _apply(vlm, ("model.model.language_model.layers", "model.language_model.model.layers",
                        "model.language_model.layers"))


def apply_vision(vlm):
    """Compile the blocks of the backbone's vision tower: `model.model.visual.blocks` in the Qwen-VL family (meant to
    be used with vision_attn, whose per-layer code has no host reads), the encoder layers of a SigLIP / CLIP style
    tower elsewhere."""
    return _apply(vlm, ("model.model.visual.blocks", "model.model.vision_tower.vision_model.encoder.layers",
                        "model.vision_tower.vision_model.encoder.layers", "model.model.vision_tower.encoder.layers"))


def apply_world_model(backbone):
    """Compile the transformer blocks of a video world model used as the backbone (`transformer.blocks` in the Wan
    and Cosmos-Predict2 interfaces). Hidden states are read by hooks on the blocks, which stay outside the compiled
    forward."""
    return _apply(backbone, ("transformer.blocks", "transformer.transformer_blocks"))


def apply_head(framework):
    """Compile the transformer blocks of the action head: starVLA's cross-attention DiT (and the self-attention
    transformer some heads put in front of it), and the residual blocks of the adapter head. A head works on a few
    dozen action tokens per sample, so apart from the projections of the conditioning its time is the latency of
    many small operations."""
    kinds = []
    try:
        from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer

        kinds += [(DiT, "transformer_blocks"), (SelfAttentionTransformer, "transformer_blocks")]
    except ImportError:
        pass
    try:
        from starVLA.model.modules.action_model.VLA_AdapterHeader import MLPResNet

        kinds.append((MLPResNet, "mlp_resnet_blocks"))
    except ImportError:
        pass
    found = False
    for module in framework.modules():
        for kind, name in kinds:
            blocks = getattr(module, name, None) if isinstance(module, kind) else None
            if blocks is not None and len(blocks):
                if not getattr(blocks, "_starvla_compiled", False):
                    _compile_forwards(blocks, dynamic=True)
                    blocks._starvla_compiled = True
                found = True
    return found

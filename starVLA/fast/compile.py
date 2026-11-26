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
import inspect

import torch

_gc = {"layers": 0, "calls": 0, "frozen": False, "reported": False}
_FREEZE_AFTER_STEPS = 4  # training forwards before the freeze
_failed_kinds = set()


def _raise_cache_limits(entries):
    import torch._dynamo

    # one cache entry per layer instance (or per specialisation) on a shared code object
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 4 * entries))
    _gc["layers"] += entries


# Inductor settings for the action heads, applied to their compiled code only (forward and backward):
#   fallback_random: random operations (the DiT heads' dropout) use eager's kernels and generator. Otherwise compiled
#     dropout draws its masks from Inductor's own random stream, so a compiled step uses other masks than the default.
#   triton.coalesce_tiling_analysis off: the analysis does not support dynamic shapes (Inductor's own note) and
#     intermittently failed on a DiT block (a sympy assertion: sympy decides is_constant at random sample points).
HEAD_OPTIONS = {"fallback_random": True, "triton.coalesce_tiling_analysis": False}


def _compiled_for_training(eager, dynamic, options=None):
    compiled = torch.compile(eager, dynamic=dynamic, **({"options": options} if options else {}))
    state = {"works": None}  # None until the first compiled call has returned
    kind = _kind(eager)  # layers of one structure: once one fails to compile, the others are not tried

    def call(*args, **kwargs):
        if not torch.is_grad_enabled() or state["works"] is False:
            return eager(*args, **kwargs)
        if state["works"] is None and kind in _failed_kinds:
            state["works"] = False
            return eager(*args, **kwargs)
        try:
            out = compiled(*args, **kwargs)
        except Exception as error:
            if state["works"]:  # it has worked before: this is the caller's error
                raise
            state["works"] = False  # a module the compiler cannot handle stays eager
            if kind is not None:  # and so do the layers of the same structure (Qwen3.5: its linear-attention layers)
                _failed_kinds.add(kind)
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


def _kind(method):
    """A layer's class and the classes of its children (Qwen3.5's decoder layers are one class with either linear or
    full attention inside); None for anything but a module's method."""
    owner = getattr(method, "__self__", None)
    if not isinstance(owner, torch.nn.Module):
        return None
    return (type(owner), tuple(type(child) for child in owner.children()))


def _compile_forwards(layers, dynamic, options=None):
    _raise_cache_limits(len(layers))
    for layer in layers:
        layer.forward = _compiled_for_training(layer.forward, dynamic, options)


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


def _apply(vlm, paths, dynamic=True, prepare=None):
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
        _no_training_cache(owner)
        if prepare is not None:
            prepare(layers, owner, path, vlm)
        layers._starvla_compiled = True
    return True


def _no_training_cache(model):
    """A language model (Qwen3.5, Gemma 4) builds a cache in every forward unless it is given
    use_cache=False, and the config defaults it to True, training included. Each compiled layer then guards on the
    state of its own cache layer (initialized or not, its device), which differs between the layers and between the
    calls of one step: Dynamo recompiled the layers more than a hundred times, about 40 s each, before the first
    training step ended (Qwen3.5-0.8B, B300). Gemma 4's language model gets such a cache as well. Training
    forwards here that are not given a cache get use_cache=False: nothing reads the cache of a training step, and the
    layers compute the same values without it (Gemma 4 shares keys and values between layers through its own
    shared_kv_states, not the cache). Language models under transformers 5 and any with linear-attention layers;
    the pinned 4.57 families keep their measured behaviour."""
    import transformers

    layer_types = getattr(getattr(model, "config", None), "layer_types", None) or ()
    hybrid = "linear_attention" in layer_types
    if not (hybrid or int(transformers.__version__.split(".")[0]) >= 5) or not callable(getattr(model, "forward", None)) \
            or "use_cache" not in inspect.signature(model.forward).parameters or \
            getattr(model.forward, "_starvla_no_cache", False):
        return
    forward = model.forward

    def no_cache(*args, **kwargs):
        if torch.is_grad_enabled() and kwargs.get("cache") is None and kwargs.get("past_key_values") is None:
            kwargs["use_cache"] = False  # the multimodal model passes its config default (True) explicitly
        return forward(*args, **kwargs)

    no_cache._starvla_no_cache = True
    model.forward = no_cache


def _linear_attention(vlm):
    config = getattr(getattr(vlm, "model", None), "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        if "linear_attention" in (getattr(cfg, "layer_types", None) or ()):
            return True
    return False


def apply_lm(vlm):
    """Compile the decoder layers of the backbone's language model: `model.model.language_model.layers` in the
    Qwen-VL (Qwen3.5 included), Gemma 4 and MiniCPM-V (transformers 5) families; the other places are where other
    Hugging Face multimodal wrappers keep them. A layer Dynamo cannot trace (e.g. a linear-attention kernel of an
    extension) stays eager on its own; the others are compiled.

    Not applied to a language model with linear-attention layers (Qwen3.5: three Gated DeltaNet layers per full
    attention layer, run by the flash-linear-attention Triton kernels). Compiling it cost Qwen3.5-0.8B + OFT a first
    step of 934 s and was still slower once warm, 461 ms per step against 329 ms with the same paths and this one left
    out (B300, two GPUs, session 14); with the GR00T head the first step outlasted NCCL's 600 s timeout."""
    if _linear_attention(vlm):
        return False
    return _apply(vlm, _LM_PATHS, prepare=_causal_mask_for_none)


_LM_PATHS = ("model.model.language_model.layers", "model.language_model.model.layers",
             "model.language_model.layers",
             "model.llm.model.layers",  # MiniCPM-V with its own (remote) code
             "model.language_model.model.encoder.layers")  # Florence-2: starVLA runs the encoder only


_explicit = {"depth": 0}


def _causal_mask_for_none(layers, owner, path, vlm=None):
    """A causal language model under SDPA gets no mask (None) for a batch without padding: transformers then lets the
    kernel apply causality itself (is_causal). The compiled layers are specialised on a mask tensor, and the first
    batch of equal lengths compiled every layer again (Gemma 4 + PI at batch 4: step 161, nine graphs, 19 s, in the
    middle of training). During the language model's training forwards transformers' own builder is told that the
    mask cannot be skipped (masking_utils._ignore_causal_mask_sdpa), so such a batch gets the causal mask it builds
    for any other batch, with the same layout: the same attention through another kernel. A mask made outside the
    builder did not do: Dynamo also guards on the layout of the mask's base tensor. The setting holds for the whole
    forward of the interface's model: Gemma 4's multimodal model builds the masks before it calls its language model.
    Not for an encoder (Florence-2's), whose layers get no causal mask."""
    from transformers import masking_utils

    top = getattr(vlm, "model", None)
    owner = top if isinstance(top, torch.nn.Module) else owner
    if path.endswith("encoder.layers") or not hasattr(masking_utils, "_ignore_causal_mask_sdpa") \
            or not callable(getattr(owner, "forward", None)) or getattr(owner.forward, "_starvla_explicit_mask", False):
        return
    original = masking_utils._ignore_causal_mask_sdpa
    if not getattr(original, "_starvla_explicit_mask", False):
        def ignore_causal_mask_sdpa(*args, **kwargs):
            return False if _explicit["depth"] else original(*args, **kwargs)

        ignore_causal_mask_sdpa._starvla_explicit_mask = True
        masking_utils._ignore_causal_mask_sdpa = ignore_causal_mask_sdpa
    forward = owner.forward

    def explicit_mask(*args, **kwargs):
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        _explicit["depth"] += 1
        try:
            return forward(*args, **kwargs)
        finally:
            _explicit["depth"] -= 1

    explicit_mask._starvla_explicit_mask = True
    owner.forward = explicit_mask


def apply_vision(vlm):
    """Compile the blocks of the backbone's vision tower: `model.model.visual.blocks` in the Qwen-VL family (meant to
    be used with vision_attn, whose per-layer code has no host reads), the encoder layers of a SigLIP / CLIP style
    tower elsewhere. Not a tower whose attention reads the chunk lengths to the host in every layer and runs one call
    per chunk (MiniCPM-V 4.6 off flash-attention): every layer would break the graph there, and MiniCPM-RobotManip
    ran 1,693 against 1,548 ms per step with it."""
    from starVLA.fast.attention import _per_chunk

    model = getattr(vlm, "model", None)
    if isinstance(model, torch.nn.Module) and _per_chunk(model):
        return False
    return _apply(vlm, ("model.model.visual.blocks", "model.model.vision_tower.vision_model.encoder.layers",
                        "model.vision_tower.vision_model.encoder.layers",
                        "model.model.vision_tower.encoder.layers",  # Gemma 4, MiniCPM-V 4.6 / 4.7 (transformers 5)
                        "model.vpm.encoder.layers",  # MiniCPM-V with its own (remote) code
                        "model.vision_tower.blocks"))  # Florence-2's DaViT: one entry per stage


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
                    _compile_forwards(blocks, dynamic=True, options=HEAD_OPTIONS)
                    blocks._starvla_compiled = True
                found = True
    return found

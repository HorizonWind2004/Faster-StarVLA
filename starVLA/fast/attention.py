"""PyTorch's scaled-dot-product attention in place of flash-attention.

Both compute the same attention; they are different kernels, so results differ in rounding only. With
flash-attention the fast paths that are written for SDPA (attn_mask, vision_attn, the compiled layers) do not apply,
and on Qwen2.5-VL-3B the default code is faster with flash-attention while the accelerated code is faster with SDPA:
874 against 1,176 ms per step upstream, 652 against 466 ms with every fast path (QwenOFT, 2 x B300, batch 16).

transformers picks the kernel when a layer runs (ALL_ATTENTION_FUNCTIONS[config._attn_implementation]) and offers
`set_attn_implementation` to change it on a loaded model; a model without that is left alone.

Gemma 4 loads with neither flash-attention (its 256-wide heads) nor SDPA (its audio tower has none) in transformers
5.24, so its recipes run eager attention, which materializes every score matrix in fp32: on LIBERO's two views of
2,520 patches each the vision tower alone needs some 9 GiB per layer at batch 16. There the language model and the
vision tower are switched to SDPA and the audio tower keeps eager attention.

A vision tower whose attention off flash-attention reads the chunk lengths to the host in every layer and runs one
call per image slice and per 2 x 2 window (MiniCPM-V 4.6), and that vision_attn does not batch, runs flash-attention
instead when it is installed: kept there when the recipe loads it, moved there from SDPA or eager otherwise
(MiniCPM-RobotManip loads MiniCPM-V 4.6 with SDPA).
"""
from starVLA.fast import attention_implementation


def use_sdpa(vlm):
    model = getattr(vlm, "model", None)
    switch = getattr(model, "set_attn_implementation", None)
    if not callable(switch):
        return False
    source = _replaced(model)
    keep = _per_chunk(model)
    kept_configs = {id(m.config) for m in keep if getattr(m, "config", None) is not None}
    moved = _flash_towers(model, kept_configs)
    if source is None:
        return moved
    try:
        if source != "flash_attention_2" or keep:
            raise ValueError  # the whole model cannot (or should not) run SDPA: never switch it as a whole
        switch("sdpa")
    except (ValueError, ImportError):  # a sub-model without SDPA (Gemma 4's audio tower): switch the others one by one
        for module in model.modules():
            sub_switch = getattr(module, "set_attn_implementation", None)
            if module is not model and callable(sub_switch) and attention_implementation(module) == source and \
                    getattr(module, "_supports_sdpa", True) is not False and \
                    not any(m in keep for m in module.modules()):
                try:
                    sub_switch("sdpa")
                except (ValueError, ImportError):
                    pass  # keeps its implementation
    for module in model.modules():  # Qwen2.5-VL's text model keeps a copy
        if getattr(module, "_attn_implementation", None) == source and attention_implementation(module) == "sdpa":
            module._attn_implementation = "sdpa"
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if language_model is not None:  # the paths written for SDPA look at the language model and the vision tower
        return attention_implementation(language_model) == "sdpa"
    return source not in _implementations(model)


_PER_CHUNK = "torch.split(tensor, lengths.tolist()"  # transformers' vision attention without flash-attention


def _per_chunk(model):
    """Attention modules that, off flash-attention, read the chunk lengths back to the host in every layer and run
    one attention call per chunk (image slice, or window: MiniCPM-V 4.6 has one per 2 x 2 patches), and that
    vision_attn does not batch. Their sub-models keep flash-attention, whose single varlen call per layer is the
    faster form: switching MiniCPM-V 4.6's vision tower made MiniCPMGR00T 3x slower (570 -> 1,721 ms per step)."""
    import inspect

    from starVLA.fast.vision_attn import _FAMILIES

    found, seen = set(), {}
    for module in model.modules():
        cls = type(module)
        if cls not in seen:
            forward = getattr(cls.forward, "_starvla_original", cls.forward)
            try:
                source = inspect.getsource(forward)
            except (OSError, TypeError):
                source = ""
            seen[cls] = _PER_CHUNK in source and cls.__module__ not in _FAMILIES
        if seen[cls]:
            found.add(module)
    return found


def _flash_towers(model, kept_configs):
    """Move the per-chunk towers (the sub-models whose config the per-chunk attention modules read) that are not on
    flash-attention there, where the class supports it and flash-attention is installed. True if one moved."""
    moved = False
    for module in model.modules():
        sub_switch = getattr(module, "set_attn_implementation", None)
        if module is model or not callable(sub_switch) or id(getattr(module, "config", None)) not in kept_configs or \
                attention_implementation(module) == "flash_attention_2" or \
                not getattr(module, "_supports_flash_attn", False):
            continue
        try:
            sub_switch("flash_attention_2")
            moved = True
        except (ValueError, ImportError):
            pass  # flash-attention not installed: keeps its implementation
    return moved


def _replaced(model):
    """The implementation that SDPA replaces: flash-attention, or eager attention when the model runs eager only
    because one of its sub-models has no SDPA (Gemma 4: its audio tower; loading it with SDPA is refused, and so is
    flash-attention, whose head size limit its 256-wide heads exceed). An eager model that could load with SDPA was
    put on eager on purpose and is left alone."""
    implementations = _implementations(model)
    if "flash_attention_2" in implementations:
        return "flash_attention_2"
    without_sdpa = [module for module in model.modules() if getattr(module, "_supports_sdpa", None) is False and
                    hasattr(module, "set_attn_implementation")]
    if "eager" in implementations and without_sdpa and any(
            module is not model and getattr(module, "_supports_sdpa", None) is True and
            attention_implementation(module) == "eager" and not any(m in without_sdpa for m in module.modules())
            for module in model.modules()):
        return "eager"
    return None


def _implementations(model):
    return {attention_implementation(module) for module in model.modules()} - {None}

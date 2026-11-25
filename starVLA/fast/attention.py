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
"""
from starVLA.fast import attention_implementation


def use_sdpa(vlm):
    model = getattr(vlm, "model", None)
    switch = getattr(model, "set_attn_implementation", None)
    if not callable(switch):
        return False
    source = _replaced(model)
    if source is None:
        return False
    try:
        if source != "flash_attention_2":
            raise ValueError  # the whole model cannot run SDPA: never switch it as a whole
        switch("sdpa")
    except (ValueError, ImportError):  # a sub-model without SDPA (Gemma 4's audio tower): switch the others one by one
        for module in model.modules():
            sub_switch = getattr(module, "set_attn_implementation", None)
            if module is not model and callable(sub_switch) and attention_implementation(module) == source and \
                    getattr(module, "_supports_sdpa", True) is not False:
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

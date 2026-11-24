"""PyTorch's scaled-dot-product attention in place of flash-attention.

Both compute the same attention; they are different kernels, so results differ in rounding only. With
flash-attention the fast paths that are written for SDPA (attn_mask, vision_attn, the compiled layers) do not apply,
and on Qwen2.5-VL-3B the default code is faster with flash-attention while the accelerated code is faster with SDPA:
874 against 1,176 ms per step upstream, 652 against 466 ms with every fast path (QwenOFT, 2 x B300, batch 16).

transformers picks the kernel when a layer runs (ALL_ATTENTION_FUNCTIONS[config._attn_implementation]) and offers
`set_attn_implementation` to change it on a loaded model; a model without that is left alone.
"""
from starVLA.fast import attention_implementation


def use_sdpa(vlm):
    model = getattr(vlm, "model", None)
    switch = getattr(model, "set_attn_implementation", None)
    if not callable(switch) or "flash_attention_2" not in _implementations(model):
        return False
    try:
        switch("sdpa")
    except (ValueError, ImportError):  # a sub-model without SDPA (Gemma 4's audio tower): switch the others one by one
        for module in model.modules():
            sub_switch = getattr(module, "set_attn_implementation", None)
            if module is not model and callable(sub_switch) and attention_implementation(module) == "flash_attention_2":
                try:
                    sub_switch("sdpa")
                except (ValueError, ImportError):
                    pass  # keeps flash-attention
    for module in model.modules():  # Qwen2.5-VL's text model keeps a copy
        if getattr(module, "_attn_implementation", None) == "flash_attention_2" and \
                attention_implementation(module) != "flash_attention_2":
            module._attn_implementation = "sdpa"
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if language_model is not None:  # the paths written for SDPA look at the language model and the vision tower
        return attention_implementation(language_model) != "flash_attention_2"
    return "flash_attention_2" not in _implementations(model)


def _implementations(model):
    return {attention_implementation(module) for module in model.modules()} - {None}

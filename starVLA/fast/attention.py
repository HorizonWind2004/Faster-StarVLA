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
    switch("sdpa")
    for module in model.modules():  # Qwen2.5-VL's text model keeps a copy
        if getattr(module, "_attn_implementation", None) == "flash_attention_2":
            module._attn_implementation = "sdpa"
    return "flash_attention_2" not in _implementations(model)


def _implementations(model):
    return {attention_implementation(module) for module in model.modules()} - {None}

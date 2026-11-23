"""A direct formulation of transformers' SDPA attention-mask builder.

transformers (4.57) builds the 4-D boolean attention mask of a causal LM by applying a per-element predicate through
four nested torch.vmap calls (masking_utils.sdpa_mask_recent_torch). For the common case, a causal mask combined with
a 2-D padding mask, the result is simply

    mask[b, 0, q, k] = (k_index <= q_index) & padding_mask[b, k_index]

Building it with two broadcasted tensor ops gives the same boolean tensor; the vmap formulation costs about 150 ms of
host time per call at batch 16 x 205 tokens, while the GPU waits. `install()` registers the direct builder for the
"sdpa" attention implementation through transformers' own registry (AttentionMaskInterface.register); every case it
does not cover (extra and/or mask functions, packed sequences, local attention) is delegated to the original.

Applies to every backbone whose transformers model builds its mask with masking_utils.create_causal_mask (the Qwen,
Gemma, Llama and Phi families in transformers 4.57), whatever the framework on top.
"""
import sys

import torch
from transformers import masking_utils

_installed = False


def install():
    global _installed
    if _installed:
        return
    original = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0,
                  mask_function=masking_utils.causal_mask_function, attention_mask=None, local_size=None,
                  allow_is_causal_skip=True, **kwargs):
        if mask_function is not masking_utils.causal_mask_function or local_size is not None:
            return original(batch_size=batch_size, cache_position=cache_position, kv_length=kv_length,
                            kv_offset=kv_offset, mask_function=mask_function, attention_mask=attention_mask,
                            local_size=local_size, allow_is_causal_skip=allow_is_causal_skip, **kwargs)
        q_length = cache_position.shape[0]
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        if allow_is_causal_skip and masking_utils._ignore_causal_mask_sdpa(
                padding_mask, q_length, kv_length, kv_offset, local_size):
            return None
        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset
        mask = (kv_arange[None, :] <= cache_position[:, None])[None, None, :, :]      # causal_mask_function
        if padding_mask is not None:
            mask = mask & padding_mask[:, kv_arange][:, None, None, :].to(mask.device)  # padding_mask_function
        return mask.expand(batch_size, 1, q_length, kv_length).contiguous()

    sdpa_mask._starvla_original = original
    masking_utils.AttentionMaskInterface.register("sdpa", sdpa_mask)
    _installed = True


def apply(vlm):
    """Register the direct builder if the backbone's transformers model uses masking_utils."""
    modeling = sys.modules.get(type(getattr(vlm, "model", None)).__module__)
    if not hasattr(modeling, "create_causal_mask") or "sdpa" not in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS:
        return False
    install()
    return True

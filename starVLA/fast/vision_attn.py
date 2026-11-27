"""vision_attn: one batched attention call per vision layer instead of one per image, for the Qwen vision towers.

The implementation is `batch_vision_attention` in starVLA/model/modules/vlm/monkey_patch.py (described there): a
layout is batched only if a probe on the running device shows the batched call is bitwise identical, forward and
backward. This module is its entry in the fast-path registry. `vision_attn_relaxed` (not an exact fast path) batches
every uniform layout, without the probe.
"""
from starVLA.model.modules.vlm import monkey_patch

_FAMILIES = monkey_patch._ATTENTION_FAMILIES  # transformers modeling module -> (attention, vision model, windowed)
decisions = monkey_patch.decisions  # (chunks, length, heads, head_dim, dtype, device, with gradients) -> bitwise


def _every_uniform_layout(layout, heads, head_dim, dtype, device):
    return layout


def apply(vlm):
    """Backbone hook: batch the vision attention if the backbone's vision tower is one of the Qwen families."""
    return monkey_patch.batch_vision_attention(vlm)


def apply_relaxed(vlm):
    """vision_attn for every uniform layout, without the bitwise probe (process-wide, like the patch itself)."""
    if not apply(vlm):
        return False
    monkey_patch._bitwise = _every_uniform_layout
    return True

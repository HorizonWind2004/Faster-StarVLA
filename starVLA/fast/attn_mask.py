"""attn_mask: the causal + padding attention mask from two tensor ops instead of nested torch.vmap.

The implementation is `patch_attention_mask` in starVLA/model/modules/vlm/monkey_patch.py (described there); this
module is its entry in the fast-path registry.
"""
from starVLA.model.modules.vlm import monkey_patch


def apply(vlm):
    """Backbone hook: register the direct builder if the backbone's transformers model uses masking_utils."""
    return monkey_patch.patch_attention_mask(vlm)

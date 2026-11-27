"""vision_pos: the vision position tables built once per grid size instead of in every forward.

The implementation is `cache_vision_positions` in starVLA/model/modules/vlm/monkey_patch.py (described there); this
module is its entry in the fast-path registry.
"""
from starVLA.model.modules.vlm import monkey_patch

_memoized_helper = monkey_patch._memoized_helper


def apply(vlm):
    """Backbone hook: memoize the position tables of a Qwen vision tower (transformers 4), or of a tower that builds
    them with the helpers of transformers.vision_utils (transformers 5)."""
    return monkey_patch.cache_vision_positions(vlm)

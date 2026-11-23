"""Project the action head's conditioning once per sample instead of once per repeated copy.

The flow-matching frameworks train several noise levels per sample: they repeat the VLM's hidden states R times
(`last_hidden.repeat(repeated_diffusion_steps, 1, 1)`, R = 8 for the GR00T head) and hand the R x batch rows to the
action DiT. Every cross-attention layer then projects all of them to keys and values, in fp32: R identical copies
of the largest matrix products of the head (about 150 conditioning tokens of the VLM's width, against a few dozen
action tokens). Here the copies are detected, the two projections run on the first copy, and the result is repeated.

The values are the same; a gradient is summed over the copies in a different order, so results differ in rounding.
Nothing is assumed about the caller: the DiT's conditioning is compared with its first rows in every training call
(two tensor comparisons), and a call whose conditioning is not R copies of the same rows runs as before.
"""
import torch

_copies = None  # (rows, repeats) while a DiT forward with repeated conditioning runs


def _repeated(states):
    """(rows, repeats) when every tensor of `states` is `repeats` copies of its first `rows` rows, else None."""
    first = states[0]
    total = first.shape[0]
    like_row_0 = (first == first[:1]).flatten(1).all(dim=1).nonzero().flatten()
    if like_row_0.numel() < 2:
        return None
    rows = int(like_row_0[1])
    if total % rows:
        return None
    repeats = total // rows
    same = [(state.unflatten(0, (repeats, rows)) == state[:rows]).all() for state in states if state.shape[0] == total]
    if len(same) != len(states) or not bool(torch.stack(same).all()):
        return None
    return rows, repeats


def _first_copy(module, args):  # forward pre-hook of a cross-attention's key / value projection
    if _copies is not None and args[0].shape[0] == _copies[0] * _copies[1]:
        return (args[0][:_copies[0]],)


def _all_copies(module, args, output):
    if _copies is not None and output.shape[0] == _copies[0]:
        return output.repeat(_copies[1], *([1] * (output.dim() - 1)))


def apply(target):
    """Patch the action DiT (class-wide; `target` is not used). False when this build has no such head."""
    global _copies
    try:
        from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
    except ImportError:
        return False
    if getattr(DiT.forward, "_starvla_condition_once", False):
        return True
    original = DiT.forward

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        global _copies
        if not self.training or encoder_hidden_states is None:
            return original(self, hidden_states, encoder_hidden_states, *args, **kwargs)
        if not getattr(self, "_starvla_condition_once", False):
            for block in self.transformer_blocks:
                attention = getattr(block, "attn1", None)
                if getattr(attention, "is_cross_attention", False):
                    for projection in (attention.to_k, attention.to_v):
                        projection.register_forward_pre_hook(_first_copy)
                        projection.register_forward_hook(_all_copies)
            self._starvla_condition_once = True
        states = encoder_hidden_states if isinstance(encoder_hidden_states, (list, tuple)) else [encoder_hidden_states]
        _copies = _repeated(states)
        try:
            return original(self, hidden_states, encoder_hidden_states, *args, **kwargs)
        finally:
            _copies = None

    forward._starvla_condition_once = True
    DiT.forward = forward
    return True

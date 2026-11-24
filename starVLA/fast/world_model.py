"""The Wan backbone's text encoder on the prompt's tokens instead of 512 padded positions.

Every training step the Wan interface encodes the batch's instructions with UMT5-XXL (4.7 B parameters), padded to a
fixed 512 tokens, and then keeps only each instruction's own tokens and replaces every other position with zeros
(Wan's text conditioning format). A LIBERO instruction has about 20 tokens: nearly all of the encoder's work is on
padding whose result is thrown away.

Here the instructions are padded to the longest one of the batch instead, and the result is padded back to 512 with
zeros exactly as upstream does. The padding positions are masked keys in every attention layer (their weight after
the softmax is exactly zero) and UMT5's relative position bias depends on the distance between two tokens only, so
every kept position has the same value as upstream's up to the order of the sums over the sequence: a rounding-level
change, not an exact one.

Applies only where the interface's encoder is the code described above (checked on its source); Cosmos-Predict2
keeps its padded T5 outputs, which depend on the padding length, and is left alone.
"""
import inspect

import torch

_UPSTREAM_MARKERS = ('padding="max_length"', "embeds[:seq_len]", "new_zeros(max_length - embeds.size(0)")


def _zero_padded_upstream(backbone):
    encode = getattr(type(backbone), "_encode_text", None)
    if encode is None or getattr(encode, "_starvla_text_longest", False):
        return encode is not None
    try:
        source = inspect.getsource(encode)
    except (OSError, TypeError):
        return False
    return all(marker in source for marker in _UPSTREAM_MARKERS)


def text_longest(backbone):
    """Backbone hook: True if the backbone is Wan's interface and its text encoding was replaced."""
    if not all(hasattr(backbone, name) for name in ("tokenizer", "text_encoder")) or not _zero_padded_upstream(backbone):
        return False
    if getattr(backbone._encode_text, "_starvla_text_longest", False):
        return True

    def _encode_text(instructions, max_length=512):
        device = next(backbone.text_encoder.parameters()).device
        text_inputs = backbone.tokenizer(
            instructions,
            padding="longest",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = text_inputs.attention_mask
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            text_embeds = backbone.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        text_embeds = [embeds[:seq_len] for embeds, seq_len in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([embeds, embeds.new_zeros(max_length - embeds.size(0), embeds.size(1))]) for embeds in text_embeds],
            dim=0,
        )
        return text_embeds.to(dtype=torch.bfloat16)

    _encode_text._starvla_text_longest = True
    backbone._encode_text = _encode_text
    return True

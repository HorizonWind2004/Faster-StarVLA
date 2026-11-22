"""Batched attention for the Qwen3-VL vision tower when every image has the same number of patches.

With any attention implementation other than flash_attention_2, transformers' Qwen3VLVisionAttention.forward
(4.57) reads the chunk lengths back to the host in every layer (`lengths.tolist()`, a GPU sync per layer) and then
runs scaled_dot_product_attention once per image, followed by a concatenation. A training batch of 16 samples x 2
views is 32 images: 24 layers x 32 attention calls per step, and the GPU waits for the host between all of them.

When all chunks have the same length (the usual case in training, where every view is resized to one resolution) the
per-image calls are one batched call: the patches of image i are rows i*L:(i+1)*L, so [1, heads, n*L, d] is viewed
as [n, heads, L, d]. Attention never mixes images either way. Chunk lengths are read once per vision forward (the
first layer) and reused by the other layers. Unequal chunks, flash_attention_2 and eager attention use the original
implementation.
"""
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3

_installed = False


def _uniform_chunks(cu_seqlens):
    """(number of chunks, chunk length) when all chunks are equal, else None. One host read per cu_seqlens tensor."""
    cached = _uniform_chunks.cache
    if cached is not None and cached[0] is cu_seqlens:
        return cached[1]
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    result = (len(lengths), lengths[0]) if lengths and all(x == lengths[0] for x in lengths) else None
    _uniform_chunks.cache = (cu_seqlens, result)
    return result


_uniform_chunks.cache = None


def install():
    global _installed
    if _installed:
        return
    original = qwen3.Qwen3VLVisionAttention.forward

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        impl = self.config._attn_implementation
        uniform = None if impl in ("eager", "flash_attention_2") else _uniform_chunks(cu_seqlens)
        if uniform is None:
            return original(self, hidden_states, cu_seqlens, rotary_pos_emb=rotary_pos_emb,
                            position_embeddings=position_embeddings, **kwargs)
        chunks, length = uniform
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = qwen3.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [seq, heads, d] -> [chunks, heads, length, d]: one batch element per image
        def per_image(states):
            return states.reshape(chunks, length, self.num_heads, -1).transpose(1, 2)

        attention_interface = qwen3.ALL_ATTENTION_FUNCTIONS[impl]
        attn_output = attention_interface(
            self,
            per_image(query_states),
            per_image(key_states),
            per_image(value_states),
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]  # [chunks, length, heads, d]
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    forward._starvla_original = original
    qwen3.Qwen3VLVisionAttention.forward = forward
    _installed = True

"""Batched attention for the Qwen vision towers when every image has the same number of patches.

The vision attention of Qwen2-VL, Qwen2.5-VL and Qwen3-VL (the same code in transformers 4.57) receives all patches
of all images as one sequence plus the chunk boundaries (cu_seqlens). With any implementation but flash-attention it
reads the chunk lengths back to the host in every layer (`lengths.tolist()`: a GPU sync per layer) and then runs one
attention call per chunk: 24 layers x 32 images per training step at batch 16 x 2 views.

When all chunks have the same length (the usual case in training, where every view is resized to one resolution) the
chunks are a batch: one attention call per layer on [chunks, heads, length, dim], and no host read in the layers.
The layout is derived once per vision forward from grid_thw on the host. Qwen2.5-VL alternates between full
attention (one chunk per frame) and window attention (one chunk per window); each kind is batched when it is
uniform. Any other case falls back to the upstream code. Same outputs and gradients, bit for bit.
"""
import sys

_FAMILIES = {  # transformers modeling module -> (attention class, vision model class, has window attention)
    "transformers.models.qwen3_vl.modeling_qwen3_vl": ("Qwen3VLVisionAttention", "Qwen3VLVisionModel", False),
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl": (
        "Qwen2_5_VLVisionAttention", "Qwen2_5_VisionTransformerPretrainedModel", True),
    "transformers.models.qwen2_vl.modeling_qwen2_vl": ("VisionAttention", "Qwen2VisionTransformerPretrainedModel", False),
}
_installed = set()
_MAX_ENTRIES = 16


class _State:
    """Chunk layouts of the vision forward in progress, each (number of chunks, chunk length) or None when the
    chunks differ: `frames` for full attention, `windows` for window attention. Set once per vision forward."""

    frames = None
    windows = None
    windowed = False


def _uniform(lengths):
    if lengths and all(x == lengths[0] for x in lengths):
        return (len(lengths), lengths[0])
    return None


def _frame_layout(rows):
    return _uniform([h * w for t, h, w in rows for _ in range(t)])  # cu_seqlens: one chunk of h*w patches per frame


def _window_layout(vision_model, grid_thw):
    bounds = vision_model.get_window_index(grid_thw)[1]  # cumulative window ends, as the forward computes them
    bounds = [b for i, b in enumerate(bounds) if i == 0 or b != bounds[i - 1]]  # torch.unique_consecutive
    return _uniform([b - a for a, b in zip(bounds, bounds[1:])])


def _layout(cu_seqlens):
    if not _State.windowed:
        return _State.frames
    chunks = cu_seqlens.shape[0] - 1
    for layout in (_State.frames, _State.windows):
        if layout is not None and layout[0] == chunks:
            return layout
    return None


def install(modeling):
    """Patch the vision attention and vision model classes of one transformers modeling module."""
    if modeling.__name__ in _installed:
        return
    attention_name, model_name, windowed = _FAMILIES[modeling.__name__]
    attention_class, model_class = getattr(modeling, attention_name), getattr(modeling, model_name)
    original = attention_class.forward
    original_model_forward = model_class.forward
    layouts = {}  # (model, grid) -> (frames, windows)

    def model_forward(self, hidden_states, grid_thw, **kwargs):
        rows = tuple(tuple(row) for row in grid_thw.tolist())  # one host read per forward
        key = (id(self), rows)
        if key not in layouts:
            if len(layouts) >= _MAX_ENTRIES:
                layouts.pop(next(iter(layouts)))
            layouts[key] = (_frame_layout(rows), _window_layout(self, grid_thw) if windowed else None)
        _State.frames, _State.windows = layouts[key]
        _State.windowed = windowed
        try:
            return original_model_forward(self, hidden_states, grid_thw, **kwargs)
        finally:
            _State.frames = _State.windows = None
            _State.windowed = False

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        impl = self.config._attn_implementation
        uniform = None if impl in ("eager", "flash_attention_2") else _layout(cu_seqlens)
        if uniform is None:
            return original(self, hidden_states, cu_seqlens, rotary_pos_emb=rotary_pos_emb,
                            position_embeddings=position_embeddings, **kwargs)
        chunks, length = uniform
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = modeling.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [seq, heads, d] -> [chunks, heads, length, d]: one batch element per chunk
        def per_chunk(states):
            return states.reshape(chunks, length, self.num_heads, -1).transpose(1, 2)

        attention_interface = modeling.ALL_ATTENTION_FUNCTIONS[impl]
        attn_output = attention_interface(
            self,
            per_chunk(query_states),
            per_chunk(key_states),
            per_chunk(value_states),
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]  # [chunks, length, heads, d]
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    forward._starvla_original = original
    attention_class.forward = forward
    model_class.forward = model_forward
    _installed.add(modeling.__name__)


def apply(vlm):
    """Batch the vision attention if the backbone's vision tower is one of the Qwen families."""
    visual = getattr(getattr(getattr(vlm, "model", None), "model", None), "visual", None)
    family = type(visual).__module__
    if visual is None or family not in _FAMILIES:
        return False
    install(sys.modules[family])
    return True

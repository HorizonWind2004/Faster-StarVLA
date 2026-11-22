"""Memoized position indices for the Qwen3-VL vision tower.

Qwen3VLVisionModel computes, in every forward and with a Python loop over the images that reads GPU scalars back to
the host, two things that depend only on the grid sizes (grid_thw), not on any weight:

  rot_pos_emb(grid_thw)                 the rotary position table of every patch (a lookup in a fixed buffer)
  fast_pos_embed_interpolate(grid_thw)  the four neighbour indices and bilinear weights of every patch

In training every batch has the same grids (all views are resized to one resolution), so both are built once per
distinct grid_thw and reused: the rotary table as a whole, and the indices / weights of the interpolation. The part
of fast_pos_embed_interpolate that touches the trainable position embedding (lookup, weighting, the sum of the four
neighbours, the per-image reordering) runs every time exactly as upstream, so its values and gradients are the
upstream ones. About 27 ms of host time per step at 32 images.
"""
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3

_installed = False
_MAX_ENTRIES = 16


def _grid_key(grid_thw):
    return tuple(tuple(row) for row in grid_thw.tolist())  # one host read per forward


def _remember(cache, key, value):
    if len(cache) >= _MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def install():
    global _installed
    if _installed:
        return
    original_rot = qwen3.Qwen3VLVisionModel.rot_pos_emb
    rot_cache, interp_cache = {}, {}

    def rot_pos_emb(self, grid_thw):
        inv_freq = self.rotary_pos_emb.inv_freq
        key = (_grid_key(grid_thw), str(inv_freq.device), str(inv_freq.dtype), id(self))
        hit = rot_cache.get(key)
        if hit is None:
            hit = _remember(rot_cache, key, original_rot(self, grid_thw).detach())
        return hit

    def interp_indices(self, grid):
        """idx_tensor and weight_tensor of the upstream function, built from host integers (same arithmetic)."""
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        return idx_tensor, weight_tensor

    def fast_pos_embed_interpolate(self, grid_thw):
        grid = _grid_key(grid_thw)
        weight = self.pos_embed.weight
        key = (grid, str(weight.device), str(weight.dtype), self.num_grid_per_side)
        hit = interp_cache.get(key)
        if hit is None:
            hit = _remember(interp_cache, key, interp_indices(self, grid))
        idx_tensor, weight_tensor = hit

        # from here on: upstream, with host integers for the grid sizes
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for _, h, w in grid])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    rot_pos_emb._starvla_original = original_rot
    fast_pos_embed_interpolate._starvla_original = qwen3.Qwen3VLVisionModel.fast_pos_embed_interpolate
    qwen3.Qwen3VLVisionModel.rot_pos_emb = rot_pos_emb
    qwen3.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    _installed = True

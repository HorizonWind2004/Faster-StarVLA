"""TF32 for fp32 matrix products (level "max", or by name: STARVLA_FAST=all,tf32).

The action heads of the flow-matching frameworks (QwenGR00T, QwenPI, ...) run in fp32 while the backbone runs in bf16
autocast. PyTorch's default computes fp32 matrix products at full precision; with TF32 the GPU rounds the inputs of
each product to 10 mantissa bits and accumulates in fp32, which is several times faster on tensor cores. This is a
precision setting, not a reordering: results differ from the default in more than the last bit. bf16, which the
backbone already uses, keeps 8 mantissa bits.
"""


def apply_tf32(target):
    import torch

    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True

"""Fast paths for the OpenPI frameworks (PI0, PI05). They do not use a `get_vlm_model` backbone, so each of these is
a framework-hook path that recognises them by what it patches.

worker_examples: the framework's `_prepare_examples` (every view resized and padded to the model's resolution and
scaled, missing views filled in, state and actions padded: CPU work on the batch as the data set returns it) runs in
the data-loader workers, on a stand-in for the framework that has its settings and methods but no model. `apply`
checks once that the stand-in returns exactly what the framework returns. In the training process the framework
gets the worker's tensors for a batch whose images, prompts, states and actions are the objects the worker saw;
any other call (inference, an edited batch) is prepared as before. Bitwise identical.

unused_views: these models take a fixed set of camera views (base, left wrist, right wrist). A dataset with fewer
cameras gets an all-black image for each missing view; the image still goes through the vision tower and its 256
tokens stay in the sequence, masked out of every attention (30% of the tokens of a LIBERO sample, which has two
cameras). When a view is inactive for the whole batch, it is left out instead. No token that is left attends to a
masked token, and position ids count active tokens only (cumsum of the padding mask), so the outputs at the
remaining positions are the same function of the same operands. The sequence is shorter, so sums run in a different
order: results differ in rounding.

openpi_one_pass: the camera views of a batch go through the SigLIP tower in one call instead of one call per view.
The same images through the same layers; a larger batch per kernel, so results can differ in rounding.

openpi_sdpa: the SigLIP tower and the joint PaliGemma / action-expert layers compute attention as matmul, softmax
in fp32, matmul, which materialises a [batch, heads, tokens, tokens] weight tensor twice per layer.
`scaled_dot_product_attention` computes the same attention (same mask, same scale) in one fused kernel. The softmax
then runs in the activations' dtype instead of fp32, so results differ in rounding.

openpi_compile: Inductor compiles the SigLIP encoder layers and the framework's joint layer function (one graph per
layer index) for training calls, as compile_lm / compile_vision do for the Qwen backbones. Same operations, fused
kernels: results differ in rounding.
"""
import types

import torch

# ---------------------------------------------------------------------------------------------- worker_examples
# The implementation is `prepare_examples_in_workers` in starVLA/dataloader/worker_prep.py (described there); these
# are its entries in the fast-path registry.
from starVLA.dataloader import worker_prep as _worker_prep
from starVLA.dataloader.worker_prep import example_stats as stats  # noqa: F401


def worker_examples(framework):
    """Framework hook: check that workers can run this framework's `_prepare_examples`, and let it return theirs."""
    return _worker_prep.prepare_examples_in_workers(framework)


def collate(cfg, collate_fn):
    """Data hook: the collate function for the workers; `collate_fn` itself when no framework was set up, or when
    this DataLoader is for another model."""
    return _worker_prep.examples_collate(cfg, collate_fn)


# ---------------------------------------------------------------------------------------------- openpi_one_pass


def one_pass(framework):
    embed_prefix = getattr(framework, "embed_prefix", None)
    vlm = getattr(framework, "vlm", None)
    if not callable(embed_prefix) or not callable(getattr(vlm, "embed_image", None)):
        return False
    if getattr(embed_prefix, "_starvla_one_pass", False):
        return True

    def embed_views_together(self, images, img_masks, lang_tokens, lang_masks):
        images = list(images)
        if len(images) < 2 or any(image.shape != images[0].shape for image in images):
            return embed_prefix(images, img_masks, lang_tokens, lang_masks)
        together = vlm.embed_image(torch.cat(images)).split(images[0].shape[0])
        embedded = {id(image): part for image, part in zip(images, together)}
        vlm.embed_image = lambda image: embedded[id(image)]  # what embed_prefix asks for, view by view
        try:
            return embed_prefix(images, img_masks, lang_tokens, lang_masks)
        finally:
            del vlm.embed_image

    embed_views_together._starvla_one_pass = True
    framework.embed_prefix = types.MethodType(embed_views_together, framework)
    return True


# ---------------------------------------------------------------------------------------------- unused_views


def unused_views(framework):
    embed_prefix = getattr(framework, "embed_prefix", None)
    if not callable(embed_prefix) or not hasattr(framework, "image_keys"):
        return False
    if getattr(embed_prefix, "_starvla_unused_views", False):
        return True

    def embed_used_views(self, images, img_masks, lang_tokens, lang_masks):
        used = [index for index, mask in enumerate(img_masks) if bool(mask.any())]
        if len(used) < len(img_masks):
            images, img_masks = [images[index] for index in used], [img_masks[index] for index in used]
        return embed_prefix(images, img_masks, lang_tokens, lang_masks)

    embed_used_views._starvla_unused_views = True
    framework.embed_prefix = types.MethodType(embed_used_views, framework)
    return True


# ---------------------------------------------------------------------------------------------- openpi_sdpa


def _sdpa_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """The signature and result of the eager attention functions of the OpenPI Gemma and SigLIP modules."""
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    if attention_mask is not None:  # additive, 0 or a large negative number
        attention_mask = attention_mask[:, :, :, : key.shape[-2]].to(query.dtype)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout if module.training else 0.0, scale=scaling)
    return output.transpose(1, 2).contiguous(), None


def use_sdpa(framework):
    """The framework's forward and predict_action run with the fused attention in place of the eager function of
    the OpenPI Gemma and SigLIP modules; outside these calls the modules are as they were."""
    if not callable(getattr(framework, "forward_shared_gemma_layer", None)):
        return False
    try:
        from starVLA.model.modules.vlm.openpi_transformers.gemma import modeling_gemma
        from starVLA.model.modules.vlm.openpi_transformers.siglip import modeling_siglip
    except ImportError:
        return False
    modules = (modeling_gemma, modeling_siglip)
    if not all(callable(getattr(module, "eager_attention_forward", None)) for module in modules):
        return False

    def fused(method):
        def call(*args, **kwargs):
            saved = [module.eager_attention_forward for module in modules]
            for module in modules:
                module.eager_attention_forward = _sdpa_attention
            try:
                return method(*args, **kwargs)
            finally:
                for module, function in zip(modules, saved):
                    module.eager_attention_forward = function

        call._starvla_openpi_sdpa = True
        return call

    for name in ("forward", "predict_action"):
        method = getattr(framework, name, None)
        if callable(method) and not getattr(method, "_starvla_openpi_sdpa", False):
            setattr(framework, name, fused(method))
    return True


# ---------------------------------------------------------------------------------------------- openpi_compile


def compile_layers(framework):
    joint = getattr(framework, "forward_shared_gemma_layer", None)
    language_model = getattr(getattr(framework, "vlm", None), "language_model", None)
    if not callable(joint) or not hasattr(language_model, "layers"):
        return False
    if getattr(framework, "_starvla_openpi_compiled", False):
        return True
    from starVLA.fast import compile as regional

    vision_model = getattr(getattr(framework.vlm, "vision_tower", None), "vision_model", None)
    vision_layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
    if vision_layers is not None and len(vision_layers):
        regional._compile_forwards(vision_layers, dynamic=True)
    regional.compile_method(framework, "forward_shared_gemma_layer", variants=len(language_model.layers))
    framework._starvla_openpi_compiled = True
    return True

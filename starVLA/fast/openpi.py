"""Fast paths for the OpenPI frameworks (PI0, PI05).

unused_views: these models take a fixed set of camera views (base, left wrist, right wrist). A dataset with fewer
cameras gets an all-black image for each missing view; the image still goes through the vision tower and its 256
tokens stay in the sequence, masked out of every attention (30% of the tokens of a LIBERO sample, which has two
cameras). When a view is inactive for the whole batch, it is left out instead.

No token that is left attends to a masked token, and position ids count active tokens only (cumsum of the padding
mask), so the outputs at the remaining positions are the same function of the same operands. The sequence is
shorter, so sums run in a different order: results differ in rounding.
"""
import types


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

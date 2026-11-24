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

openpi_sdpa: the SigLIP tower and the joint PaliGemma / action-expert layers compute attention as matmul, softmax
in fp32, matmul, which materialises a [batch, heads, tokens, tokens] weight tensor twice per layer.
`scaled_dot_product_attention` computes the same attention (same mask, same scale) in one fused kernel. The softmax
then runs in the activations' dtype instead of fp32, so results differ in rounding.
"""
import threading
import types
import weakref

import torch

# ---------------------------------------------------------------------------------------------- worker_examples

_SETTINGS = (bool, int, float, str, tuple)
_arrived = weakref.WeakValueDictionary()  # id(first image of a batch) -> its PreparedExamples, where it is consumed
_lock = threading.Lock()  # batches arrive in the DataLoader's pinning thread
stats = {"used": 0}
_stand_in = None  # of the framework most recently accelerated in this process


class StandIn:
    """A framework's plain settings and its class's methods, without the model: enough for `_prepare_examples`."""

    def __init__(self, framework):
        from omegaconf import OmegaConf

        self.framework_class = type(framework)
        self.settings = {}
        for name, value in vars(framework).items():
            if name.startswith("_"):
                continue
            inner = vars(value).get("_cfg") if hasattr(value, "__dict__") else None  # the trainer's access-tracking wrapper
            if OmegaConf.is_config(inner):
                self.settings[name] = inner
            elif isinstance(value, _SETTINGS) or OmegaConf.is_config(value):
                self.settings[name] = value

    def __getattr__(self, name):
        state = self.__dict__
        if "settings" not in state:  # while unpickling
            raise AttributeError(name)
        if name in state["settings"]:
            return state["settings"][name]
        method = getattr(state["framework_class"], name)
        if not isinstance(method, types.FunctionType):
            raise AttributeError(name)
        return types.MethodType(method, self)


def _parts(example):
    """The objects of a sample that `_prepare_examples` reads."""
    images = example["image"]
    images = tuple(images) if isinstance(images, (list, tuple)) else (images,)
    return images, example.get("lang", example.get("language")), example.get("state"), example.get("action"), \
        example.get("image_mask")


def _same_parts(mine, theirs):
    (images, lang, *arrays), (their_images, their_lang, *their_arrays) = mine, theirs
    return (len(images) == len(their_images) and all(a is b for a, b in zip(images, their_images))
            and lang == their_lang and all(a is b for a, b in zip(arrays, their_arrays)))


class PreparedExamples:
    """What a worker's `_prepare_examples` returned for one batch. It travels as batch[0]["_prepared_examples"]."""

    def __init__(self, batch, tensors):
        self.parts = [_parts(example) for example in batch]  # the same objects as in the batch, also after pickling
        self.tensors = tensors
        self._announce()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._announce()

    def _announce(self):
        with _lock:
            _arrived[id(self.parts[0][0][0])] = self

    def pin_memory(self):  # called by the DataLoader's pinning thread
        self.tensors = {key: value.pin_memory() if torch.is_tensor(value) else value for key, value in self.tensors.items()}
        return self


class ExamplesCollate:
    """`collate_fn`, followed by the framework's `_prepare_examples` for the batch it returns."""

    def __init__(self, collate_fn, stand_in):
        self.collate_fn, self.stand_in = collate_fn, stand_in

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if isinstance(batch, list) and batch and isinstance(batch[0], dict) and "image" in batch[0] and "action" in batch[0]:
            batch[0]["_prepared_examples"] = PreparedExamples(batch, self.stand_in._prepare_examples(batch, True))
        return batch


def _take(examples):
    try:
        with _lock:
            prepared = _arrived.get(id(_parts(examples[0])[0][0]))
        if prepared is not None and len(examples) == len(prepared.parts) and all(
                _same_parts(mine, _parts(example)) for mine, example in zip(prepared.parts, examples)):
            stats["used"] += 1
            if stats["used"] == 1:
                print("[starVLA.fast] worker_examples: training batches arrive prepared by the data-loader workers", flush=True)
            return dict(prepared.tensors)
    except (IndexError, KeyError, TypeError):
        pass
    return None


def _probe(framework, stand_in):
    import pickle

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    examples = [{"image": [Image.fromarray(rng.integers(0, 256, (200 + 24 * views, 224, 3), dtype=np.uint8)) for _ in range(views)],
                 "lang": "pick up the block", "state": rng.uniform(-1, 1, (1, 5)).astype(np.float32),
                 "action": rng.uniform(-1, 1, (min(3, framework.action_horizon), min(5, framework.action_dim))).astype(np.float32)}
                for views in (1, 2)]
    theirs = framework._prepare_examples(examples, True)
    ours = pickle.loads(pickle.dumps(stand_in))._prepare_examples(examples, True)
    return theirs.keys() == ours.keys() and all(
        torch.equal(theirs[key], ours[key]) and theirs[key].dtype == ours[key].dtype if torch.is_tensor(theirs[key])
        else theirs[key] == ours[key] for key in theirs)


def worker_examples(framework):
    """Framework hook: check that workers can run this framework's `_prepare_examples`, and let it return theirs."""
    global _stand_in
    original = getattr(framework, "_prepare_examples", None)
    if not callable(original):
        return False
    if getattr(original, "_starvla_worker_examples", False):
        return True
    try:
        stand_in = StandIn(framework)
        if not _probe(framework, stand_in):
            return False
    except Exception as error:  # a framework that prepares its samples differently: leave it alone
        print(f"[starVLA.fast] worker_examples: {type(framework).__name__} not supported ({type(error).__name__}: {error})",
              flush=True)
        return False

    def _prepare_examples(self, examples, include_actions):
        prepared = _take(examples) if include_actions else None
        return prepared if prepared is not None else original(examples, include_actions)

    _prepare_examples._starvla_worker_examples = True
    framework._prepare_examples = types.MethodType(_prepare_examples, framework)
    _stand_in = stand_in
    return True


def collate(cfg, collate_fn):
    """Data hook: the collate function for the workers; `collate_fn` itself when no framework was set up."""
    return collate_fn if _stand_in is None else ExamplesCollate(collate_fn, _stand_in)


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

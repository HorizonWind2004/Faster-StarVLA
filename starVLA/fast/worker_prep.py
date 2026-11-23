"""Build the VLM inputs of a training batch in the data-loader workers.

A framework's forward() hands the images and prompts of a batch to its backbone interface:

    qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

That call is the VLM's processor: chat template, tokenization, image resize / normalization / patchification. It is
CPU work (about 0.19 s per step at batch 16 x 2 images) done in the training process while the GPU waits. The
data-loader workers have the samples first and spare time, so they make the same call and send the result along
with the batch. In the training process the interface then returns what the worker built.

No framework and no interface contains code for this:

  * the worker runs the interface's own `build_qwenvl_inputs`, on a stand-in for the interface that has the
    processor and the config but no model. `apply` checks once, on a small batch, that the stand-in returns exactly
    what the interface returns; an interface whose input construction needs more than that is left alone.
  * the worker's result is used only for a call with the very image objects of that batch and the very prompts the
    worker used. Any other call (inference, a framework that edits the images, a prompt the workers did not expect)
    is answered by the interface as before.

The one thing a worker cannot see is how a framework edits the prompt before the call. Most frameworks pass the
sample's "lang" unchanged, which is the default; the others register their edit with @prompt("<framework name>")
below. A missing or wrong registration costs the speedup, never correctness, and is reported once.

Workers also compute the multimodal rotary position ids (Qwen2.5-VL, Qwen3-VL), which the model otherwise derives
at the start of every forward with a per-sample host loop.
"""
import pickle
import threading
import weakref
from types import SimpleNamespace

import torch

from starVLA.model.framework.share_tools import add_discretized_state_to_instruction

# ---------------------------------------------------------------------------------------------- prompts

_PROMPTS = {}  # framework name -> factory(cfg) -> function(examples) -> the instructions the framework will pass


def prompt(framework_name):
    def register(factory):
        _PROMPTS[framework_name] = factory
        return factory

    return register


def _lang(examples):
    return [example["lang"] for example in examples]


def _lang_with_state(examples):
    instructions = _lang(examples)
    if "state" in examples[0]:
        instructions = add_discretized_state_to_instruction(instructions, [example["state"] for example in examples])
    return instructions


@prompt("QwenOFT")
def _qwen_oft(cfg):
    chunk_len = int(cfg.framework.action_model.action_horizon)
    suffix = f" Please predict the next {chunk_len} robot actions: <action>{'🔍' * chunk_len}<action>."
    return lambda examples: [instruction + suffix for instruction in _lang_with_state(examples)]


@prompt("QwenPI_v3")
def _qwen_pi_v3(cfg):
    return _lang_with_state


# ---------------------------------------------------------------------------------------------- worker side


class Builder:
    """What a worker needs to make an interface's `build_qwenvl_inputs` call without the model. Pickled once in the
    training process and unpickled in each worker, which gives every worker its own processor."""

    def __init__(self, interface):
        self.interface_class = type(interface)
        self.stand_in = SimpleNamespace(processor=interface.processor, config=interface.config,
                                        model=SimpleNamespace(device="cpu"))
        core = getattr(interface.model, "model", None)  # e.g. Qwen3VLModel inside Qwen3VLForConditionalGeneration
        self.rope = None
        if callable(getattr(type(core), "get_rope_index", None)) and getattr(core, "config", None) is not None:
            self.rope = (type(core), core.config)

    def __call__(self, images, instructions):
        inputs = self.interface_class.build_qwenvl_inputs(self.stand_in, images=images, instructions=instructions)
        if self.rope is not None and "position_ids" not in inputs and "pixel_values_videos" not in inputs:
            model_class, config = self.rope
            # get_rope_index reads only the model config and these inputs
            inputs["position_ids"], _ = model_class.get_rope_index(
                SimpleNamespace(config=config), input_ids=inputs["input_ids"],
                image_grid_thw=inputs.get("image_grid_thw"), attention_mask=inputs["attention_mask"],
            )
        return inputs


_arrived = weakref.WeakValueDictionary()  # id(first image of a batch) -> its Prepared, in the consuming process


class Prepared:
    """The inputs a worker built for one batch. It travels as batch[0]["_prepared"] and, where the batch is consumed,
    is found again through the batch's first image object."""

    def __init__(self, images, instructions, inputs):
        self.images = [tuple(sample) for sample in images]  # the same objects as in the batch, also after pickling
        self.instructions = list(instructions)
        self.inputs = inputs
        self._announce()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._announce()

    def _announce(self):
        if self.images and self.images[0]:
            _arrived[id(self.images[0][0])] = self

    def pin_memory(self):  # called by the DataLoader's pinning thread
        for key, value in self.inputs.items():
            if torch.is_tensor(value):
                self.inputs[key] = value.pin_memory()
        return self

    def same_images(self, images):
        return len(images) == len(self.images) and all(
            len(sample) == len(mine) and all(a is b for a, b in zip(sample, mine))
            for sample, mine in zip(images, self.images))


class Collate:
    """`collate_fn`, followed by the interface's input construction for the batch it returns."""

    def __init__(self, collate_fn, builder, instructions_of):
        self.collate_fn = collate_fn
        self.instructions_of = instructions_of
        self.payload = pickle.dumps(builder)
        self._builder = None

    def __getstate__(self):
        return {**self.__dict__, "_builder": None}

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if not (isinstance(batch, list) and batch and isinstance(batch[0], dict)
                and "image" in batch[0] and "lang" in batch[0]):
            return batch
        if self._builder is None:  # once per worker process
            self._builder = pickle.loads(self.payload)
        images = [example["image"] for example in batch]
        instructions = self.instructions_of(batch)
        batch[0]["_prepared"] = Prepared(images, instructions, self._builder(images, instructions))
        return batch


# ---------------------------------------------------------------------------------------------- training process

_builder = None  # the Builder of the interface most recently accelerated in this process
stats = {"used": 0, "other prompt": 0}
_reported = threading.Event()


def _take(images, instructions):
    """The inputs a worker built for exactly this call, or None."""
    try:
        prepared = _arrived.get(id(images[0][0]))
    except (IndexError, KeyError, TypeError):
        return None
    if prepared is None or not prepared.same_images(images):
        return None
    if list(instructions) != prepared.instructions:
        stats["other prompt"] += 1
        if not _reported.is_set():
            _reported.set()
            print("[starVLA.fast] worker_prep: this framework edits the prompt in a way the data-loader workers do not "
                  "know, so the inputs are built in the training process. Register the edit with @prompt in "
                  f"starVLA/fast/worker_prep.py.\n  framework: {instructions[0]!r}\n  workers:   "
                  f"{prepared.instructions[0]!r}", flush=True)
        return None
    stats["used"] += 1
    return prepared.inputs


def _same(a, b):
    return a.keys() == b.keys() and all(
        torch.equal(a[key].cpu(), b[key].cpu()) and a[key].dtype == b[key].dtype for key in a.keys())


def _probe(interface, builder):
    """Does a worker's Builder return what the interface returns? Two samples of different prompt length, two views."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    images = [[Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)] for _ in range(2)]
    instructions = ["pick up the block", "open the top drawer and put the bowl inside"]
    theirs = interface.build_qwenvl_inputs(images=images, instructions=instructions)
    ours = pickle.loads(pickle.dumps(builder))(images, instructions)
    if "position_ids" in ours and "position_ids" not in theirs:
        position_ids, _ = interface.model.model.get_rope_index(
            input_ids=theirs["input_ids"], image_grid_thw=theirs.get("image_grid_thw"),
            attention_mask=theirs["attention_mask"])
        if not torch.equal(position_ids.cpu(), ours["position_ids"]):
            return False
        ours = {key: value for key, value in ours.items() if key != "position_ids"}
    return _same(dict(theirs), dict(ours))


def apply(interface):
    """Backbone hook: check that workers can build this interface's inputs, and let it return what they built."""
    global _builder
    original = getattr(interface, "build_qwenvl_inputs", None)
    if not callable(original) or getattr(interface, "processor", None) is None or not hasattr(interface, "model"):
        return False
    if getattr(original, "_starvla_worker_prep", False):
        return True
    try:
        builder = Builder(interface)
        if not _probe(interface, builder):
            return False
    except Exception as error:  # an interface that builds its inputs differently: leave it alone
        print(f"[starVLA.fast] worker_prep: {type(interface).__name__} not supported ({type(error).__name__}: {error})",
              flush=True)
        return False

    def build_qwenvl_inputs(images, instructions, solutions=None, **kwargs):
        prepared = _take(images, instructions) if solutions is None and not kwargs else None
        if prepared is None:
            return original(images, instructions, solutions=solutions, **kwargs)
        device = interface.model.device
        return type(prepared)(data={key: value.to(device) for key, value in prepared.items()})  # a new one per call

    build_qwenvl_inputs._starvla_worker_prep = True
    interface.build_qwenvl_inputs = build_qwenvl_inputs
    _builder = builder
    return True


def collate(cfg, collate_fn):
    """Data hook: the collate function for the workers; `collate_fn` itself when no interface was set up."""
    if _builder is None:
        return collate_fn
    factory = _PROMPTS.get(str(cfg.framework.get("name", "")))
    return Collate(collate_fn, _builder, factory(cfg) if factory is not None else _lang)

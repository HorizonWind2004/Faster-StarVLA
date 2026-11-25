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

Upstream leaves UMT5 in training mode during training, with its dropout (rate 0.1) active. Dropout on fewer positions
draws other masks and moves the random stream for everything after it, so while the encoder has an active dropout
the upstream encoding runs unchanged: in training as upstream configures it this path does nothing; it applies to
evaluation and inference (and to a recipe that puts the text encoder in eval mode).
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


def _active_dropout(module):
    return module.training and any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in module.modules())


def text_longest(backbone):
    """Backbone hook: True if the backbone is Wan's interface and its text encoding was replaced."""
    if not all(hasattr(backbone, name) for name in ("tokenizer", "text_encoder")) or not _zero_padded_upstream(backbone):
        return False
    if getattr(backbone._encode_text, "_starvla_text_longest", False):
        return True
    upstream = backbone._encode_text

    def _encode_text(instructions, max_length=512):
        if _active_dropout(backbone.text_encoder):  # other dropout masks, another random stream: keep upstream's
            return upstream(instructions, max_length=max_length)
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


# ---------------------------------------------------------------------------------------------- video_prep
"""video_prep (exact): the observation images of a world-model backbone resized and normalized in the data-loader
workers. Every step the Wan / Cosmos-Predict2 interface passes each sample's views through diffusers' VideoProcessor
(resize to 480 x 832 for Wan, 320 x 576 for Cosmos-Predict2, then scale to [-1, 1]) on the CPU of the training process
while the GPU waits. The workers have the samples first: they make the same call with a copy of the same processor
and send the tensors with the batch. The interface's call for a sample whose images are the batch's own objects gets
the worker's tensor; any other call (inference, a framework that edits the images) runs as before.
"""
import re  # noqa: E402
import threading  # noqa: E402
import weakref  # noqa: E402

_SIZE = re.compile(r"height, width = (\d+), (\d+)")
_videos = {"builder": None}  # (VideoProcessor, height, width) of the backbone most recently set up
_video_arrived = weakref.WeakValueDictionary()  # id(first image of a sample) -> the PreparedVideos holding it
_video_lock = threading.Lock()
video_stats = {"used": 0}


def _video_size(backbone):
    encode = getattr(type(backbone), "_encode_images_vae", None)
    try:
        source = inspect.getsource(encode)
    except (OSError, TypeError):
        return None
    match = _SIZE.search(source)
    if match is None or "self.video_processor.preprocess_video(sample_imgs, height=height, width=width)" not in source:
        return None
    return int(match.group(1)), int(match.group(2))


class PreparedVideos:
    """The preprocessed views of one batch, one tensor per sample. Found again through each sample's first image."""

    def __init__(self, images, videos, size):
        self.images, self.videos, self.size = [tuple(sample) for sample in images], videos, size
        self._announce()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._announce()

    def _announce(self):
        with _video_lock:
            for sample in self.images:
                if sample:
                    _video_arrived[id(sample[0])] = self

    def pin_memory(self):
        self.videos = [video.pin_memory() for video in self.videos]
        return self

    def take(self, sample, size):
        if size != self.size:
            return None
        for mine, video in zip(self.images, self.videos):
            if len(mine) == len(sample) and all(a is b for a, b in zip(mine, sample)):
                return video
        return None


class VideoCollate:
    """`collate_fn`, followed by the backbone's preprocessing of each sample's views."""

    def __init__(self, collate_fn, builder):
        self.collate_fn, self.builder = collate_fn, builder

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        if not (isinstance(batch, list) and batch and isinstance(batch[0], dict) and "image" in batch[0]):
            return batch
        processor, height, width = self.builder
        images = [sample["image"] if isinstance(sample["image"], (list, tuple)) else [sample["image"]] for sample in batch]
        videos = [processor.preprocess_video(list(sample), height=height, width=width) for sample in images]
        batch[0]["_videos"] = PreparedVideos(images, videos, (height, width))
        return batch


class _VideoProcessor:
    """The backbone's VideoProcessor, answering with the workers' tensor where there is one."""

    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self.__dict__["_processor"], name)

    def preprocess_video(self, video, height=None, width=None, **kwargs):
        if not kwargs and isinstance(video, (list, tuple)) and video:
            with _video_lock:
                prepared = _video_arrived.get(id(video[0]))
            found = prepared.take(video, (height, width)) if prepared is not None else None
            if found is not None:
                video_stats["used"] += 1
                return found
        return self._processor.preprocess_video(video, height=height, width=width, **kwargs)


def video_prep(backbone):
    """Backbone hook: a world-model interface whose `_encode_images_vae` preprocesses each sample with its
    VideoProcessor at a fixed size gets the tensors the data-loader workers made."""
    import pickle

    processor = getattr(backbone, "video_processor", None)
    size = _video_size(backbone)
    if processor is None or size is None:
        return False
    if isinstance(processor, _VideoProcessor):
        return True
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    sample = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    try:
        copy = pickle.loads(pickle.dumps(processor))  # what a worker has
        if not torch.equal(processor.preprocess_video(sample, height=size[0], width=size[1]),
                           copy.preprocess_video(sample, height=size[0], width=size[1])):
            return False
    except Exception:  # a processor that cannot travel to the workers: leave it alone
        return False
    backbone.video_processor = _VideoProcessor(processor)
    _videos["builder"] = (copy, size[0], size[1])
    return True


def video_collate(cfg, collate_fn):
    """Data hook: the collate function for the workers; `collate_fn` itself when no backbone was set up."""
    if _videos["builder"] is None or isinstance(collate_fn, VideoCollate):
        return collate_fn
    return VideoCollate(collate_fn, _videos["builder"])

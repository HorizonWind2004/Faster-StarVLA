"""Opt-in fast paths of the trainer. Every switch is an environment variable, off unless set to a non-zero value;
with none set the trainer runs the upstream code path.

STARVLA_PREP_AHEAD   prepare the next batch's model inputs (prompt construction + VLM processor, CPU only) in a
                     background thread while the GPU works on the current step. Batches are taken from the data
                     loader in the same order and the prepared tensors are the ones forward() would have built.
STARVLA_SKIP_LM_HEAD QwenOFT training reads the VLM's last hidden state only: call the backbone without the
                     language-model head, whose logits (batch x tokens x 151,936) are never used. Same hidden
                     state (the last decoder layer's output), same gradients.
STARVLA_FAST_ATTN_MASK build the causal + padding attention mask with two broadcasted tensor ops instead of transformers'
                     four nested torch.vmap calls (see starVLA/model/modules/vlm/fast_masks.py). Same boolean mask.
"""
import os
import queue
import threading


def switch(name):
    return os.environ.get(name, "0") not in ("", "0", "false", "False")


class BatchPrefetcher:
    """Pulls batches with `next_batch` (the trainer's own, epoch-aware fetch) and runs `prepare` on each, one step
    ahead of the training loop. get() returns (batch, prepared) in data-loader order."""

    def __init__(self, next_batch, prepare, depth=2):
        self._next_batch, self._prepare = next_batch, prepare
        self._queue = queue.Queue(maxsize=depth)
        self._thread = threading.Thread(target=self._run, name="batch-prefetcher", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                batch = self._next_batch()
                self._queue.put((batch, self._prepare(batch), None))
        except BaseException as error:  # surfaced by get() in the training thread
            self._queue.put((None, None, error))

    def get(self):
        batch, prepared, error = self._queue.get()
        if error is not None:
            raise error
        return batch, prepared


def make_prefetcher(trainer):
    """A BatchPrefetcher for this trainer, or None when the switch is off or the framework has no prepare_inputs()."""
    if not switch("STARVLA_PREP_AHEAD"):
        return None
    module = getattr(trainer.model, "module", trainer.model)
    prepare = getattr(module, "prepare_inputs", None)
    if prepare is None:
        return None
    return BatchPrefetcher(trainer._next_batch_sync, prepare)

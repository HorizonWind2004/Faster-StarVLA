"""worker_prep: the VLM inputs of a training batch built in the data-loader workers.

The implementation is starVLA/dataloader/worker_prep.py (described there): the workers run the interface's own
`build_qwenvl_inputs` (or a framework's own `_build_vlm_inputs`) on a stand-in without the model, after a probe that
the stand-in returns what the interface returns. What they build is used only by the interface it was built for, for
the call it was built for (the same images, by identity and content, and the same prompts), while the processor,
tokenizer and config are as they were when the workers started, and once. Any other call runs in the training
process as before. This module holds the entries of the fast-path registry.
"""
from starVLA.dataloader import worker_prep as _impl
from starVLA.dataloader.worker_prep import Builder, Collate, Prepared, prompt  # noqa: F401
from starVLA.dataloader.worker_prep import input_stats as stats  # noqa: F401


def apply(interface):
    """Backbone hook: check that workers can build this interface's inputs, and let it return what they built."""
    return _impl.prepare_inputs_in_workers(interface)


def apply_framework(framework):
    """Framework hook: the same for a framework that builds its backbone's inputs itself (MiniCPM-RobotManip)."""
    return _impl.prepare_framework_inputs_in_workers(framework)


def collate(cfg, collate_fn):
    """Data hook: the collate function for the workers; `collate_fn` itself when no interface was set up, when the
    framework's training call is not the one the workers build for, or when this DataLoader is for another model."""
    return _impl.inputs_collate(cfg, collate_fn)

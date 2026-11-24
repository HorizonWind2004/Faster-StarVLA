"""starVLA.fast: opt-in fast paths for training and inference, behind one setting.

    export STARVLA_FAST=exact    # every fast path whose results are bitwise identical to the default code
    export STARVLA_FAST=all      # exact + the paths that change floating-point rounding only (compiled layers, SDPA)
    export STARVLA_FAST=max      # all + the precision settings (TF32 for fp32 matrix products, bf16 softmax in PI0)
    export STARVLA_FAST=exact,-worker_prep,compile_lm    # a level and/or names; "-name" removes one
    unset, "off" or "0"          # the default code paths

A fast path belongs to no framework and no backbone. Each one checks the model it is given and is skipped where it
does not apply, so the same setting works for every recipe; `accelerate_backbone` prints what it turned on. The
callers are four hooks, none of them in a framework:

    get_vlm_model()            -> fast.accelerate_backbone(vlm)       every backbone interface
    build_framework()          -> fast.accelerate_framework(model)    the framework as a whole (action head, precision)
    the trainer                -> fast.accelerate_engine(engine)      the DeepSpeed engine
    build_dataloader()         -> fast.worker_collate(cfg, collate)   the data-loader workers (worker_prep, worker_examples)

To add a fast path: a module here with `apply(target) -> bool` (False = does not apply), and a line in FAST_PATHS.
"""
import importlib
import os
from typing import NamedTuple


class FastPath(NamedTuple):
    exact: bool  # bitwise identical results
    hook: str  # "backbone" | "framework" | "engine"
    apply: str  # "module:function" under starVLA.fast
    summary: str
    legacy: tuple = ()  # the per-path environment variables of the first release, still honoured
    collate: str = ""  # "module:function" wrapping the data loader's collate function, for work moved to the workers


FAST_PATHS = {  # applied in this order
    "sdpa": FastPath(False, "backbone", "attention:use_sdpa",
                     "PyTorch SDPA attention where the config asks for flash-attention: the same attention from "
                     "another kernel, and the one the fast paths below accelerate"),
    "attn_mask": FastPath(True, "backbone", "attn_mask:apply",
                          "causal + padding attention mask from two tensor ops instead of nested torch.vmap",
                          ("STARVLA_FAST_ATTN_MASK",)),
    "hidden_only": FastPath(True, "backbone", "hidden_only:apply",
                            "no vocabulary logits when the caller asks for hidden states and has no labels",
                            ("STARVLA_SKIP_LM_HEAD", "STARVLA_INFER_SKIP_LM_HEAD")),
    "vision_attn": FastPath(True, "backbone", "vision_attn:apply",
                            "one batched attention call per vision layer instead of one per image, for the "
                            "shapes where a probe on this device shows the batched call is bitwise identical",
                            ("STARVLA_BATCHED_VISION_ATTN",)),
    "vision_pos": FastPath(True, "backbone", "vision_pos:apply",
                           "vision position tables built once per grid size instead of in every forward",
                           ("STARVLA_VISION_POS_CACHE",)),
    "worker_prep": FastPath(True, "backbone", "worker_prep:apply",
                            "the interface's input construction (VLM processor) in the data-loader workers",
                            ("STARVLA_PREP_IN_WORKERS",), "worker_prep:collate"),
    "worker_examples": FastPath(True, "framework", "openpi:worker_examples",
                                "PI0 / PI05: the framework's sample preparation (image resize and padding, state "
                                "and action padding) in the data-loader workers", (), "openpi:collate"),
    "grad_norm": FastPath(True, "engine", "deepspeed:fused_grad_norm",
                          "DeepSpeed ZeRO-2 clipping norm without an fp64 copy of every gradient",
                          ("STARVLA_DS_FUSED_GRAD_NORM",)),
    "allreduce": FastPath(True, "engine", "deepspeed:plain_allreduce",
                          "DeepSpeed ZeRO-2 gradient buckets all-reduced in place",
                          ("STARVLA_DS_PLAIN_ALLREDUCE",)),
    "grad_cast": FastPath(True, "engine", "deepspeed:cast_while_flattening",
                          "DeepSpeed ZeRO-2 step: the partition's gradients are cast to fp32 while they are "
                          "flattened, in one pass instead of two"),
    "vision_attn_relaxed": FastPath(False, "backbone", "vision_attn:apply_relaxed",
                                    "the batched vision attention for every uniform shape (last-bit differences)"),
    "compile_lm": FastPath(False, "backbone", "compile:apply_lm",
                           "decoder layers compiled with Inductor for training calls",
                           ("STARVLA_COMPILE_LM",)),
    "compile_vision": FastPath(False, "backbone", "compile:apply_vision",
                               "vision blocks compiled with Inductor for training calls",
                               ("STARVLA_COMPILE_VISION",)),
    "condition_once": FastPath(False, "framework", "condition_once:apply",
                               "the action DiT projects its conditioning once per sample, not once per repeated "
                               "copy (the frameworks repeat it for several noise levels per sample)"),
    "compile_head": FastPath(False, "framework", "compile:apply_head",
                             "the transformer blocks of the DiT action heads compiled with Inductor for training calls"),
    # before unused_views, which wraps it: the views that remain are the ones embedded together
    "openpi_one_pass": FastPath(False, "framework", "openpi:one_pass",
                                "PI0 / PI05: the camera views of a batch go through SigLIP in one call"),
    "unused_views": FastPath(False, "framework", "openpi:unused_views",
                             "PI0 / PI05: a camera view that no sample of the batch has is left out instead of "
                             "being encoded as a black image and masked"),
    "openpi_compile": FastPath(False, "framework", "openpi:compile_layers",
                               "PI0 / PI05: SigLIP layers and the joint PaliGemma / action-expert layers compiled "
                               "with Inductor for training calls"),
    # measured slower than the default on two ranks; selectable by name only
    "reduce_to_owner": FastPath(True, "engine", "deepspeed:reduce_to_owner",
                                "DeepSpeed ZeRO-2 gradient buckets reduced to the owning rank only",
                                ("STARVLA_DS_REDUCE_TO_OWNER",)),
    # precision settings rather than reorderings (tier N2); selectable by name or with the level "max"
    "openpi_sdpa": FastPath(False, "framework", "openpi:use_sdpa",
                            "PI0 / PI05: fused scaled-dot-product attention in the SigLIP tower and the joint "
                            "PaliGemma / action-expert layers instead of matmul, fp32 softmax, matmul (the softmax "
                            "then runs in bf16)"),
    "tf32": FastPath(False, "framework", "precision:apply_tf32",
                     "TF32 for fp32 matrix products (the fp32 action heads); lower precision than the default"),
}
PRECISION = frozenset({"openpi_sdpa", "tf32"})  # change the precision of an operation, not only its order
BY_NAME_ONLY = frozenset({"reduce_to_owner"})
LEVELS = {
    "exact": frozenset(n for n, p in FAST_PATHS.items() if p.exact and n not in BY_NAME_ONLY),
    "all": frozenset(n for n in FAST_PATHS if n not in BY_NAME_ONLY | PRECISION),
    "max": frozenset(n for n in FAST_PATHS if n not in BY_NAME_ONLY),
}
_OFF = ("", "0", "off", "none", "false", "False")
_cache = (None, frozenset())


def selection():
    """The names selected by STARVLA_FAST, plus any legacy per-path variable that is set."""
    global _cache
    spec = os.environ.get("STARVLA_FAST", "")
    legacy = tuple(os.environ.get(var, "") for path in FAST_PATHS.values() for var in path.legacy)
    if _cache[0] == (spec, legacy):
        return _cache[1]
    names = set()
    for token in (t.strip() for t in spec.split(",")):
        if token in _OFF:
            continue
        if token in LEVELS:
            names |= LEVELS[token]
        elif token.lstrip("-") in FAST_PATHS:
            (names.discard if token.startswith("-") else names.add)(token.lstrip("-"))
        else:
            raise ValueError(f"STARVLA_FAST: unknown entry {token!r}; levels: {sorted(LEVELS)}, "
                             f"fast paths: {sorted(FAST_PATHS)}")
    for name, path in FAST_PATHS.items():
        if any(os.environ.get(var, "0") not in _OFF for var in path.legacy):
            names.add(name)
    _cache = ((spec, legacy), frozenset(names))
    return _cache[1]


def enabled(name):
    return name in selection()


def attention_implementation(*modules):
    """The attention implementation ("sdpa", "flash_attention_2", "eager") of the first of these Hugging Face
    (sub)models whose config states one."""
    for module in modules:
        implementation = getattr(getattr(module, "config", None), "_attn_implementation", None)
        if implementation is not None:
            return implementation
    return None


def _run(hook, target):
    """Apply the selected fast paths of one hook to `target`; returns (applied, not applicable) names."""
    applied, skipped = [], []
    for name, path in FAST_PATHS.items():
        if path.hook != hook or name not in selection():
            continue
        module, function = path.apply.split(":")
        ok = getattr(importlib.import_module(f"starVLA.fast.{module}"), function)(target)
        (applied if ok else skipped).append(name)
    if applied or skipped:
        note = f"; not applicable here: {', '.join(skipped)}" if skipped else ""
        print(f"[starVLA.fast] {type(target).__name__}: {', '.join(applied) or 'nothing'}{note}", flush=True)
    return applied, skipped


def accelerate_backbone(vlm):
    """Backbone fast paths for a VLM interface (anything whose `.model` is the Hugging Face model). Returns `vlm`."""
    _run("backbone", vlm)
    return vlm


def accelerate_framework(framework):
    """Fast paths for a built framework (anything `build_framework` returns). Returns `framework`."""
    _run("framework", framework)
    return framework


def accelerate_engine(engine):
    """Optimizer-side fast paths for the engine the trainer steps (the DeepSpeed engine Accelerate returns)."""
    if not getattr(engine, "_starvla_fast", False):
        _run("engine", engine)
        try:
            engine._starvla_fast = True
        except AttributeError:
            pass
    return engine


def worker_collate(cfg, collate_fn):
    """`collate_fn`, wrapped by the selected fast paths that move work into the data-loader workers."""
    for name, path in FAST_PATHS.items():
        if path.collate and enabled(name):
            module, function = path.collate.split(":")
            collate_fn = getattr(importlib.import_module(f"starVLA.fast.{module}"), function)(cfg, collate_fn)
    return collate_fn

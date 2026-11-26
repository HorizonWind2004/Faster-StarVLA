"""Leave out of the optimizer the sub-models that never receive a gradient.

The Wan interface runs its UMT5-XXL text encoder and its VAE inside `torch.no_grad()` but leaves their parameters
trainable (`requires_grad_(False)` is commented out there; the Cosmos-Predict2 interface sets it). starVLA's
optimizer groups take every parameter, so DeepSpeed ZeRO-2 keeps an fp32 master copy and two Adam moments for these
6.5 B parameters (WanGR00T: 11.5 B trainable, 5 B of them reachable by the loss), fills their gradients with zeros,
runs the AdamW update over them and all-gathers them after every step.

The update leaves them unchanged bit for bit: with a zero gradient both moments stay zero and the Adam step is
0 / eps = 0, and the decoupled weight decay subtracts lr * weight_decay * p, which rounds away in fp32 whenever
lr * weight_decay < 2^-25 (LIBERO's recipe: at most 1e-4 * 1e-8). So upstream training already keeps them frozen;
here they are frozen before the optimizer is built (requires_grad = False, which DeepSpeed's ZeRO optimizer leaves out).

Not an exact path: the gradient-clipping norm is a sum over each rank's partition, and the partitions are cut at
other places when the zero gradients are left out, so the norm (and the clipped step) can differ in the last bit.
Optimizer checkpoints are not interchangeable between runs with and without this path; model weights are.

A sub-model is frozen only when (checked on the interface's source) every call into it is inside a
`with torch.no_grad():` block, the framework's own code does not touch it, and the configured learning rates and
weight decay keep the decay below fp32 rounding. The framework's own sub-models are checked the same way on its
source: ABot-M0 runs its VGGT-1B spatial model's aggregator under no_grad only and leaves it trainable too.
"""
import ast
import inspect
import textwrap

import torch

_NEUTRAL = {"parameters", "named_parameters", "buffers", "modules", "to", "eval", "train", "requires_grad_"}
_DECAY_BOUND = 2.0 ** -25


def _chain(node):
    """['self', 'vae', 'encode'] for self.vae.encode; None for anything that is not an attribute chain on a name."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + names[::-1]
    return None


def _no_grad(item):
    call = item.context_expr
    return isinstance(call, ast.Call) and _chain(call.func) in (["torch", "no_grad"], ["torch", "inference_mode"])


def _sources(cls):
    """The source of the class and of its bases up to torch.nn.Module (the methods it inherits); None if unreadable."""
    out = []
    for base in cls.__mro__:
        if base in (object, torch.nn.Module) or base.__module__.startswith("torch."):
            continue
        try:
            out.append(textwrap.dedent(inspect.getsource(base)))
        except (OSError, TypeError):
            return None
    return out


def _calls_under_no_grad(cls, name):
    """Every call into self.<name> (self.<name>(...), self.<name>.encode(...)) and every alias of it (x = self.<name>)
    in the class's source is inside a no_grad block, and there is at least one call. Reading its attributes, the
    module bookkeeping calls (parameters, to, eval, ...) and handing it to another object (Wan's generate builds a
    WanPipeline) do not count; what this misses the runtime guard in apply() reports."""
    sources = _sources(cls)
    try:
        trees = [ast.parse(source) for source in sources or ()]
    except SyntaxError:
        return False
    if not trees:
        return False
    calls, aliases = [], []

    def is_self_attr(node):
        return isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.value, ast.Name) and \
            node.value.id == "self"

    def visit(node, guarded):
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(_no_grad(item) for item in node.items):
            guarded = True
        if isinstance(node, ast.Call):
            chain = _chain(node.func)
            if chain and chain[:2] == ["self", name] and not (len(chain) == 3 and chain[2] in _NEUTRAL):
                calls.append(guarded)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and is_self_attr(node.value):
            aliases.append(guarded)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    for tree in trees:
        visit(tree, False)
    return bool(calls) and all(calls) and all(aliases)


def _guard(name, module):
    """Fail loudly if a frozen sub-model is ever run with gradients enabled in training: upstream would train it."""
    def check(*_):
        if torch.is_grad_enabled() and module.training:
            raise RuntimeError(f"[starVLA.fast] frozen_no_grad froze {name}, which runs here with gradients enabled "
                               f"(upstream trains it); run with STARVLA_FAST=all,-frozen_no_grad")

    for part in (module, *module.children()):
        part.register_forward_pre_hook(check)


def _mentions(cls, name):
    sources = _sources(cls)
    return sources is None or any(f".{name}" in source for source in sources)


def _decay_rounds_away(framework):
    trainer = getattr(getattr(framework, "config", None), "trainer", None)
    try:
        rates = [float(v) for v in dict(trainer.learning_rate).values()]
        decay = float(trainer.optimizer.weight_decay)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(rates) and max(rates) * decay < _DECAY_BOUND


def _candidates(framework):
    """(owner, name, module): the sub-models of the world-model backbone and of the framework itself. A backbone's
    sub-model also has to be left alone by the framework's code; the framework's own are checked on its source."""
    backbone = getattr(framework, "backbone", None)
    if isinstance(backbone, torch.nn.Module):
        for name, module in backbone.named_children():
            if not _mentions(type(framework), name):
                yield backbone, name, module
    for name, module in framework.named_children():
        if module is not backbone:
            yield framework, name, module


def apply(framework):
    if not isinstance(framework, torch.nn.Module) or not _decay_rounds_away(framework):
        return False
    frozen = []
    for owner, name, module in _candidates(framework):
        params = [p for p in module.parameters() if p.requires_grad]
        if not params or not _calls_under_no_grad(type(owner), name):
            continue
        module.requires_grad_(False)
        _guard(name, module)
        frozen.append(f"{name} ({sum(p.numel() for p in params) / 1e9:.1f} B)")
    if frozen:
        print(f"[starVLA.fast] frozen_no_grad: {', '.join(frozen)} run under torch.no_grad only: left out of the "
              f"optimizer", flush=True)
    return bool(frozen)

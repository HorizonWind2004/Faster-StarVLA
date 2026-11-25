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
weight decay keep the decay below fp32 rounding.
"""
import ast
import inspect
import textwrap

_SUB_MODELS = ("text_encoder", "vae")
_NEUTRAL_CALLS = {"parameters", "named_parameters", "buffers", "modules", "to", "eval", "train", "requires_grad_"}
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
    import torch

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
    """Every call into self.<name> in the class's source is inside a no_grad block (and there is at least one)."""
    sources = _sources(cls)
    try:
        trees = [ast.parse(source) for source in sources or ()]
    except SyntaxError:
        return False
    if not trees:
        return False
    found = []

    def visit(node, guarded):
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(_no_grad(item) for item in node.items):
            guarded = True
        if isinstance(node, ast.Call):
            chain = _chain(node.func)
            if chain and chain[:2] == ["self", name] and not (len(chain) == 3 and chain[2] in _NEUTRAL_CALLS):
                found.append(guarded)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    for tree in trees:
        visit(tree, False)
    return bool(found) and all(found)


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


def apply(framework):
    backbone = getattr(framework, "backbone", None)
    if backbone is None or not _decay_rounds_away(framework):
        return False
    frozen = []
    for name in _SUB_MODELS:
        module = getattr(backbone, name, None)
        params = [p for p in getattr(module, "parameters", lambda: ())() if p.requires_grad]
        if not params or not _calls_under_no_grad(type(backbone), name) or _mentions(type(framework), name):
            continue
        module.requires_grad_(False)
        frozen.append(f"{name} ({sum(p.numel() for p in params) / 1e9:.1f} B)")
    if frozen:
        print(f"[starVLA.fast] frozen_no_grad: {', '.join(frozen)} run under torch.no_grad only: left out of the "
              f"optimizer", flush=True)
    return bool(frozen)

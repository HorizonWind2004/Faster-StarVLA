"""No vocabulary logits for callers that only read hidden states.

Most frameworks condition their action head on the backbone's hidden states:

    outputs = vlm(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.hidden_states[-1]

The Hugging Face `...ForConditionalGeneration` wrapper then also projects every position onto the vocabulary
(batch x tokens x 150 k logits, forward and backward), which nobody reads. With this fast path such a call (hidden
states requested, no labels) runs the wrapper's own forward with the language-model head switched off, so every
other field of the output is exactly what upstream returns, including whichever convention the wrapper has for
hidden_states[-1]. `logits` is computed on first access, so a caller that does read it still gets the upstream
value. Calls with labels, or without output_hidden_states=True, are untouched.

Applies to every backbone whose model has an `lm_head`. If the first such call fails without the head (a wrapper that
post-processes its logits), the default forward is used from then on.

What such a call returns is a `HiddenOnlyOutput`, not the transformers ModelOutput itself: attribute and key access
go to the wrapper's output, but it is not an instance of ModelOutput and has no `to_tuple()`; and `logits`, when it
is read, is computed at that moment (under the autocast / no_grad context of the reader, not of the forward).

The replaced `forward` of the model and of its head are instances of module-level classes that hold their module,
not closures: they report the signature of the forward they replace (transformers' generate() inspects it to decide
which inputs to build), a deep copy of the model gets its own, and the model stays picklable.
"""
import inspect
import threading


class _Call(threading.local):
    """The hidden-only call in progress in this thread: the head whose call is recorded instead of run."""

    head = None
    head_input = None


_call = _Call()


class HiddenOnlyOutput:
    """The wrapper's outputs, with the language-model head applied only if `logits` is read."""

    def __init__(self, outputs, head, head_input):
        self._outputs = outputs
        self._head = head
        self._head_input = head_input
        self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            self._logits = self._head(self._head_input)  # the module, so that its hooks run (DeepSpeed ZeRO-3)
        return self._logits

    def __getattr__(self, name):  # hidden_states, past_key_values, attentions, loss, ...
        if name.startswith("__") or name in ("_outputs", "_head", "_head_input", "_logits"):
            raise AttributeError(name)  # not yet set (copy, pickle): no recursion
        return getattr(self._outputs, name)

    def __getitem__(self, key):
        return self.logits if key == "logits" else self._outputs[key]

    def __contains__(self, key):  # generate() looks for the cache with `name in outputs`
        return key == "logits" or key in self._outputs


class _Forward:
    """A module's `forward`, replaced on the instance. Holds the module instead of closing over it."""

    def __init__(self, owner):
        self.owner = owner
        self.replaced = vars(owner).get("forward")  # a forward someone else had set on the instance, if any

    def original(self, *args, **kwargs):
        if self.replaced is not None:
            return self.replaced(*args, **kwargs)
        return type(self.owner).forward(self.owner, *args, **kwargs)

    @property
    def __signature__(self):
        return inspect.signature(
            self.replaced if self.replaced is not None else type(self.owner).forward.__get__(self.owner)
        )


class _HeadForward(_Forward):
    """The language-model head: its own forward, except during a hidden-only call of its model in this thread, where
    it keeps its input and computes nothing."""

    def __call__(self, hidden_states, *args, **kwargs):
        if _call.head is not self.owner:
            return self.original(hidden_states, *args, **kwargs)
        _call.head_input = hidden_states
        return None


class _HiddenOnlyForward(_Forward):
    """The model: its own forward, with the head switched off for a call that asks for hidden states and has no
    labels."""

    def __init__(self, owner, head):
        super().__init__(owner)
        self.head = head
        self.works = None  # None until the first hidden-only call has run

    def __call__(self, *args, **kwargs):
        hidden_only = (
            kwargs.get("output_hidden_states") is True
            and kwargs.get("labels") is None
            and kwargs.get("return_dict", True) is not False
            # an interface that already keeps the logits of a few positions only (Gemma 4, MiniCPM-V pass
            # logits_to_keep=1): nothing left to save, and Gemma 4 post-processes its logits (soft-capping)
            and not (isinstance(kwargs.get("logits_to_keep"), int) and kwargs["logits_to_keep"] > 0)
        )
        if not hidden_only or self.works is False or _call.head is not None:
            return self.original(*args, **kwargs)
        _call.head, _call.head_input = self.head, None
        try:
            outputs = self.original(*args, **kwargs)
        except Exception as error:
            if self.works:  # it has worked before: this is the caller's error
                raise
            _call.head = None
            self.works = False  # a wrapper that needs its logits: leave it alone from now on
            print(f"[starVLA.fast] hidden_only: {type(self.owner).__name__} cannot run without its language-model "
                  f"head ({type(error).__name__}: {error}); using the default forward", flush=True)
            return self.original(*args, **kwargs)
        finally:
            head_input, _call.head, _call.head_input = _call.head_input, None, None
        if head_input is None:  # the head was not called: nothing was skipped, nothing to add
            return outputs
        self.works = True
        return HiddenOnlyOutput(outputs, self.head, head_input)


def apply(vlm):
    model = getattr(vlm, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return False
    if isinstance(vars(model).get("forward"), _HiddenOnlyForward):
        return True
    lm_head.forward = _HeadForward(lm_head)
    model.forward = _HiddenOnlyForward(model, lm_head)
    return True

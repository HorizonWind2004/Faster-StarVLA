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

Applies to every backbone whose model has an `lm_head`.
"""


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
            self._logits = self._head(self._head_input)
        return self._logits

    def __getattr__(self, name):  # hidden_states, past_key_values, attentions, loss, ...
        return getattr(self._outputs, name)

    def __getitem__(self, key):
        return self.logits if key == "logits" else self._outputs[key]


def apply(vlm):
    model = getattr(vlm, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return False
    if getattr(model, "_starvla_hidden_only", False):
        return True
    upstream_forward, upstream_head = model.forward, lm_head.forward
    call = {"skip": False, "head_input": None}

    def head_forward(hidden_states, *args, **kwargs):
        if not call["skip"]:
            return upstream_head(hidden_states, *args, **kwargs)
        call["head_input"] = hidden_states
        return None

    def forward(*args, **kwargs):
        hidden_only = (
            kwargs.get("output_hidden_states") is True
            and kwargs.get("labels") is None
            and kwargs.get("return_dict", True) is not False
        )
        if not hidden_only:
            return upstream_forward(*args, **kwargs)
        call["skip"] = True
        try:
            outputs = upstream_forward(*args, **kwargs)
        finally:
            call["skip"] = False
        head_input, call["head_input"] = call["head_input"], None
        return HiddenOnlyOutput(outputs, upstream_head, head_input)

    lm_head.forward = head_forward
    model.forward = forward
    model._starvla_hidden_only = True
    return True

"""starVLA.fast on tiny random-weight backbones (CPU, seconds): the selection syntax, and that every exact backbone
fast path leaves hidden states, logits and gradients bitwise identical on Qwen3-VL and Qwen2.5-VL. With
STARVLA_TEST_VLMS set to model directories, also worker_prep with their real processors.

    python tests/test_fast_paths.py        # or: pytest tests/test_fast_paths.py
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import torch

BACKBONE_PATHS = ("attn_mask", "hidden_only", "vision_attn", "vision_pos")


def _tiny(family):
    torch.manual_seed(0)
    text = dict(vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]})  # head_dim 16 = 2 * (4 + 2 + 2)
    if family == "qwen3_5":  # transformers 5: three linear-attention layers and one full-attention layer
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        text = dict(vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
                    linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=512, pad_token_id=0,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 0],
                                     "mrope_interleaved": True, "partial_rotary_factor": 0.25})
        vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=4, spatial_merge_size=2,
                      temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=36)
        config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=150, video_token_id=151,
                               vision_start_token_id=152, vision_end_token_id=153)
        return Qwen3_5ForConditionalGeneration._from_config(config, attn_implementation="sdpa").float()
    if family == "qwen3":
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=4, spatial_merge_size=2,
                      temporal_patch_size=2, out_hidden_size=64, num_position_embeddings=36,
                      deepstack_visual_indexes=[0])
        config = Qwen3VLConfig(text_config=text, vision_config=vision, image_token_id=150, video_token_id=151,
                               vision_start_token_id=152, vision_end_token_id=153)
        model = Qwen3VLForConditionalGeneration._from_config(config, attn_implementation="sdpa")
    else:
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        vision = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=4, spatial_merge_size=2,
                      temporal_patch_size=2, out_hidden_size=64, window_size=16, fullatt_block_indexes=[1],
                      spatial_patch_size=4)
        config = Qwen2_5_VLConfig(text_config=text, vision_config=vision, image_token_id=150, video_token_id=151,
                                  vision_start_token_id=152, vision_end_token_id=153)
        model = Qwen2_5_VLForConditionalGeneration._from_config(config, attn_implementation="sdpa")
    return model.float()


def _inputs(model, grids):
    """Two samples; `grids` lists the (h, w) patch grid of each image, split evenly between the samples."""
    torch.manual_seed(1)
    vision = model.config.vision_config
    patch_dim = 3 * vision.temporal_patch_size * vision.patch_size**2
    merge = vision.spatial_merge_size**2
    per_sample = len(grids) // 2
    rows = []
    for sample in range(2):
        ids = [5, 6]
        for h, w in grids[sample * per_sample:(sample + 1) * per_sample]:
            ids += [152] + [150] * (h * w // merge) + [153]
        ids += list(range(10, 14 + 3 * sample))  # different text lengths: left padding on the shorter sample
        rows.append(ids)
    length = max(len(r) for r in rows)
    input_ids = torch.tensor([[0] * (length - len(r)) + r for r in rows])
    attention_mask = torch.tensor([[0] * (length - len(r)) + [1] * len(r) for r in rows])
    inputs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=torch.randn(sum(h * w for h, w in grids), patch_dim),
        image_grid_thw=torch.tensor([[1, h, w] for h, w in grids]),
    )
    if model.config.model_type == "qwen3_5":  # transformers 5 processors return the modality of every token
        inputs["mm_token_type_ids"] = (input_ids == 150).long()
    return inputs


def _vision_attention_calls(model, inputs):
    """Number of scaled_dot_product_attention calls inside one forward of the vision tower."""
    import torch.nn.functional as F

    with torch.no_grad():  # let vision_attn probe this shape before the calls are counted
        model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    calls, original = [0], F.scaled_dot_product_attention

    def counting(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    F.scaled_dot_product_attention = counting
    try:
        with torch.no_grad():
            model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    finally:
        F.scaled_dot_product_attention = original
    return calls[0]


def _run(model, inputs):
    model.zero_grad()
    outputs = model(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1]
    hidden.float().pow(2).mean().backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return hidden.detach().clone(), grads, outputs


def _check(family):
    from starVLA import fast

    model = _tiny(family).train()
    cases = {"uniform": [(8, 8)] * 4, "mixed sizes": [(8, 8), (8, 12), (4, 8), (8, 8)]}
    reference = {}
    calls_before = _vision_attention_calls(model, _inputs(model, cases["uniform"]))
    for name, grids in cases.items():
        hidden, grads, outputs = _run(model, _inputs(model, grids))
        reference[name] = (hidden, grads, outputs.logits.detach().clone())

    os.environ["STARVLA_FAST"] = ",".join(BACKBONE_PATHS)
    applied, skipped = fast._run("backbone", SimpleNamespace(model=model))
    # transformers 5 builds its masks differently: attn_mask checks its builder against the installed one first
    expected = [p for p in BACKBONE_PATHS if not (family == "qwen3_5" and p == "attn_mask" and p in skipped)]
    assert sorted(applied) == sorted(expected) and not set(skipped) - set(BACKBONE_PATHS) - set(expected) - {"attn_mask"}, (applied, skipped)

    for name, grids in cases.items():
        hidden, grads, outputs = _run(model, _inputs(model, grids))
        ref_hidden, ref_grads, ref_logits = reference[name]
        assert type(outputs).__name__ == "HiddenOnlyOutput", type(outputs)
        assert torch.equal(hidden, ref_hidden), f"{family} / {name}: hidden state differs"
        assert grads.keys() == ref_grads.keys()
        for key in grads:
            assert torch.equal(grads[key], ref_grads[key]), f"{family} / {name}: gradient of {key} differs"
        assert torch.equal(outputs.logits, ref_logits), f"{family} / {name}: lazily computed logits differ"
    # a call with labels takes the upstream path
    inputs = _inputs(model, cases["uniform"])
    assert model(**inputs, labels=inputs["input_ids"], output_hidden_states=True).loss is not None
    # the batched vision attention really ran on the uniform case: one attention call per vision layer
    calls_after = _vision_attention_calls(model, inputs)
    depth = len(model.model.visual.blocks)
    assert calls_after == depth < calls_before, (calls_before, calls_after, depth)
    from starVLA.fast import vision_attn

    assert vision_attn.decisions and all(vision_attn.decisions.values()), vision_attn.decisions  # probes ran, passed
    print(f"ok {family}: {len(grads)} gradients, hidden states and logits bitwise identical "
          f"({', '.join(cases)}; fast paths: {', '.join(applied)}); "
          f"vision attention calls per forward {calls_before} -> {calls_after}")


def _check_flash_attention():
    """A backbone loaded with flash-attention. At the exact level it keeps flash-attention and the SDPA-only fast
    paths report "not applicable" instead of failing in the first training step; at the all level it is switched to
    SDPA first, and then they apply."""
    from starVLA import fast
    from starVLA.fast.attention import _implementations

    model = _tiny("qwen2_5")
    for config in (model.config, model.model.visual.config, model.model.language_model.config):
        config._attn_implementation_internal = "flash_attention_2"  # cannot be loaded this way without a GPU
    os.environ["STARVLA_FAST"] = "exact"
    applied, skipped = fast._run("backbone", SimpleNamespace(model=model))
    assert applied == ["hidden_only", "vision_pos"], applied
    assert skipped == ["attn_mask", "vision_attn", "worker_prep"], skipped  # worker_prep: no processor on this bare model
    assert _implementations(model) == {"flash_attention_2"}
    os.environ["STARVLA_FAST"] = "all"
    applied, skipped = fast._run("backbone", SimpleNamespace(model=model))
    assert applied == ["sdpa", "attn_mask", "hidden_only", "vision_attn", "vision_pos", "vision_attn_relaxed",
                       "compile_lm", "compile_vision"], applied
    assert skipped == ["worker_prep", "compile_world_model", "text_longest"] and _implementations(model) == {"sdpa"}, skipped
    with torch.no_grad():  # an inference call: the compiled layers are for training calls
        model.eval()(**_inputs(model, [(4, 4), (4, 4)]), output_hidden_states=True, return_dict=True)
    print(f"ok flash-attention: kept at the exact level, SDPA at the all level ({', '.join(applied)})")


class _Samples:
    """Six samples the way the LeRobot datasets return them: two views, an instruction, a state, an action chunk."""

    def __len__(self):
        return 6

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(index)
        return {"image": [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
                "lang": "put the bowl on the plate" + " and then close the drawer" * (index % 3),
                "state": rng.uniform(-1, 1, (1, 8)).astype(np.float32), "action": rng.normal(size=(8, 7))}


def _same_batch(batch):
    return batch


def _check_worker_prep(model_dir):
    """worker_prep with the real processor of `model_dir` (no weights are loaded): what the interface returns for a
    batch whose inputs a data-loader worker built is what it returns without workers, for a framework that passes the
    instruction unchanged and for two that edit it; any other call is answered by the interface itself."""
    import torch.nn as nn
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoProcessor

    from starVLA import fast
    from starVLA.fast import worker_prep
    from starVLA.model.framework.share_tools import add_discretized_state_to_instruction

    if "Qwen2.5-VL" in model_dir:
        from transformers import Qwen2_5_VLModel as core_class

        from starVLA.model.modules.vlm.QWen2_5 import _QWen_VL_Interface as interface_class
    else:
        from transformers import Qwen3VLModel as core_class

        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface as interface_class
    cfg = OmegaConf.create({"framework": {"name": "QwenGR00T", "qwenvl": {"base_vlm": model_dir},
                                          "action_model": {"action_horizon": 8}}, "datasets": {"vla_data": {}}})
    core = core_class.__new__(core_class)  # get_rope_index reads only the config
    nn.Module.__init__(core)
    core.config = AutoConfig.from_pretrained(model_dir)
    interface = interface_class.__new__(interface_class)
    nn.Module.__init__(interface)
    interface.processor = AutoProcessor.from_pretrained(model_dir)
    interface.processor.tokenizer.padding_side = "left"
    interface.config = cfg
    interface.__dict__["model"] = SimpleNamespace(device=torch.device("cpu"), model=core)
    original = interface.build_qwenvl_inputs

    os.environ["STARVLA_FAST"] = "worker_prep"
    applied, skipped = fast._run("backbone", interface)
    assert applied == ["worker_prep"] and not skipped, (applied, skipped)

    suffix = f" Please predict the next 8 robot actions: <action>{'🔍' * 8}<action>."
    with_state = lambda batch: add_discretized_state_to_instruction([e["lang"] for e in batch], [e["state"] for e in batch])
    prompts = {"QwenGR00T": lambda batch: [e["lang"] for e in batch], "QwenPI_v3": with_state,
               "QwenOFT": lambda batch: [instruction + suffix for instruction in with_state(batch)]}
    for framework, prompt in prompts.items():
        cfg.framework.name = framework
        loader = DataLoader(_Samples(), batch_size=2, num_workers=2, collate_fn=fast.worker_collate(cfg, _same_batch),
                            multiprocessing_context="spawn" if framework == "QwenOFT" else None)  # must pickle
        used = worker_prep.stats["used"]
        for batch in loader:
            images, instructions = [e["image"] for e in batch], prompt(batch)
            got = interface.build_qwenvl_inputs(images=images, instructions=instructions)
            reference = original(images=images, instructions=instructions)
            position_ids, _ = core.get_rope_index(input_ids=reference["input_ids"], image_grid_thw=reference["image_grid_thw"],
                                                  attention_mask=reference["attention_mask"])
            assert type(got) is type(reference) and set(got.keys()) == set(reference.keys()) | {"position_ids"}
            for key in reference.keys():
                assert torch.equal(got[key], reference[key]) and got[key].dtype == reference[key].dtype, (framework, key)
            assert torch.equal(got["position_ids"], position_ids), framework
        assert worker_prep.stats["used"] == used + 3, (framework, worker_prep.stats)
    # a prompt the workers did not expect, and a call with other images: the interface answers itself
    batch = next(iter(loader))
    images = [e["image"] for e in batch]
    other = interface.build_qwenvl_inputs(images=images, instructions=["something else"] * 2)
    assert "position_ids" not in other and worker_prep.stats["other prompt"] == 1
    fresh = interface.build_qwenvl_inputs(images=[list(sample) for sample in images][::-1], instructions=prompts["QwenOFT"](batch)[::-1])
    assert "position_ids" not in fresh and worker_prep.stats["used"] == 9
    print(f"ok worker_prep {os.path.basename(model_dir)}: inputs built in workers are identical for "
          f"{', '.join(prompts)} ({', '.join(sorted(reference.keys()))} + position_ids)")


def _check_condition_once():
    """The action DiT with conditioning that is 4 copies of 3 rows (what the flow-matching frameworks hand it): with
    condition_once the key / value projections see 3 rows instead of 12, and loss and gradients are the same up to
    rounding (fp64 here). Conditioning that is not repeated runs as before."""
    from starVLA.fast import condition_once
    from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

    def step(layerwise, repeats):
        torch.manual_seed(0)
        dit = DiT(num_attention_heads=4, attention_head_dim=16, output_dim=32, num_layers=4, dropout=0.0,
                  interleave_self_attention=True, cross_attention_dim=48).train().double()
        condition = [torch.randn(3, 11, 48, dtype=torch.double).repeat(repeats, 1, 1) for _ in range(4)]
        mask = torch.ones(3 * repeats, 11, dtype=torch.bool)
        mask[:, -2:] = False
        projected = []
        for block in dit.transformer_blocks:
            if block.attn1.is_cross_attention:
                block.attn1.to_k.register_forward_hook(lambda module, args, output: projected.append(args[0].shape[0]))
        loss = dit(hidden_states=torch.randn(3 * repeats, 5, 64, dtype=torch.double),
                   encoder_hidden_states=condition if layerwise else condition[0],
                   timestep=torch.randint(0, 1000, (3 * repeats,)), encoder_attention_mask=mask,
                   return_pre_output=True).pow(2).mean()
        loss.backward()
        return loss.item(), torch.cat([p.grad.flatten() for p in dit.parameters() if p.grad is not None]), projected, dit

    cases = [(layerwise, repeats) for layerwise in (False, True) for repeats in (4, 1)]
    before = [step(*case) for case in cases]
    assert condition_once.apply(step(False, 1)[3]) and not condition_once.apply(torch.nn.Linear(2, 2))
    for (layerwise, repeats), (loss, grads, projected, _) in zip(cases, before):
        fast_loss, fast_grads, fast_projected, _ = step(layerwise, repeats)
        assert abs(fast_loss - loss) <= 1e-12 * abs(loss) and (fast_grads - grads).norm() <= 1e-10 * grads.norm()
        assert projected == [3 * repeats] * 2 and fast_projected == [3] * 2, (projected, fast_projected)
    print("ok condition_once: key / value projections on 3 rows instead of 12, loss and gradients equal to rounding")


def test_cast_while_flattening():
    """The fp32 gradient that DeepSpeed's step builds is bit for bit the same when it is cast while it is flattened."""
    import torch

    from starVLA.fast import deepspeed

    class DeepSpeedZeroOptimizer:  # the two lines of DeepSpeedZeroOptimizer.step that the path touches
        cpu_offload = False
        flatten = staticmethod(torch._C._nn.flatten_dense_tensors)

        def step(self, closure=None):
            self.result = self.flatten(self.gradients).to(torch.float32)

    class Engine:
        optimizer = DeepSpeedZeroOptimizer()

    torch.manual_seed(0)
    base = torch.randn(4, 5, 6).to(torch.bfloat16)
    Engine.optimizer.gradients = [base, base.transpose(0, 2), torch.randn(7).to(torch.bfloat16), torch.zeros(3, dtype=torch.bfloat16)]
    Engine.optimizer.step()
    expected = Engine.optimizer.result
    assert deepspeed.cast_while_flattening(Engine) and deepspeed.cast_while_flattening(Engine)
    Engine.optimizer.step()
    assert Engine.optimizer.result.dtype == torch.float32 and torch.equal(Engine.optimizer.result, expected)
    assert Engine.optimizer.flatten is DeepSpeedZeroOptimizer.flatten  # restored after the step
    assert not deepspeed.cast_while_flattening(object())
    print("ok cast_while_flattening: the fp32 gradient is identical")

    class DeepSpeedZeroOptimizer:  # noqa: F811  DeepSpeed's clipping, verbatim
        loss_scale, clip_grad = 1.0, 1.0

        def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
            combined_scale = self.loss_scale
            if self.clip_grad > 0.:
                clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
                clip = torch.clamp(clip, min=1.0)
                combined_scale = clip * self.loss_scale
            for grad in grad_groups_flat:
                grad.data.mul_(1. / combined_scale)

    class Engine:  # noqa: F811
        optimizer = DeepSpeedZeroOptimizer()

    gradient = torch.randn(1000)
    gradient[:4] = torch.tensor([0.0, -0.0, float("inf"), 1e-42])
    for norm in (0.25, 0.999998, 1.0, 3.0):
        expected, actual = gradient.clone(), gradient.clone()
        DeepSpeedZeroOptimizer.unscale_and_clip_grads(Engine.optimizer, [expected], torch.tensor(norm))
        assert deepspeed.skip_unit_clip(Engine)
        Engine.optimizer.unscale_and_clip_grads([actual], torch.tensor(norm))
        assert torch.equal(actual, expected) and torch.equal(torch.signbit(actual), torch.signbit(expected)), norm
        assert torch.equal(actual, gradient) == (norm < 0.9999985), norm  # untouched exactly when nothing is clipped
    print("ok skip_unit_clip: the gradient is identical with and without clipping")


def test_compile_fallbacks():
    """compile_world_model finds the blocks of a world-model backbone; a layer the compiler cannot handle runs eagerly
    instead of failing the step; hidden_only falls back to the default forward for a wrapper that needs its logits."""
    import torch

    from starVLA.fast import compile as regional
    from starVLA.fast import hidden_only

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    class World(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.transformer = torch.nn.Module()
            setattr(self.transformer, name, torch.nn.ModuleList([Block(), Block()]))

    for name in ("blocks", "transformer_blocks"):
        assert regional.apply_world_model(World(name)), name
    assert not regional.apply_world_model(torch.nn.Linear(2, 2))

    def broken(x):  # the first compiled call fails: from then on the eager forward runs
        raise RuntimeError("cannot compile this")

    original_compile = torch.compile
    torch.compile = lambda function, **kwargs: broken
    try:
        call = regional._compiled_for_training(lambda x: x + 1, dynamic=True)
    finally:
        torch.compile = original_compile
    x = torch.zeros(2, requires_grad=True)
    assert torch.equal(call(x), x + 1) and torch.equal(call(x), x + 1)

    class Wrapper(torch.nn.Module):  # post-processes its logits: cannot run without the head
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 3)

        def forward(self, x, output_hidden_states=False, labels=None, return_dict=True):
            logits = self.lm_head(x)
            return {"logits": logits.softmax(-1), "hidden_states": (x,)}

    vlm = torch.nn.Module()
    vlm.model = Wrapper()
    assert hidden_only.apply(vlm)
    x = torch.randn(2, 4)
    out = vlm.model(x, output_hidden_states=True)
    assert torch.equal(out["logits"], Wrapper.forward(vlm.model, x)["logits"])
    class Head(torch.nn.Module):  # dropout in a compiled head block draws the same masks as eager
        def __init__(self):
            super().__init__()
            self.linear, self.dropout = torch.nn.Linear(16, 16), torch.nn.Dropout(0.2)

        def forward(self, x):
            return self.dropout(self.linear(x)) + x

    torch.manual_seed(0)
    head = Head().train()
    compiled = regional._compiled_for_training(head.forward, dynamic=True, options=regional.HEAD_OPTIONS)
    for rows in (4, 7):
        results = []
        for function in (head.forward, compiled):
            x = torch.randn(rows, 16, generator=torch.Generator().manual_seed(rows), requires_grad=True)
            head.zero_grad()
            torch.manual_seed(1)
            function(x).sum().backward()
            results.append((x.grad.clone(), head.linear.weight.grad.clone(), torch.randn(2)))
        assert all(torch.equal(a, b) for a, b in zip(*results)), rows
    print("ok compile and hidden_only fallbacks; compile_world_model finds Wan / Cosmos-Predict2 blocks; "
          "a compiled head's dropout draws eager's masks")


def test_other_backbones():
    """Pieces for Qwen3.5, Gemma 4, MiniCPM-V and Florence-2 that need no weights: the compile paths of Florence-2 and
    of remote-code MiniCPM-V; a layer structure that fails to compile is not tried again, another one still is;
    hidden_only leaves interfaces that keep few logits (Gemma 4 soft-caps them) alone; the memoized
    transformers.vision_utils helpers return the upstream values and honour precomputed tables."""
    import torch

    from starVLA.fast import compile as regional
    from starVLA.fast import hidden_only, vision_pos

    def tree(path):
        root = torch.nn.Module()
        node = root
        names = path.split(".")
        for name in names[:-1]:
            child = torch.nn.Module()
            setattr(node, name, child)
            node = child
        setattr(node, names[-1], torch.nn.ModuleList([torch.nn.Linear(2, 2)]))
        return root

    assert regional.apply_lm(tree("model.language_model.model.encoder.layers"))  # Florence-2
    assert regional.apply_lm(tree("model.llm.model.layers"))  # MiniCPM-V, remote code
    assert regional.apply_vision(tree("model.vision_tower.blocks"))  # Florence-2's DaViT
    assert regional.apply_vision(tree("model.vpm.encoder.layers"))
    assert regional.apply_vision(tree("model.model.vision_tower.encoder.layers"))  # Gemma 4, MiniCPM-V 4.6

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.inner(x)

    class Full(Linear):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Identity()

    attempts = []

    def fake_compile(function, **kwargs):
        def compiled(*args, **kw):
            attempts.append(type(function.__self__.inner).__name__)
            if isinstance(function.__self__.inner, torch.nn.Linear):
                raise RuntimeError("cannot trace this kernel")
            return function(*args, **kw)
        return compiled

    original_compile = torch.compile
    torch.compile = fake_compile
    try:
        layers = [Linear(), Full(), Linear(), Full()]
        calls = [regional._compiled_for_training(layer.forward, dynamic=True) for layer in layers]
    finally:
        torch.compile = original_compile
    x = torch.ones(1, 2, requires_grad=True)
    for layer, call in zip(layers, calls):
        assert torch.equal(call(x), layer.forward(x))
    assert attempts == ["Linear", "Identity", "Identity"], attempts  # the second linear-attention layer is not tried

    class SoftCapped(torch.nn.Module):  # Gemma 4: logits_to_keep=1 and soft-capped logits
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 3)

        def forward(self, x, output_hidden_states=False, labels=None, return_dict=True, logits_to_keep=0):
            logits = self.lm_head(x[:, -logits_to_keep:] if logits_to_keep else x)
            return {"logits": torch.tanh(logits / 30.0) * 30.0, "hidden_states": (x,)}

    vlm = torch.nn.Module()
    vlm.model = SoftCapped()
    assert hidden_only.apply(vlm)
    x = torch.randn(2, 5, 4)
    out = vlm.model(x, output_hidden_states=True, logits_to_keep=1)
    assert isinstance(out, dict) and torch.equal(out["logits"], SoftCapped.forward(vlm.model, x, logits_to_keep=1)["logits"])

    counted = []

    def get_vision_position_ids(grid_thw, spatial_merge_size, include_temporal=False, kwargs=None):
        if kwargs is not None and (position_ids := kwargs.pop("position_ids", None)) is not None:
            return position_ids
        counted.append(1)
        return torch.arange(int(grid_thw.prod(-1).sum())) * spatial_merge_size

    memo = vision_pos._memoized_helper(get_vision_position_ids)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    first = memo(grid, 2, kwargs={})
    again = memo(grid.clone(), 2, kwargs={})
    assert torch.equal(first, get_vision_position_ids(grid, 2)) and again is first and len(counted) == 2
    assert not torch.equal(memo(grid, 1), first)  # another argument: computed
    given = {"position_ids": torch.zeros(3)}
    assert torch.equal(memo(grid, 2, kwargs=given), torch.zeros(3)) and not given  # popped, as upstream does
    assert torch.equal(memo(torch.tensor([[1, 2, 2]]), 2), get_vision_position_ids(torch.tensor([[1, 2, 2]]), 2))
    from starVLA import fast

    framework = torch.nn.Module()  # MiniCPM-RobotManip: the framework's own model holds the backbone as `vlm`
    framework.model = torch.nn.Module()
    framework.model.vlm = _tiny("qwen3")
    os.environ["STARVLA_FAST"] = "hidden_only"
    fast.accelerate_framework(framework)
    assert getattr(framework.model.vlm, "_starvla_hidden_only", False)
    interface = SimpleNamespace(model=_tiny("qwen3"))  # a backbone that went through get_vlm_model is not redone
    fast.accelerate_backbone(interface)
    framework.model.vlm = interface.model
    calls = []
    original_run = fast._run
    fast._run = lambda hook, target: calls.append(hook) or ([], [])
    try:
        fast.accelerate_framework(framework)
    finally:
        fast._run = original_run
    assert calls == ["framework"], calls
    os.environ.pop("STARVLA_FAST")
    print("ok other backbones: Florence-2 / MiniCPM-V compile paths, one failed compile per layer structure, "
          "hidden_only with logits_to_keep, memoized vision_utils helpers, a framework's own backbone")


def test_text_longest():
    """text_longest on Wan's interface: the upstream _encode_text (called unbound on a stand-in with a tiny random
    UMT5) and the fast one give the same tensor shape, zeros past every instruction, and the kept positions equal up
    to rounding (bitwise here, in fp32 on the CPU); an interface with other text code is left alone."""
    import types

    import torch
    from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

    from starVLA.fast import world_model
    from starVLA.model.modules.world_model.Wan2 import _Wan2_Interface

    model_dir = os.path.join(os.environ.get("ASSETS", ""), "models", "Qwen3-VL-4B-Instruct")
    if not os.path.isdir(model_dir):
        print("skip text_longest: no tokenizer under $ASSETS/models")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # any right-padding tokenizer serves (Wan's is UMT5's)
    tokenizer.padding_side = "right"
    torch.manual_seed(0)
    encoder = UMT5EncoderModel(UMT5Config(vocab_size=len(tokenizer), d_model=32, d_kv=8, d_ff=64,
                                          num_layers=2, num_heads=4, dropout_rate=0.1)).eval()
    stand_in = types.SimpleNamespace(tokenizer=tokenizer, text_encoder=encoder)
    instructions = ["pick up the black bowl", "open the top drawer of the cabinet and put the bowl inside it"]
    upstream = _Wan2_Interface._encode_text(stand_in, instructions, max_length=64)

    class Wan(_Wan2_Interface):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.tokenizer, self.text_encoder = tokenizer, encoder

    wan = Wan()
    assert world_model.text_longest(wan)
    fast = wan._encode_text(instructions, max_length=64)
    assert fast.shape == upstream.shape and fast.dtype == upstream.dtype
    lengths = tokenizer(instructions, return_attention_mask=True)["attention_mask"]
    for row, mask in enumerate(lengths):
        assert not fast[row, len(mask):].any()
    assert torch.allclose(fast.float(), upstream.float(), atol=1e-2, rtol=0), (fast.float() - upstream.float()).abs().max()
    encoder.train()  # active dropout: upstream's encoding, with upstream's random draws
    torch.manual_seed(3)
    upstream_train = _Wan2_Interface._encode_text(stand_in, instructions, max_length=64)
    torch.manual_seed(3)
    assert torch.equal(wan._encode_text(instructions, max_length=64), upstream_train)
    encoder.eval()

    class Other(torch.nn.Module):
        tokenizer = text_encoder = None

        def _encode_text(self, instructions, max_length=512):
            return None

    assert not world_model.text_longest(Other())
    print(f"ok text_longest: upstream with active dropout; in eval mode same shape and zeros, largest difference {(fast.float() - upstream.float()).abs().max():.1e}")


def test_adamw_compiled():
    """The written-out AdamW update follows torch.optim.AdamW to rounding over several steps (eager here; on the GPU it
    is compiled), and the path does not apply to anything but a fused torch AdamW."""
    import torch

    from starVLA.fast import optimizer

    lr, betas, eps, decay = 2.5e-5, (0.9, 0.95), 1e-8, 1e-8
    torch.manual_seed(0)
    reference = torch.nn.Parameter(torch.randn(4096) * 0.02)
    ours, m, v, step = reference.detach().clone(), torch.zeros(4096), torch.zeros(4096), torch.zeros(())
    adamw = torch.optim.AdamW([reference], lr=lr, betas=betas, eps=eps, weight_decay=decay)
    for _ in range(5):
        gradient = torch.randn(4096) * 1e-3
        reference.grad = gradient.clone()
        adamw.step()
        step += 1
        t = step.double()
        optimizer._adamw(ours, gradient, m, v, 1 - lr * decay, betas[0], betas[1], lr / (1 - betas[0] ** t),
                         (1 - betas[1] ** t).sqrt(), eps)
    assert ((ours - reference.detach()).abs() / reference.detach().abs().clamp_min(1e-30)).max() < 1e-5
    assert not optimizer.apply(object())
    print("ok adamw_compiled: the update follows torch's AdamW to rounding over 5 steps")


def test_condition_once():
    _in_subprocess("condition_once")


def test_selection():
    from starVLA import fast

    saved = dict(os.environ)
    try:
        for var in [v for v in os.environ if v.startswith("STARVLA_")]:
            del os.environ[var]
        assert fast.selection() == frozenset()
        os.environ["STARVLA_FAST"] = "exact"
        assert fast.selection() == fast.LEVELS["exact"] and not fast.enabled("compile_lm") and not fast.enabled("sdpa")
        assert next(iter(fast.FAST_PATHS)) == "sdpa"  # applied before the paths that are written for SDPA
        assert not fast.enabled("reduce_to_owner")
        os.environ["STARVLA_FAST"] = "all"
        assert not fast.enabled("tf32") and not fast.enabled("openpi_sdpa")  # precision settings: level "max" or by name
        assert not fast.enabled("reduce_to_owner")  # by name only
        os.environ["STARVLA_FAST"] = "all,tf32"
        assert fast.enabled("tf32") and fast.enabled("compile_lm") and not fast.enabled("openpi_sdpa")
        os.environ["STARVLA_FAST"] = "max"
        assert fast.selection() == fast.LEVELS["all"] | {"tf32", "openpi_sdpa"} and not fast.enabled("reduce_to_owner")
        os.environ["STARVLA_FAST"] = "all,-worker_prep"
        assert fast.enabled("compile_vision") and not fast.enabled("worker_prep")
        assert fast.enabled("vision_attn_relaxed") and "vision_attn_relaxed" not in fast.LEVELS["exact"]
        os.environ["STARVLA_FAST"] = "off"
        os.environ["STARVLA_BATCHED_VISION_ATTN"] = "1"  # variable of the first release
        assert fast.selection() == frozenset({"vision_attn"})
        # the nine variables of the first release select the paths they named (nothing relaxed)
        first_release = ("PREP_IN_WORKERS", "SKIP_LM_HEAD", "FAST_ATTN_MASK", "BATCHED_VISION_ATTN", "VISION_POS_CACHE",
                         "DS_FUSED_GRAD_NORM", "COMPILE_LM", "DS_PLAIN_ALLREDUCE", "COMPILE_VISION")
        for name in first_release:
            os.environ["STARVLA_" + name] = "1"
        assert fast.selection() == frozenset(n for n, p in fast.FAST_PATHS.items() if p.legacy) - {"reduce_to_owner"}
        for name in ("COMPILE_LM", "COMPILE_VISION"):
            del os.environ["STARVLA_" + name]
        assert fast.selection() == frozenset(n for n in fast.LEVELS["exact"] if fast.FAST_PATHS[n].legacy)
        os.environ["STARVLA_FAST"] = "nonsense"
        try:
            fast.selection()
        except ValueError:
            pass
        else:
            raise AssertionError("unknown entry accepted")
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _in_subprocess(family):
    # the vision patches are installed on transformers classes for the life of the process: one process per family
    result = subprocess.run([sys.executable, __file__, family], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


def test_qwen3_vl_bitwise():
    _in_subprocess("qwen3")


def test_qwen3_5_bitwise():
    """Qwen3.5 needs transformers 5 (run with that environment); skipped on upstream's pinned 4.57."""
    try:
        import transformers.models.qwen3_5  # noqa: F401
    except ImportError:
        print("skipped qwen3_5: this transformers has no Qwen3.5")
        return
    _check("qwen3_5")


def test_qwen2_5_vl_bitwise():
    _in_subprocess("qwen2_5")


def test_flash_attention_backbone():
    _in_subprocess("flash")


def test_worker_prep():
    """Needs real processors: STARVLA_TEST_VLMS=<dir of Qwen3-VL-4B-Instruct>,<dir of Qwen2.5-VL-3B-Instruct>."""
    model_dirs = [d for d in os.environ.get("STARVLA_TEST_VLMS", "").split(",") if d]
    if not model_dirs:
        print("skipped worker_prep: STARVLA_TEST_VLMS is not set")
    for model_dir in model_dirs:
        _in_subprocess(model_dir)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            _check_worker_prep(sys.argv[1])
        else:
            {"flash": _check_flash_attention, "condition_once": _check_condition_once}.get(
                sys.argv[1], lambda: _check(sys.argv[1]))()
    else:
        test_selection()
        print("ok selection")
        test_qwen3_vl_bitwise()
        test_qwen2_5_vl_bitwise()
        test_qwen3_5_bitwise()
        test_flash_attention_backbone()
        test_condition_once()
        test_cast_while_flattening()
        test_compile_fallbacks()
        test_adamw_compiled()
        test_other_backbones()
        test_text_longest()
        test_worker_prep()

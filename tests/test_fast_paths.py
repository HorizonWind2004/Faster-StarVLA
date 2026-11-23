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
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=torch.randn(sum(h * w for h, w in grids), patch_dim),
        image_grid_thw=torch.tensor([[1, h, w] for h, w in grids]),
    )


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
    assert sorted(applied) == sorted(BACKBONE_PATHS) and not skipped, (applied, skipped)

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
                       "compile_lm", "compile_vision", "condition_once"], applied
    assert skipped == ["worker_prep"] and _implementations(model) == {"sdpa"}
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
        return loss.item(), torch.cat([p.grad.flatten() for p in dit.parameters() if p.grad is not None]), projected

    cases = [(layerwise, repeats) for layerwise in (False, True) for repeats in (4, 1)]
    before = [step(*case) for case in cases]
    assert condition_once.apply(None)
    for (layerwise, repeats), (loss, grads, projected) in zip(cases, before):
        fast_loss, fast_grads, fast_projected = step(layerwise, repeats)
        assert abs(fast_loss - loss) <= 1e-12 * abs(loss) and (fast_grads - grads).norm() <= 1e-10 * grads.norm()
        assert projected == [3 * repeats] * 2 and fast_projected == [3] * 2, (projected, fast_projected)
    print("ok condition_once: key / value projections on 3 rows instead of 12, loss and gradients equal to rounding")


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
        assert not fast.enabled("tf32") and not fast.enabled("reduce_to_owner")  # by name only
        os.environ["STARVLA_FAST"] = "all,tf32"
        assert fast.enabled("tf32") and fast.enabled("compile_lm")
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
        assert fast.selection() == fast.LEVELS["all"] - {"vision_attn_relaxed", "sdpa", "condition_once"}
        for name in ("COMPILE_LM", "COMPILE_VISION"):
            del os.environ["STARVLA_" + name]
        assert fast.selection() == fast.LEVELS["exact"]
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
        test_flash_attention_backbone()
        test_condition_once()
        test_worker_prep()

#!/usr/bin/env python3
"""Training configs for the frameworks that the LIBERO recipe's YAML does not cover by changing `framework.name`
alone: the LIBERO recipe (data, trainer) with the framework block of that framework's own config, sized for
Qwen3-VL-4B (hidden size 2560) and LIBERO (7-dim actions and state). LangForce and QwenDual ship no training config;
they get QwenGR00T's action head (the head they use) on top of their own defaults.

    python3 coverage/make_configs.py <starVLA tree> <out dir>
"""
import dataclasses
import sys
from pathlib import Path

from omegaconf import OmegaConf

tree, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(tree))
out.mkdir(parents=True, exist_ok=True)
HIDDEN = 2560
base = OmegaConf.load(tree / 'examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml')


def write(name, framework, **data):
    cfg = base.copy()
    cfg.framework = framework
    cfg.framework.qwenvl.vl_hidden_dim = HIDDEN
    cfg.framework.qwenvl.attn_implementation = 'sdpa'
    for key, value in {'action_dim': 7, 'state_dim': 7}.items():
        cfg.framework.action_model[key] = value
    for key, value in data.items():
        cfg.datasets.vla_data[key] = value
    OmegaConf.save(cfg, out / f'{name}.yaml')
    print(out / f'{name}.yaml')


def defaults(module, class_suffix='DefaultConfig'):
    import importlib

    mod = importlib.import_module(f'starVLA.model.framework.VLM4A.{module}')
    cls = next(c for n, c in vars(mod).items() if n.endswith(class_suffix) and dataclasses.is_dataclass(c))
    return OmegaConf.create(dataclasses.asdict(cls()))


adapter = OmegaConf.load(tree / 'starVLA/config/training/starvla_train_adapter.yaml').framework
adapter.action_model.hidden_dim = HIDDEN
write('adapter', adapter)

diffusion = OmegaConf.load(tree / 'starVLA/config/training/starvla_train_discrete_diffusion.yaml').framework
steps = int(base.framework.action_model.action_horizon)  # the head requires the chunk length the LIBERO data gives
diffusion.action_model.update(action_horizon=steps, future_action_window_size=steps - 1)
write('discrete_diffusion', diffusion)

gr00t_head = defaults('QwenGR00T').action_model  # the frameworks set its cross-attention width from the VLM
for name, module in (('LangForce', 'LangForce'), ('QwenDual', 'QwenDual')):
    framework = OmegaConf.merge(OmegaConf.create({'action_model': gr00t_head}), defaults(module), base.framework)
    framework.name = name
    # LangForce requires the bare instruction as the prompt
    write(name.lower(), framework, **({'CoT_prompt': '{instruction}'} if name == 'LangForce' else {}))

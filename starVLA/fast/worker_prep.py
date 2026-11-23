"""Build the model inputs of a training batch in the data-loader worker.

A framework's forward() turns a batch (list of samples) into VLM inputs with the VLM's processor: chat template,
tokenization, image resize / normalization / patchification. That is CPU work (about 0.19 s per step for QwenOFT at
batch 16 x 2 images), done in the training process while the GPU waits. The data-loader workers have the samples
first and spare time, so they run the same steps and attach the result to the batch:

    batch[0]["_prepared"] = {"inputs": {input_ids, attention_mask, pixel_values, image_grid_thw, ...}}

The batch stays the list of samples every other consumer expects (evaluation, statistics). In forward() the
framework takes `fast.prepared_inputs(examples, device)` when present and builds the inputs itself otherwise.

Input construction is specific to a framework (its prompt) and a processor, so each supported combination registers
a builder here with @builder("<framework name>"); `collate_for` returns None for a config no builder accepts, and
the default collate_fn is used. A builder must produce exactly what the framework's forward() would.
"""
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction

ACTION_TOKEN = "🔍"  # QwenOFT.action_token


_BUILDERS = {}  # framework name -> factory(cfg) -> collate_fn, or None if it does not handle this config


def builder(framework_name):
    def register(factory):
        _BUILDERS[framework_name] = factory
        return factory

    return register


def collate_for(cfg):
    """The preparing collate_fn for this config, or None when no builder handles it."""
    factory = _BUILDERS.get(str(cfg.framework.get("name", "")))
    return factory(cfg) if factory is not None else None


class QwenOFTInputs:
    """QwenOFT on a Qwen3-VL backbone. The prompt construction must stay identical to
    QwenOFT._build_training_inputs and _QWen3_VL_Interface.build_qwenvl_inputs. Also computes Qwen3-VL's multimodal
    rotary position ids, which the model otherwise derives at the start of every forward with a host loop."""

    def __init__(self, cfg):
        self.base_vlm = str(cfg.framework.qwenvl.base_vlm)
        self.chunk_len = int(cfg.framework.action_model.action_horizon)
        vla_data = cfg.datasets.vla_data
        self.cot_prompt = str(vla_data.get("CoT_prompt", "")) if "CoT_prompt" in vla_data else None
        self._processor = None
        self._rope = None

    def _rope_position_ids(self, inputs):
        """Qwen3-VL's multimodal rotary position ids of the batch: Qwen3VLModel.get_rope_index, which reads only the
        model config and these inputs. The model computes them at the start of every forward with a per-sample host
        loop unless they are passed in."""
        if self._rope is None:  # once per worker process
            from types import SimpleNamespace

            from transformers import AutoConfig, Qwen3VLModel

            self._rope = (Qwen3VLModel.get_rope_index, SimpleNamespace(config=AutoConfig.from_pretrained(self.base_vlm)))
        get_rope_index, owner = self._rope
        position_ids, _ = get_rope_index(
            owner, inputs["input_ids"], inputs.get("image_grid_thw"), None, attention_mask=inputs["attention_mask"]
        )
        return position_ids

    def _get_processor(self):
        if self._processor is None:  # once per worker process
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.base_vlm)
            self._processor.tokenizer.padding_side = "left"
        return self._processor

    def __call__(self, batch):
        batch_images = [example["image"] for example in batch]
        instructions = [example["lang"] for example in batch]
        state = [example["state"] for example in batch] if "state" in batch[0] else None
        if state is not None:
            instructions = add_discretized_state_to_instruction(instructions, state)
        action_tokens = ACTION_TOKEN * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = self.cot_prompt.replace("{instruction}", instruction) if self.cot_prompt is not None else instruction
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = self._get_processor().apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        qwen_inputs = dict(inputs)
        qwen_inputs["position_ids"] = self._rope_position_ids(qwen_inputs)
        batch[0]["_prepared"] = {"inputs": qwen_inputs}
        return batch


@builder("QwenOFT")
def _qwen_oft(cfg):
    base_vlm = str(cfg.framework.qwenvl.base_vlm)
    # only the VLMs that modules/vlm/get_vlm_model routes to the Qwen3-VL interface
    if "Qwen2.5-VL" in base_vlm or "nora" in base_vlm.lower() or "Qwen3-VL" not in base_vlm:
        return None
    return QwenOFTInputs(cfg)

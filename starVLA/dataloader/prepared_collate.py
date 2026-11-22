"""collate_fn that also builds the model inputs of a QwenOFT training batch in the data-loader worker.

QwenOFT.forward() turns a batch (list of samples) into Qwen3-VL inputs with the VLM's processor: chat template,
tokenization, image resize / normalization / patchification. That is CPU work, about 0.19 s per step at batch 16 x 2
images, done in the training process while the GPU waits. The data-loader workers have the samples first and spare
time, so with STARVLA_PREP_IN_WORKERS they run the same steps and attach the result to the batch:

    batch[0]["_prepared"] = {"qwen_inputs": {input_ids, attention_mask, pixel_values, image_grid_thw}}

The batch stays the list of samples every other consumer expects (evaluation, statistics); forward() uses the
prepared tensors when present and builds them itself otherwise. The prompt construction below must stay identical to
QwenOFT._build_training_inputs and _QWen3_VL_Interface.build_qwenvl_inputs.
"""
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction

ACTION_TOKEN = "🔍"  # QwenOFT.action_token


class QwenOFTPreparedCollate:
    def __init__(self, cfg):
        self.base_vlm = str(cfg.framework.qwenvl.base_vlm)
        self.chunk_len = int(cfg.framework.action_model.action_horizon)
        vla_data = cfg.datasets.vla_data
        self.cot_prompt = str(vla_data.get("CoT_prompt", "")) if "CoT_prompt" in vla_data else None
        self._processor = None

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
        batch[0]["_prepared"] = {"qwen_inputs": dict(inputs)}
        return batch


def prepared_collate_for(cfg):
    """The preparing collate_fn for this config, or None when the framework / VLM has none."""
    if str(cfg.framework.get("name", "")) != "QwenOFT":
        return None
    base_vlm = str(cfg.framework.qwenvl.base_vlm)
    # only the VLMs that modules/vlm/get_vlm_model routes to the Qwen3-VL interface
    if "Qwen2.5-VL" in base_vlm or "nora" in base_vlm.lower() or "Qwen3-VL" not in base_vlm:
        return None
    return QwenOFTPreparedCollate(cfg)

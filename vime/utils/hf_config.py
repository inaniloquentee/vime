"""HuggingFace config loading, including composite DeepSeek V4.1 checkpoints."""

from transformers import AutoConfig, DeepseekV3Config, PretrainedConfig


class DeepseekV41Config(DeepseekV3Config):
    model_type = "deepseek_v41"


def load_hf_config(path, **kwargs):
    config, _ = PretrainedConfig.get_config_dict(path, **kwargs)
    if config.get("model_type") not in {"deepseek_v41", "deepseek_v41_text"}:
        return AutoConfig.from_pretrained(path, **kwargs)
    text = dict(config.pop("text_config", {}))
    config.pop("vision_config", None)
    # Match Miles: keep top-level tokenizer/quantization metadata and flatten text.
    config = {**text, **config}
    config.setdefault("first_k_dense_replace", 0)
    return DeepseekV41Config.from_dict(config)

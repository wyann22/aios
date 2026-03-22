from .base import BaseLLMModel
from .config import ModelConfig
from .weight import load_weights


def create_model(model_path: str, config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if "qwen3" in model_name:
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(config)
    
    raise ValueError(f"Unsupported model: {model_path}")


__all__ = ["BaseLLMModel", "ModelConfig", "load_weights", "create_model"]

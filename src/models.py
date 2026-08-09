"""
Model registry for Paper A — Beyond Accuracy.
7 models from 0.5B to 8B. load_model() handles fp16 and 4-bit NF4.
"""

from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    size_b: float
    use_4bit: bool


MODEL_REGISTRY = [
    ModelConfig("Qwen2.5-0.5B",  "Qwen/Qwen2.5-0.5B-Instruct",                  0.5,  False),
    ModelConfig("Qwen2.5-1.5B",  "Qwen/Qwen2.5-1.5B-Instruct",                  1.5,  False),
    ModelConfig("Llama-3.2-1B",  "meta-llama/Llama-3.2-1B-Instruct",             1.0,  False),
    ModelConfig("Gemma-2-2B",    "google/gemma-2-2b-it",                          2.0,  False),
    ModelConfig("Qwen2.5-3B",    "Qwen/Qwen2.5-3B-Instruct",                     3.0,  False),
    ModelConfig("Llama-3.2-3B",  "meta-llama/Llama-3.2-3B-Instruct",             3.0,  False),
    ModelConfig("Llama-3.1-8B",  "meta-llama/Meta-Llama-3.1-8B-Instruct",        8.0,  True),
]

MODEL_NAMES = [m.name for m in MODEL_REGISTRY]
MODEL_SIZES = {m.name: m.size_b for m in MODEL_REGISTRY}


def get_model_by_name(name: str) -> ModelConfig:
    for m in MODEL_REGISTRY:
        if m.name == name:
            return m
    raise ValueError(f"Model '{name}' not in registry. Options: {[m.name for m in MODEL_REGISTRY]}")


def load_model(config: ModelConfig):
    """Load model + tokenizer. Returns (model, tokenizer)."""
    print(f"[models] Loading {config.name} ({config.hf_id}) ...")

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    print(f"[models] {config.name} loaded.")
    return model, tokenizer
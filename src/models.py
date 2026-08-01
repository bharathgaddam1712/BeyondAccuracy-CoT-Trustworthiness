"""
Model registry and loader for LIGHT paper experiments.
All models are loaded locally — no paid API calls.
"""

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str           # display name used in plots / CSVs
    hf_id: str          # HuggingFace model ID
    size_b: float       # parameter count in billions (for axis labeling)
    load_in_4bit: bool  # True only for 8B on T4 (VRAM ceiling)

MODEL_REGISTRY: list[ModelConfig] = [
    ModelConfig("Qwen2.5-0.5B",  "Qwen/Qwen2.5-0.5B-Instruct",                   0.5,  False),
    ModelConfig("Qwen2.5-1.5B",  "Qwen/Qwen2.5-1.5B-Instruct",                   1.5,  False),
    ModelConfig("Llama-3.2-1B",  "meta-llama/Llama-3.2-1B-Instruct",             1.0,  False),
    ModelConfig("Gemma-2-2B",    "google/gemma-2-2b-it",                          2.0,  False),
    ModelConfig("Qwen2.5-3B",    "Qwen/Qwen2.5-3B-Instruct",                     3.0,  False),
    ModelConfig("Llama-3.2-3B",  "meta-llama/Llama-3.2-3B-Instruct",             3.0,  False),
    ModelConfig("Llama-3.1-8B",  "meta-llama/Meta-Llama-3.1-8B-Instruct",        8.0,  True),
]

# Sizes in billions, sorted — used for x-axis of scaling plots
MODEL_SIZES = [m.size_b for m in MODEL_REGISTRY]
MODEL_NAMES  = [m.name   for m in MODEL_REGISTRY]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(config: ModelConfig, device: str = "auto"):
    """
    Load a model and its tokenizer.
    4-bit quantization is applied only when config.load_in_4bit is True.
    Returns (model, tokenizer).
    """
    print(f"[load] {config.name} ({config.hf_id})")

    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def get_model_by_name(name: str) -> ModelConfig:
    for m in MODEL_REGISTRY:
        if m.name == name:
            return m
    raise ValueError(f"Unknown model: {name}. Valid names: {MODEL_NAMES}")

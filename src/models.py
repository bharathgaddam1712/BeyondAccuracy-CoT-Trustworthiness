"""Original models; main experiments all use the SAME floating-point precision."""
from dataclasses import dataclass
@dataclass(frozen=True)
class ModelConfig:
    name:str
    hf_id:str
    size_b:float
    use_4bit:bool=False
MODEL_REGISTRY=[
    ModelConfig("Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct",0.5),
    ModelConfig("Qwen2.5-1.5B","Qwen/Qwen2.5-1.5B-Instruct",1.5),
    ModelConfig("Llama-3.2-1B","meta-llama/Llama-3.2-1B-Instruct",1.),
    ModelConfig("Gemma-2-2B","google/gemma-2-2b-it",2.),
    ModelConfig("Qwen2.5-3B","Qwen/Qwen2.5-3B-Instruct",3.),
    ModelConfig("Llama-3.2-3B","meta-llama/Llama-3.2-3B-Instruct",3.),
    ModelConfig("Llama-3.1-8B","meta-llama/Llama-3.1-8B-Instruct",8.)]
MODEL_NAMES=[m.name for m in MODEL_REGISTRY];MODEL_SIZES={m.name:m.size_b for m in MODEL_REGISTRY}
def get_model_by_name(name):return next(m for m in MODEL_REGISTRY if m.name==name)
def load_model(config,device="cuda:0",dtype="bfloat16",revision=None,nf4=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError("CUDA and a Blackwell-compatible PyTorch wheel are required")
    tokenizer=AutoTokenizer.from_pretrained(config.hf_id,revision=revision,trust_remote_code=False,use_fast=True)
    tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    kw=dict(revision=revision,trust_remote_code=False,torch_dtype=getattr(torch,dtype),attn_implementation="sdpa")
    if nf4:kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=getattr(torch,dtype),bnb_4bit_use_double_quant=True),device_map={"":device})
    model=AutoModelForCausalLM.from_pretrained(config.hf_id,**kw)
    if not nf4:model=model.to(device)
    model.eval();return model,tokenizer

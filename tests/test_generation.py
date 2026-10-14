"""Exercise generation bookkeeping without requiring a local CUDA installation."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from src.inference import generate_batch

class Tensor:
    def __init__(self,value):self.value=np.array(value)
    @property
    def shape(self):return self.value.shape
    def __getitem__(self,key):return Tensor(self.value[key])
    def bool(self):return self.value.astype(bool)
    def tolist(self):return self.value.tolist()
    def __iter__(self):return (Tensor(x) for x in self.value)

class Encoding(dict):
    def __getattr__(self,key):return self[key]
    def to(self,device):return self

class Tokenizer:
    name_or_path="fake-model"
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):return "question"
    def __call__(self,texts,**kwargs):
        assert kwargs["truncation"] is False
        return Encoding(input_ids=Tensor([[0,7,8],[7,8,9]]),attention_mask=Tensor([[0,1,1],[1,1,1]]))
    def decode(self,ids,skip_special_tokens=True):return "".join(chr(i) for i in ids)

def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(manual_seed=lambda seed:None,cuda=SimpleNamespace(manual_seed_all=lambda seed:None),inference_mode=nullcontext))

def test_per_sequence_eos_and_left_padding(monkeypatch):
    fake_torch(monkeypatch)
    model=SimpleNamespace(device="fake",config=SimpleNamespace(max_position_embeddings=100),generation_config=SimpleNamespace(eos_token_id=[2,3]),
                          generate=lambda **kw:Tensor([[0,7,8,65,3,0,0],[7,8,9,66,67,68,2]]))
    records=generate_batch(model,Tokenizer(),[[dict(role="user",content="q")]]*2,max_new_tokens=4)
    assert records[0]["prompt_token_ids"]==[7,8]
    assert records[0]["generated_token_ids"]==[65]
    assert records[1]["generated_token_ids"]==[66,67,68]
    assert all(r["stopped_eos"] for r in records)
    assert not any(r["hit_token_cap"] for r in records)

def test_context_overflow_is_not_truncated(monkeypatch):
    fake_torch(monkeypatch)
    model=SimpleNamespace(device="fake",config=SimpleNamespace(max_position_embeddings=5))
    with pytest.raises(ValueError,match="Context overflow"):
        generate_batch(model,Tokenizer(),[[dict(role="user",content="q")]]*2,max_new_tokens=4)

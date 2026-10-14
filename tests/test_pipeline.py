import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.analysis import compare,consistency_analysis,cru_analysis,self_consistency_analysis
from src.probing import locate_positions,out_of_fold
from src.protocol import atomic_json,digest,read_jsonl,append_jsonl
from src.runner import reasoning_only,trace_steps,score_record,perform_jobs

C=dict(n_bootstrap=30,n_permutations=30,seed=42,self_consistency_k=5,probe_min_class=5,probe_folds=5)

def row(mode,qid,p,y,axis="calibration"):
    return dict(model="Qwen2.5-0.5B",dataset="gsm8k",axis=axis,id=qid,mode=mode,confidence=p,correct=y,answer_parsed=True,confidence_parsed=p is not None)

def test_pairing_uses_ids_and_recomputes_ece():
    data=pd.DataFrame([row("direct","a",1,0),row("direct","b",0,1),row("cot","b",1,1),row("cot","a",0,0)])
    result=compare(data,"direct","cot",C)
    ece=next(r for r in result if r["metric"]=="ece")
    assert ece["delta"]==-1
    assert ece["n"]==2

def test_common_confidence_mask_is_explicit():
    data=pd.DataFrame([row("direct","a",1,0),row("direct","b",None,1),row("cot","a",0,0),row("cot","b",1,1)])
    ece=next(r for r in compare(data,"direct","cot",C) if r["metric"]=="ece")
    assert ece["n"]==1 and ece["n_intersection"]==2

def test_cru_not_interpreted_when_structured_is_worse():
    rows=[]
    for mode,p in [("trace_empty",.2),("trace_filler",.1),("trace_structured",.8),("trace_shuffled",.7)]:
        for qid in ("a","b"):rows.append(row(mode,qid,p,0,"ablation"))
    out=cru_analysis(pd.DataFrame(rows),C).iloc[0]
    assert pd.isna(out.cru)
    assert "not_better" in out.regime

def test_partial_consistency_fails():
    data=pd.DataFrame([dict(model="m",dataset="gsm8k",mode="direct",id="a",axis="consistency",variant=0,predicted="1",correct=True)])
    with pytest.raises(RuntimeError,match="Incomplete"):consistency_analysis(data,C)

def test_strict_scoring_no_substring_credit():
    item=dict(id="a",dataset="gsm8k",answer="1")
    assert score_record(item,"Answer: 10\nConfidence: 100","direct")["correct"] is False

def test_full_jsonl_survives_large_response_and_corruption_is_detected(tmp_path):
    p=tmp_path/"raw.jsonl";r=dict(raw_response="reasoning "*1000)
    append_jsonl(p,r);assert read_jsonl(p)==[r]
    with p.open("a") as f:f.write('{"incomplete":')
    with pytest.raises(RuntimeError,match="Incomplete JSONL"):read_jsonl(p)

def test_trace_removes_answer_and_confidence():
    assert reasoning_only("Step 1.\nStep 2.\nAnswer: 12\nConfidence: 99")=="Step 1.\nStep 2."

class CharTokenizer:
    def decode(self,ids,skip_special_tokens=True):return "".join(chr(i) for i in ids)

def test_probe_positions_precede_answer_and_confidence_without_padding():
    text="Reasoning\nAnswer: 12\nConfidence: 99"
    r=dict(prompt_token_ids=[1,2,3],generated_token_ids=[ord(x) for x in text])
    pos=locate_positions(CharTokenizer(),r)
    assert pos["prompt_end"]==2
    assert pos["pre_answer"]==3+text.index("12")-1
    assert pos["answer_end"]==3+text.index("12")+1
    assert pos["sequence_end"]==3+len(text)-1

def test_probe_scaler_and_classifier_are_out_of_fold():
    rng=np.random.default_rng(2);y=np.tile([0,1],30);X=rng.normal(size=(60,8));X[:,0]+=4*y
    scores,folds=out_of_fold(X,y,C)
    assert np.isfinite(scores).all() and set(folds)==set(range(5))
    assert len(scores)==len(y)

def test_no_duplicate_resume_keys(tmp_path):
    path=tmp_path/"raw"/"Qwen2.5-0.5B"/"accuracy.jsonl"
    r=dict(key="same",prompt_hash="hash");append_jsonl(path,r);append_jsonl(path,r)
    with pytest.raises(RuntimeError,match="Duplicate"):
        perform_jobs(C,tmp_path,None,None,"Qwen2.5-0.5B",[],"accuracy")

def test_incomplete_generation_batch_never_saves_misaligned_records(tmp_path,monkeypatch):
    monkeypatch.setattr("src.runner.generate_batch",lambda *args,**kwargs:[])
    config=dict(C,batch_size=2,max_new_tokens=10)
    jobs=[dict(key="a",messages=[dict(role="user",content="q")])]
    with pytest.raises(RuntimeError,match="different number of responses"):
        perform_jobs(config,tmp_path,None,None,"Qwen2.5-0.5B",jobs,"accuracy")
    assert read_jsonl(tmp_path/"raw"/"Qwen2.5-0.5B"/"accuracy.jsonl")==[]

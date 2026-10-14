"""CPU-only synthetic run: raw rows -> summaries -> matching plot data -> validation."""
import json
import numpy as np
import pandas as pd
from src import protocol
from src.analysis import analyze
from src.runner import score_record
from src.prompts import make_prompt

def test_complete_synthetic_run(tmp_path,monkeypatch):
    monkeypatch.setattr(protocol,"ROOT",tmp_path)
    monkeypatch.setattr(protocol,"source_hash",lambda:"fixture-source")
    c=dict(run_name="synthetic",seed=42,models=["Qwen2.5-0.5B"],consistency_datasets=["gsm8k"],
           ablation_models=[],probe_models=[],quantization_model="Llama-3.1-8B",n_bootstrap=10,n_permutations=10,self_consistency_k=5)
    root=protocol.run_dir(c)
    samples=[dict(id=f"q{i}",dataset="gsm8k",question=f"What is {i}+1?",answer=str(i+1),choices=None) for i in range(2)]
    protocol.atomic_json(root/"datasets.json",samples)
    protocol.atomic_json(root/"manifest.json",dict(config_hash=protocol.digest(c),source_hash="fixture-source",dataset_hash=protocol.digest(samples)))
    paras={s["id"]:[f"Question rewrite {label}: What is {i}+1?" for label in ["a","b","c","d","e"]] for i,s in enumerate(samples)}
    protocol.atomic_json(root/"paraphrases.json",paras)
    protocol.atomic_json(root/"paraphrases.lock.json",dict(hash=protocol.digest(paras)))
    for s in samples:
        for mode in ["direct","cot"]:
            for axis in ["accuracy","calibration","consistency"]:
                for variant in range(6 if axis=="consistency" else 1):
                    text=f"Answer: {s['answer']}"+("\nConfidence: 100" if axis=="calibration" else "")
                    r=dict(key=f"{axis}/{mode}/{s['id']}/{variant}",model="Qwen2.5-0.5B",dataset="gsm8k",id=s["id"],mode=mode,axis=axis,
                           size_b=.5,precision="main",variant=variant,gold=s["answer"],question=s["question"],choices=None,subtask=None,
                           raw_response=text,input_tokens=20,generated_tokens=8,stopped_eos=True,hit_token_cap=False,
                           **score_record(s,text,mode))
                    protocol.append_jsonl(root/"raw"/"Qwen2.5-0.5B"/f"{axis}.jsonl",r)
    report=analyze(c,require_complete=True)
    assert report["status"]=="passed"
    cells=pd.read_csv(root/"summary/metrics_by_cell.csv")
    plotted=pd.read_csv(root/"figures/calibration_gsm8k.csv")
    actual=cells[cells.axis=="calibration"].sort_values("mode")
    assert np.allclose(plotted.sort_values("mode").ece,actual.ece)
    consistency=pd.read_csv(root/"summary/consistency_paired.csv")
    assert consistency.delta.iloc[0]==0

"""GPU experiments. Single writer per model; complete JSONL checkpoints."""
import gc
import json
import random
import re
import time
from pathlib import Path
from src.protocol import check_manifest,validate_paraphrases,read_jsonl,append_jsonl,digest
from src.models import get_model_by_name,load_model
from src.inference import generate_batch,parse_dataset_answer,parse_confidence,check_correctness,PARSER_VERSION
from src.prompts import make_prompt,PROMPT_VERSION

def safe_name(name):return re.sub(r"[^A-Za-z0-9.-]","_",name)

def selected(samples,n):
    """Fixed hash selection, balanced across BBH subtasks, independent of outcomes."""
    out=[]
    for ds in sorted({s["dataset"] for s in samples}):
        items=[s for s in samples if s["dataset"]==ds]
        buckets={sub:sorted([s for s in items if s.get("subtask","")==sub],key=lambda s:digest(s["id"])) for sub in sorted({s.get("subtask","") for s in items})}
        chosen=[];offset=0
        while len(chosen)<min(n,len(items)):
            for bucket in buckets.values():
                if offset<len(bucket) and len(chosen)<n:chosen.append(bucket[offset])
            offset+=1
        out.extend(chosen)
    return out

def score_record(item,response,mode):
    prediction=parse_dataset_answer(response,item["dataset"],mode,item.get("choices"),item.get("subtask"))
    confidence=parse_confidence(response,mode)
    return dict(predicted=prediction,correct=check_correctness(prediction,item["answer"],item["dataset"]),
                confidence=confidence,answer_parsed=prediction is not None,confidence_parsed=confidence is not None)

def perform_jobs(c,root,model,tok,model_name,jobs,file_name,precision="main"):
    path=root/"raw"/safe_name(model_name)/f"{file_name}.jsonl"
    old=read_jsonl(path);existing={r["key"]:r for r in old}
    if len(existing)!=len(old):raise RuntimeError(f"Duplicate result keys in {path}")
    for job in jobs:
        if job["key"] in existing and existing[job["key"]]["prompt_hash"]!=digest(job["messages"]):
            raise RuntimeError("Resume prompt differs; use a new run")
    pending=[j for j in jobs if j["key"] not in existing]
    for start in range(0,len(pending),c["batch_size"]):
        batch=pending[start:start+c["batch_size"]];t0=time.perf_counter()
        # Sampling streams use a separate per-question call: resume/batch size cannot change seeds.
        if batch and batch[0].get("sample",False):
            outputs=[generate_batch(model,tok,[j["messages"]],j.get("max_tokens",c["max_new_tokens"]),True,j["seed"])[0] for j in batch]
        else:
            outputs=generate_batch(model,tok,[j["messages"] for j in batch],batch[0].get("max_tokens",c["max_new_tokens"]))
        if len(outputs)!=len(batch):
            raise RuntimeError("Generation returned a different number of responses than requested; no records from this batch were saved")
        elapsed=time.perf_counter()-t0
        append_jsonl(root/"performance"/f"{safe_name(model_name)}.jsonl",dict(stage=file_name,precision=precision,seconds=elapsed,n_responses=len(batch),generated_tokens=sum(o["generated_tokens"] for o in outputs),input_tokens=sum(o["input_tokens"] for o in outputs)))
        for j,out in zip(batch,outputs):
            item=j["item"];mode=j["mode"]
            record=dict(key=j["key"],id=item["id"],dataset=item["dataset"],subtask=item.get("subtask"),model=model_name,
                        size_b=get_model_by_name(model_name).size_b,mode=mode,axis=j["axis"],precision=precision,
                        variant=j.get("variant",0),sample_index=j.get("sample_index",0),gold=item["answer"],
                        question=item["question"],choices=item.get("choices"),messages=j["messages"],
                        prompt_hash=digest(j["messages"]),prompt_version=PROMPT_VERSION,parser_version=PARSER_VERSION,
                        batch_seconds=elapsed,batch_size=len(batch),**out,**score_record(item,out["raw_response"],mode))
            record.update(j.get("extra",{}));append_jsonl(path,record);existing[j["key"]]=record
        print(f"{model_name} {file_name}: {min(start+len(batch),len(pending))}/{len(pending)} new responses",flush=True)
    return [existing[j["key"]] for j in jobs]

def baseline_jobs(samples,axis,modes=("direct","cot")):
    return [dict(key=f"{axis}/{mode}/{s['id']}/0",item=s,mode=mode,axis=axis,
                 messages=make_prompt(s["question"],mode,s["dataset"],s.get("choices"),confidence=axis!="accuracy"))
            for mode in modes for s in samples]

def behavioral(c,root,model,tok,name,samples):
    perform_jobs(c,root,model,tok,name,baseline_jobs(samples,"accuracy"),"accuracy")
    perform_jobs(c,root,model,tok,name,baseline_jobs(samples,"calibration"),"calibration")

def consistency(c,root,model,tok,name,samples):
    paras=validate_paraphrases(c,root);jobs=[]
    for s in samples:
        if s["dataset"] not in c["consistency_datasets"]:continue
        for mode in ("direct","cot"):
            for variant,text in enumerate([s["question"]]+paras[s["id"]]):
                item=dict(s,question=text)
                jobs.append(dict(key=f"consistency/{mode}/{s['id']}/{variant}",item=item,mode=mode,axis="consistency",variant=variant,
                                 messages=make_prompt(text,mode,s["dataset"],s.get("choices")),extra=dict(original_question=s["question"])))
    perform_jobs(c,root,model,tok,name,jobs,"consistency")

def reasoning_only(response):
    # Strip explicit final answer and confidence fields, retaining the actual trace verbatim.
    positions=[m.start() for m in re.finditer(r"(?im)^\s*(?:final\s+)?(?:answer|confidence)\s*:|^\s*####",response)]
    return response[:min(positions)].strip() if positions else None

def trace_steps(text):
    # Keep separators with each step so permutations preserve the character inventory.
    steps=text.splitlines(keepends=True)
    if len([s for s in steps if s.strip()])<2:
        steps=re.findall(r".+?(?:[.!?](?:\s+|$)|$)",text,flags=re.S)
    return [s for s in steps if s]

def match_filler(tok,trace):
    target=len(tok.encode(trace,add_special_tokens=False))
    # Inert dots rather than an extra instruction to reason. Record exact token lengths.
    for repeat in range(max(1,target-8),target*5+20):
        value=" ."*repeat
        count=len(tok.encode(value,add_special_tokens=False))
        if count==target:return value
        if count>target+4:break
    return None

def interventions(c,root,model,tok,name,samples):
    origins={r["id"]:r for r in read_jsonl(root/"raw"/safe_name(name)/"calibration.jsonl") if r["mode"]=="cot"}
    jobs=[];eligibility=[]
    for s in selected(samples,c["ablation_n_per_dataset"]):
        source=origins.get(s["id"]);trace=reasoning_only(source["raw_response"]) if source else None
        row=dict(id=s["id"],dataset=s["dataset"],model=name,eligible=False,reason="missing_trace")
        if trace and source["stopped_eos"]:
            steps=trace_steps(trace);target=len(tok.encode(trace,add_special_tokens=False));shuffled=None
            rng=random.Random(c["seed"]+s["source_row"])
            for _ in range(100):
                order=list(steps);rng.shuffle(order);candidate="".join(order)
                if candidate!=trace and len(tok.encode(candidate,add_special_tokens=False))==target:
                    shuffled=candidate;break
            filler=match_filler(tok,trace)
            row["reason"]="no_changed_token_matched_permutation" if shuffled is None else "no_token_matched_filler"
            if shuffled is not None and filler is not None:
                row.update(eligible=True,reason="eligible",trace_tokens=target,n_steps=len(steps))
                for mode,text in (("trace_empty",""),("trace_structured",trace),("trace_shuffled",shuffled),("trace_filler",filler)):
                    jobs.append(dict(key=f"ablation/{mode}/{s['id']}/0",item=s,mode=mode,axis="ablation",
                                     messages=make_prompt(s["question"],"direct",s["dataset"],s.get("choices"),True,text),
                                     extra=dict(source_key=source["key"],trace_text=text,trace_tokens=len(tok.encode(text,add_special_tokens=False)),n_steps=len(steps))))
        eligibility.append(row)
    from src.protocol import atomic_json
    atomic_json(root/"raw"/safe_name(name)/"ablation_eligibility.json",eligibility)
    perform_jobs(c,root,model,tok,name,jobs,"ablation")
    # Independent sample-and-vote baseline. Confidence = vote share of selected answer.
    jobs=[]
    for s in selected(samples,c["ablation_n_per_dataset"]):
        for i in range(c["self_consistency_k"]):
            jobs.append(dict(key=f"self_consistency/cot/{s['id']}/{i}",item=s,mode="cot",axis="self_consistency",sample=True,
                             sample_index=i,seed=c["seed"]+s["source_row"]*100+i,
                             messages=make_prompt(s["question"],"cot",s["dataset"],s.get("choices"))))
    perform_jobs(c,root,model,tok,name,jobs,"self_consistency")

def quantization(c,root,name,samples,device,manifest):
    import torch
    items=selected(samples,c["quantization_n_per_dataset"])
    for precision,nf4,dtype in (("nf4",True,c["dtype"]),("float16",False,"float16")):
        model,tok=load_model(get_model_by_name(name),device,dtype,manifest["model_revisions"][get_model_by_name(name).hf_id],nf4)
        perform_jobs(c,root,model,tok,name,baseline_jobs(items,"quantization"),f"quantization_{precision}",precision)
        del model,tok;gc.collect();torch.cuda.empty_cache()

def run_worker(c,names,device,stages):
    import torch
    root,manifest=check_manifest(c);samples=json.loads((root/"datasets.json").read_text());torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    print(f"Worker {device}: {free/2**30:.1f}/{total/2**30:.1f} GiB free; models {names}",flush=True)
    if free/2**30<24:raise RuntimeError("Less than 24 GiB free. Wait for other GPU jobs to finish")
    for name in names:
        model,tok=load_model(get_model_by_name(name),device,c["dtype"],manifest["model_revisions"][get_model_by_name(name).hf_id])
        if "behavioral" in stages:behavioral(c,root,model,tok,name,samples)
        if "consistency" in stages:consistency(c,root,model,tok,name,samples)
        if "ablations" in stages and name in c["ablation_models"]:interventions(c,root,model,tok,name,samples)
        if "probes" in stages and name in c["probe_models"]:
            from src.probing import extract_features
            extract_features(c,root,model,tok,name,samples)
        del model,tok;gc.collect();torch.cuda.empty_cache()
        if "quantization" in stages and name==c["quantization_model"]:quantization(c,root,name,samples,device,manifest)

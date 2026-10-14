"""Immutable run manifests, frozen subsets, screened paraphrases, append-only outputs."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
from pathlib import Path
from collections import Counter
from src.models import MODEL_NAMES,ModelConfig,load_model
from src.inference import generate_batch
from src.prompts import make_paraphrase_gen_prompt

ROOT=Path(__file__).resolve().parents[1]
SCHEMA_VERSION=3

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def source_hash():
    paths=[ROOT/"run_pipeline.py"]+sorted((ROOT/"src").glob("*.py"))
    return digest({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    tmp.replace(path)

def read_jsonl(path):
    if not Path(path).exists():return []
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError as exc:raise RuntimeError(f"Incomplete JSONL line {path}:{i}; recover explicitly before resuming") from exc
    return rows

def append_jsonl(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n");f.flush()

def load_config(path):
    c=json.loads(Path(path).read_text(encoding="utf-8"))
    if set(c["models"])-set(MODEL_NAMES):raise ValueError("Unknown model in config")
    for key in ("ablation_models","probe_models"):
        if set(c[key])-set(c["models"]):raise ValueError(f"{key} must be a subset of models")
    if len(c["models"])!=len(set(c["models"])):raise ValueError("Repeated model names")
    if c["dtype"] not in ("float16","bfloat16"):raise ValueError("Use uniform float16 or bfloat16")
    if c["paraphrases_per_question"]!=5:raise ValueError("Protocol fixes five paraphrases plus the original")
    if min(c["batch_size"],c["max_new_tokens"],c["n_bootstrap"])<1:raise ValueError("Invalid run settings")
    return c

def run_dir(c):
    name=c["run_name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+",name):raise ValueError("Unsafe run name")
    return ROOT/"results"/name

def check_manifest(c):
    root=run_dir(c);m=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if m["config_hash"]!=digest(c) or m["source_hash"]!=source_hash():
        raise RuntimeError("Code/config differs from this run. Choose a NEW run_name; do not mix protocols")
    if m["dataset_hash"]!=digest(json.loads((root/"datasets.json").read_text(encoding="utf-8"))):
        raise RuntimeError("Frozen dataset file changed")
    return root,m

def freeze_datasets(c):
    """Fresh HF snapshots; selected row indices and MCQ order are saved verbatim."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    rng=random.Random(c["seed"]);api=HfApi();revisions={};samples=[]
    specifications=[("gsm8k","openai/gsm8k","main","test",c["n_gsm8k"]),
                    ("truthfulqa","truthfulqa/truthful_qa","multiple_choice","validation",c["n_truthfulqa"]),
                    ("arc","allenai/ai2_arc","ARC-Challenge","test",c["n_arc"])]
    for task in c["bbh_subtasks"]:
        specifications.append(("bbh","lukaemon/bbh",task,"test",c["n_bbh_per_subtask"]))
    for ds,repo,subset,split,n in specifications:
        if repo not in revisions:revisions[repo]=api.dataset_info(repo).sha
        data=load_dataset(repo,subset,split=split,revision=revisions[repo])
        if n>len(data):raise ValueError(f"Requested {n} but {repo}/{subset} has {len(data)}")
        for idx in sorted(rng.sample(range(len(data)),n)):
            row=data[idx];item=dict(dataset=ds,id=f"{ds}_{subset}_{idx}",source_repo=repo,source_revision=revisions[repo],source_subset=subset,source_split=split,source_row=idx)
            if ds=="gsm8k":item.update(question=row["question"],answer=row["answer"].split("####")[-1].strip(),choices=None)
            elif ds=="bbh":item.update(question=row["input"],answer=row["target"].strip().strip("()"),choices=None,subtask=subset)
            elif ds=="truthfulqa":
                pairs=list(zip(row["mc1_targets"]["choices"],row["mc1_targets"]["labels"]));rng.shuffle(pairs)
                if sum(y for _,y in pairs)!=1:raise ValueError("MC1 must have one correct choice")
                item.update(question=row["question"],choices=[s for s,_ in pairs],answer=chr(65+next(i for i,(_,y) in enumerate(pairs) if y==1)))
            else:
                labels=row["choices"]["label"];texts=row["choices"]["text"]
                item.update(question=row["question"],choices=texts,answer=chr(65+labels.index(row["answerKey"])))
            samples.append(item)
    return samples,revisions

def prepare(c):
    root=run_dir(c)
    if (root/"manifest.json").exists():return check_manifest(c)[0]
    if root.exists() and any(root.iterdir()):raise RuntimeError("Partially prepared run: use another run_name or inspect/remove its directory")
    from huggingface_hub import HfApi
    from src.models import get_model_by_name
    api=HfApi();repos=[get_model_by_name(m).hf_id for m in c["models"]]+[c["paraphrase_generator"],c["paraphrase_verifier"]]
    revisions={repo:api.model_info(repo).sha for repo in sorted(set(repos))}
    # Verify gated repositories before a long run. Model metadata alone is public.
    from huggingface_hub import hf_hub_download
    for repo,rev in revisions.items():hf_hub_download(repo,"config.json",revision=rev)
    samples,dsrevs=freeze_datasets(c)
    atomic_json(root/"datasets.json",samples)
    packages={}
    for name in ("torch","transformers","datasets","numpy","pandas","scikit-learn","bitsandbytes"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    atomic_json(root/"manifest.json",dict(schema_version=SCHEMA_VERSION,config=c,config_hash=digest(c),source_hash=source_hash(),dataset_hash=digest(samples),model_revisions=revisions,dataset_revisions=dsrevs,packages=packages,platform=platform.platform()))
    return root

def numbers(text):return Counter(re.findall(r"[+-]?\d+(?:[.,]\d+)*",text))

def parse_paraphrase_array(text):
    text=re.sub(r"^```(?:json)?\s*|\s*```$","",text.strip())
    try:value=json.loads(text)
    except json.JSONDecodeError:return []
    return value if isinstance(value,list) and all(isinstance(s,str) for s in value) else []

def paraphrase_screen(original,candidate):
    if len(candidate.strip())<10:return "too_short"
    if candidate.strip().casefold()==original.strip().casefold():return "unchanged"
    if numbers(original)!=numbers(candidate):return "numeric_mismatch"
    if re.search(r"(?i)\b(?:answer|confidence)\s*:",candidate):return "answer_leakage"
    return None

def generate_paraphrases(c,device="cuda:0"):
    import gc,torch
    root,m=check_manifest(c);samples=json.loads((root/"datasets.json").read_text())
    targets=[s for s in samples if s["dataset"] in c["consistency_datasets"]]
    path=root/"paraphrases.json"
    accepted=json.loads(path.read_text()) if path.exists() else {}
    lock=root/"paraphrases.lock.json"
    if lock.exists() and json.loads(lock.read_text())["hash"]!=digest(accepted):
        raise RuntimeError("Screened paraphrases were edited. Use a new run rather than relocking changed text")
    pending=[s for s in targets if s["id"] not in accepted]
    if pending:
        previous=read_jsonl(root/"paraphrase_generation.jsonl")
        previous_by_id={}
        for entry in previous:previous_by_id.setdefault(entry["item"]["id"],[]).append(entry)
        generator,tok=load_model(ModelConfig("paraphraser",c["paraphrase_generator"],7),device,c["dtype"],m["model_revisions"][c["paraphrase_generator"]])
        candidates={}
        for s in pending:
            history=previous_by_id.get(s["id"],[])
            values=list(dict.fromkeys(v for entry in history for v in entry["candidates"]));attempts=[]
            round_index=len(history)
            for attempt in range(c["paraphrase_attempts"]):
                result=generate_batch(generator,tok,[make_paraphrase_gen_prompt(s["question"])],c["paraphrase_max_tokens"],sample=attempt>0 or round_index>0,seed=c["seed"]+s["source_row"]*31+attempt+round_index*100000)[0]
                attempts.append(result["raw_response"])
                for value in parse_paraphrase_array(result["raw_response"]):
                    if paraphrase_screen(s["question"],value) is None and value.casefold() not in {x.casefold() for x in values}:values.append(value.strip())
                if len(values)>=10:break
            candidates[s["id"]]=dict(item=s,candidates=values,attempts=attempts,retry_round=round_index)
            append_jsonl(root/"paraphrase_generation.jsonl",candidates[s["id"]])
            print(f"Paraphrase candidates {s['id']}: {len(values)}",flush=True)
        del generator,tok;gc.collect();torch.cuda.empty_cache()
        verifier,tok=load_model(ModelConfig("verifier",c["paraphrase_verifier"],8),device,c["dtype"],m["model_revisions"][c["paraphrase_verifier"]])
        failures=[]
        for qid,entry in candidates.items():
            keep=[];judgements=[]
            for value in entry["candidates"]:
                prompt=[dict(role="system",content="Check semantic equivalence of two questions in BOTH directions. They must have exactly the same answer and preserve all entities, negations, conditions and units. Do not solve either question. Reply with only JSON: {\"equivalent\": true or false, \"reason\": \"brief explanation\"}."),dict(role="user",content=f"Original: {entry['item']['question']}\nRewrite: {value}")]
                raw=generate_batch(verifier,tok,[prompt],512)[0]["raw_response"]
                try:judgement=json.loads(re.sub(r"^```(?:json)?\s*|\s*```$","",raw.strip()))
                except json.JSONDecodeError:judgement={"equivalent":False,"reason":"invalid_verifier_output"}
                judgements.append(dict(paraphrase=value,judgement=judgement,raw=raw))
                if judgement.get("equivalent") is True:keep.append(value)
                if len(keep)==5:break
            append_jsonl(root/"paraphrase_verification.jsonl",dict(id=qid,judgements=judgements))
            if len(keep)==5:accepted[qid]=keep;atomic_json(path,accepted)
            else:failures.append(qid)
        del verifier,tok;gc.collect();torch.cuda.empty_cache()
        if failures:raise RuntimeError(f"Five screened variants unavailable for {len(failures)} questions. Re-run paraphrases to retry; no singleton silently enters evaluation. First failures: {failures[:10]}")
    atomic_json(root/"paraphrases.lock.json",dict(hash=digest(accepted),generator=c["paraphrase_generator"],verifier=c["paraphrase_verifier"],human_verified=False))
    validate_paraphrases(c,root)
    rng=random.Random(c["seed"])
    audit=[]
    for ds in c["consistency_datasets"]:
        pool=[dict(id=s["id"],dataset=ds,original=s["question"],variant=i+1,paraphrase=p,human_equivalent="",notes="") for s in targets if s["dataset"]==ds for i,p in enumerate(accepted[s["id"]])]
        audit.extend(rng.sample(pool,min(c["human_audit_per_dataset"],len(pool))))
    review=root/"paraphrase_human_audit.csv"
    if not review.exists():
        with review.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(audit[0]) if audit else ["id"]);writer.writeheader();writer.writerows(audit)
    print(f"Paraphrases ready. Human fidelity sample: {review}",flush=True)

def validate_paraphrases(c,root):
    values=json.loads((root/"paraphrases.json").read_text());samples=json.loads((root/"datasets.json").read_text())
    lock=root/"paraphrases.lock.json"
    if not lock.exists() or json.loads(lock.read_text())["hash"]!=digest(values):
        raise RuntimeError("Paraphrases changed after screening or lack a screening lock")
    for s in samples:
        if s["dataset"] not in c["consistency_datasets"]:continue
        variants=values.get(s["id"],[])
        if len(variants)!=5 or len(set(v.casefold() for v in variants))!=5:raise RuntimeError(f"Incomplete paraphrases for {s['id']}")
        if any(paraphrase_screen(s["question"],v) for v in variants):raise RuntimeError(f"Invalid paraphrases for {s['id']}")
    return values

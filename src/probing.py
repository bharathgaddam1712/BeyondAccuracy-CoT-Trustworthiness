"""Correctness probes at explicit positions, with saved out-of-fold predictions."""
import json
import math
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.protocol import atomic_json,digest,read_jsonl
from src.statistics import paired_metric,interval,holm
from src.metrics import compute_auroc

def locate_positions(tokenizer,record):
    """Token indices in prompt+generated sequence; never batch padding positions."""
    prompt=record["prompt_token_ids"];generated=record["generated_token_ids"]
    positions={"prompt_end":len(prompt)-1,"sequence_end":len(prompt)+len(generated)-1}
    # Decode prefixes of the ORIGINAL token IDs. Re-tokenization could change boundaries.
    text=tokenizer.decode(generated,skip_special_tokens=True)
    fields=list(re.finditer(r"(?im)^[ \t]*(?:final[ \t]+)?answer[ \t]*:[ \t]*([^\n]+)",text))
    if not fields:return positions
    match=fields[-1];start,end=match.span(1)
    spans=[len(tokenizer.decode(generated[:i],skip_special_tokens=True)) for i in range(len(generated)+1)]
    first=next((i for i in range(len(generated)) if spans[i+1]>start),None)
    last=next((i for i in range(len(generated)) if spans[i+1]>=end),None)
    if first is not None:positions["pre_answer"]=len(prompt)+first-1
    if last is not None:positions["answer_end"]=len(prompt)+last
    return positions

def extract_features(c,root,model,tok,name,samples):
    import torch
    from src.runner import selected,baseline_jobs,perform_jobs,safe_name
    items=selected(samples,c["probe_n_per_dataset"])
    records=perform_jobs(c,root,model,tok,name,baseline_jobs(items,"probe",c["probe_modes"]),"probe")
    layers=sorted(set(max(1,round(model.config.num_hidden_layers*f)) for f in c["probe_layer_fractions"]))
    primary=max(1,round(model.config.num_hidden_layers*c["probe_primary_fraction"]))
    if primary not in layers:raise ValueError("Primary probe layer must be in extraction layers")
    folder=root/"activations"/safe_name(name);folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/"metadata.json",dict(layers=layers,primary_layer=primary,positions=["prompt_end","pre_answer","answer_end","sequence_end"],
                 note="pre_answer precedes the explicit answer field; prompt_end is pre-generation. sequence_end is exploratory and may contain confidence."))
    for i,r in enumerate(records):
        path=folder/f"{digest(r['key'])}.npz"
        if path.exists():continue
        t0=time.perf_counter()
        ids=r["prompt_token_ids"]+r["generated_token_ids"]
        if not r["generated_token_ids"]:continue
        positions=locate_positions(tok,r)
        inputs=torch.tensor([ids],device=model.device);mask=torch.ones_like(inputs)
        with torch.inference_mode():out=model(input_ids=inputs,attention_mask=mask,output_hidden_states=True,use_cache=False)
        arrays={pos:np.stack([out.hidden_states[layer][0,index].float().cpu().numpy() for layer in layers]).astype(np.float32) for pos,index in positions.items()}
        # Token likelihood is a separately labelled baseline, not P(correct).
        begin=len(r["prompt_token_ids"])-1
        logits=out.logits[0,begin:-1].float()
        targets=torch.tensor(r["generated_token_ids"],device=model.device)
        logp=torch.log_softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1)
        arrays.update(logit=np.array(float(logp.exp().mean())),key=np.array(r["key"]),positions_json=np.array(json.dumps(positions)))
        tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)
        from src.protocol import append_jsonl
        append_jsonl(root/"performance"/f"{safe_name(name)}_features.jsonl",dict(stage="features",seconds=time.perf_counter()-t0,n_responses=1,generated_tokens=r["generated_tokens"]))
        del out,inputs,mask,logits,logp
        if i%25==0:print(f"Probe features {name}: {i+1}/{len(records)}",flush=True)

def probe_model(seed):
    return make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight="balanced",max_iter=2000,random_state=seed))

def out_of_fold(X,y,c):
    counts=np.bincount(y,minlength=2)
    if counts.min()<max(c["probe_min_class"],c["probe_folds"]):return None,None
    folds=StratifiedKFold(n_splits=c["probe_folds"],shuffle=True,random_state=c["seed"])
    scores=np.full(len(y),np.nan);fold_ids=np.full(len(y),-1)
    for fold,(train,test) in enumerate(folds.split(X,y)):
        clf=probe_model(c["seed"]);clf.fit(X[train],y[train]);scores[test]=clf.predict_proba(X[test])[:,1];fold_ids[test]=fold
    return scores,fold_ids

def analyze_probes(c,root):
    """Each model/mode/position has ONE row per original question, no paraphrase leakage."""
    from src.runner import safe_name
    rows=[];predictions=[];transfer=[]
    for name in c["probe_models"]:
        folder=root/"activations"/safe_name(name)
        if not (folder/"metadata.json").exists():continue
        meta=json.loads((folder/"metadata.json").read_text());records=read_jsonl(root/"raw"/safe_name(name)/"probe.jsonl")
        bykey={r["key"]:r for r in records};features=[]
        for p in folder.glob("*.npz"):
            with np.load(p,allow_pickle=False) as data:
                record=bykey[str(data["key"])];features.append((record,{k:data[k].copy() for k in data.files}))
        for mode in c["probe_modes"]:
            for position in meta["positions"]:
                pairs=[(r,f) for r,f in features if r["mode"]==mode and position in f]
                if not pairs:continue
                pairs.sort(key=lambda pair:pair[0]["id"])
                rr=[r for r,_ in pairs];y=np.array([r["correct"] for r in rr],int)
                if len({r["id"] for r in rr})!=len(rr):raise RuntimeError("Duplicate question in probe folds")
                strata=np.array([r["dataset"] for r in rr]);verb=np.array([r["confidence"] if r["confidence"] is not None else np.nan for r in rr])
                for li,layer in enumerate(meta["layers"]):
                    X=np.stack([f[position][li] for _,f in pairs]);scores,folds=out_of_fold(X,y,c)
                    row=dict(model=name,mode=mode,position=position,layer=layer,primary=position=="pre_answer" and layer==meta["primary_layer"],n=len(y),n_positive=int(y.sum()),n_negative=int((1-y).sum()),status="insufficient_classes")
                    if scores is None:rows.append(row);continue
                    row.update(status="computed",probe_auroc=float(roc_auc_score(y,scores)))
                    logit=np.array([float(f["logit"]) for _,f in pairs])
                    row["token_likelihood_auroc"]=compute_auroc(logit,y) if np.isfinite(logit).all() else np.nan
                    valid=np.isfinite(verb);row["n_matched"]=int(valid.sum());row["n_matched_positive"]=int(y[valid].sum())
                    if valid.sum() and min(y[valid].sum(),(1-y[valid]).sum())>=c["probe_min_class"]:
                        a=np.column_stack([verb[valid],y[valid]]);b=np.column_stack([scores[valid],y[valid]])
                        metric=lambda z:compute_auroc(z[:,0],z[:,1])
                        gap=paired_metric(a,b,metric,c["n_bootstrap"],c["seed"],strata[valid])
                        row.update({"gap_"+k:v for k,v in gap.items()})
                        row.update(verbalized_auroc_matched=metric(a),probe_auroc_matched=metric(b))
                        row["token_likelihood_auroc_matched"]=compute_auroc(logit[valid],y[valid])
                    # Conditional score bootstrap: same OOF scores, no refitting in each draw.
                    rng=np.random.default_rng(c["seed"]);aucs=[]
                    for _ in range(c["n_bootstrap"]):
                        idx=np.concatenate([rng.choice(np.flatnonzero(strata==d),sum(strata==d),True) for d in np.unique(strata)])
                        if len(np.unique(y[idx]))==2:aucs.append(roc_auc_score(y[idx],scores[idx]))
                    low,high,nvalid=interval(aucs);row.update(probe_ci_low=low,probe_ci_high=high,n_boot_valid=nvalid)
                    rows.append(row)
                    for i,r in enumerate(rr):predictions.append(dict(model=name,mode=mode,position=position,layer=layer,id=r["id"],dataset=r["dataset"],y=int(y[i]),confidence=verb[i],oof_score=scores[i],fold=int(folds[i])))
                    if position=="pre_answer" and layer==meta["primary_layer"]:
                        null_y=np.random.default_rng(c["seed"]).permutation(y);null_scores,_=out_of_fold(X,null_y,c)
                        row["label_shuffle_auroc"]=float(roc_auc_score(null_y,null_scores)) if null_scores is not None else np.nan
                        for train_ds in np.unique(strata):
                            for test_ds in np.unique(strata):
                                if train_ds==test_ds:continue
                                tr=strata==train_ds;te=strata==test_ds
                                if min(y[tr].sum(),(1-y[tr]).sum(),y[te].sum(),(1-y[te]).sum())<c["probe_min_class"]:continue
                                clf=probe_model(c["seed"]);clf.fit(X[tr],y[tr]);test_scores=clf.predict_proba(X[te])[:,1]
                                rng=np.random.default_rng(c["seed"]);vals=[];yt=y[te]
                                for _ in range(c["n_bootstrap"]):
                                    idx=rng.integers(0,len(yt),len(yt))
                                    if len(np.unique(yt[idx]))==2:vals.append(roc_auc_score(yt[idx],test_scores[idx]))
                                lo,hi,nv=interval(vals)
                                transfer.append(dict(model=name,mode=mode,position=position,layer=layer,train_dataset=train_ds,test_dataset=test_ds,n_train=int(tr.sum()),n_test=int(te.sum()),auroc=float(roc_auc_score(yt,test_scores)),ci_low=lo,ci_high=hi,n_boot_valid=nv))
    summary=root/"summary";summary.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(summary/"probes.csv",index=False)
    pd.DataFrame(predictions).to_csv(summary/"probe_oof_predictions.csv",index=False)
    pd.DataFrame(transfer).to_csv(summary/"probe_transfer.csv",index=False)
    return rows

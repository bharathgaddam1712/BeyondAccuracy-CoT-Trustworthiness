"""One source for all summaries, figures and numerical integrity checks."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from src.protocol import check_manifest,read_jsonl,atomic_json,validate_paraphrases
from src.metrics import calibration_metrics,compute_ece,compute_brier,compute_auroc,compute_aurc,compute_consistency,murphy_decomposition
from src.statistics import paired_metric,paired_randomization,holm,interval
from src.inference import check_correctness

def load_records(root):
    records=[]
    for path in sorted((root/"raw").glob("*/*.jsonl")):records.extend(read_jsonl(path))
    if not records:raise RuntimeError("No generated responses found")
    if len({(r["model"],r["precision"],r["key"]) for r in records})!=len(records):raise RuntimeError("Duplicate raw records")
    return pd.DataFrame(records)

def paired_rows(g,left,right):
    a=g[g["mode"]==left].set_index("id");b=g[g["mode"]==right].set_index("id")
    if a.index.has_duplicates or b.index.has_duplicates:raise RuntimeError("Duplicate IDs in a paired comparison")
    ids=a.index.intersection(b.index).sort_values();return a.loc[ids],b.loc[ids]

def compare(g,left,right,c,confidence=True):
    a,b=paired_rows(g,left,right);rows=[]
    base=dict(model=g.model.iloc[0],dataset=g.dataset.iloc[0],axis=g.axis.iloc[0],left=left,right=right,n_intersection=len(a))
    for metric,col in (("accuracy","correct"),("answer_compliance","answer_parsed"),("confidence_compliance","confidence_parsed")):
        if metric=="confidence_compliance" and not confidence:continue
        aa=a[col].astype(float).to_numpy();bb=b[col].astype(float).to_numpy()
        result=paired_metric(aa,bb,np.mean,c["n_bootstrap"],c["seed"])
        result["p_value"]=paired_randomization(aa,bb,np.mean,c["n_permutations"],c["seed"])
        rows.append(dict(base,metric=metric,**result))
    if confidence:
        valid=np.isfinite(a.confidence.astype(float)) & np.isfinite(b.confidence.astype(float))
        aa=np.column_stack([a.loc[valid,"confidence"].astype(float),a.loc[valid,"correct"].astype(int)])
        bb=np.column_stack([b.loc[valid,"confidence"].astype(float),b.loc[valid,"correct"].astype(int)])
        metrics={"ece":lambda z:compute_ece(z[:,0],z[:,1])["ece"],"brier":lambda z:compute_brier(z[:,0],z[:,1]),
                 "auroc":lambda z:compute_auroc(z[:,0],z[:,1]),"aurc":lambda z:compute_aurc(z[:,0],z[:,1]),
                 "reliability":lambda z:murphy_decomposition(z[:,0],z[:,1])["reliability"],
                 "resolution":lambda z:murphy_decomposition(z[:,0],z[:,1])["resolution"],
                 "uncertainty":lambda z:murphy_decomposition(z[:,0],z[:,1])["uncertainty"]}
        for name,fn in metrics.items():
            if len(aa)<2:result=dict(delta=np.nan,ci_low=np.nan,ci_high=np.nan,n=len(aa),n_boot_valid=0)
            else:result=paired_metric(aa,bb,fn,c["n_bootstrap"],c["seed"])
            if name in ("ece","brier") and len(aa)>=2:result["p_value"]=paired_randomization(aa,bb,fn,c["n_permutations"],c["seed"])
            result.update(left_value=fn(aa) if len(aa) else np.nan,right_value=fn(bb) if len(bb) else np.nan)
            rows.append(dict(base,metric=name,sample_mask="confidence_parsed_both; answer_parse_failures_scored_incorrect",**result))
    return rows

def summarize_groups(df):
    rows=[];bins=[]
    group_cols=["axis","model","size_b","precision","dataset","mode"]
    for key,g in df.groupby(group_cols):
        if key[0]=="self_consistency":continue
        row=dict(zip(group_cols,key));row.update(n_total=len(g),n_questions=g.id.nunique(),accuracy=float(g.correct.mean()),
            answer_compliance=float(g.answer_parsed.mean()),confidence_compliance=float(g.confidence_parsed.mean()),
            token_cap_fraction=float(g.hit_token_cap.mean()),mean_generated_tokens=float(g.generated_tokens.mean()),
            n_correct=int(g.correct.sum()),n_answer_parsed=int(g.answer_parsed.sum()))
        valid=g[np.isfinite(g.confidence.astype(float))]
        row["n_valid"]=len(valid)
        if len(valid):
            row.update(calibration_metrics(valid.confidence,valid.correct.astype(int)))
            for b in compute_ece(valid.confidence,valid.correct.astype(int))["bin_data"]:bins.append(dict(zip(group_cols,key),**b))
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(bins)

def consistency_analysis(df,c):
    qrows=[];comparisons=[]
    for (model,ds,mode,qid),g in df[df.axis=="consistency"].groupby(["model","dataset","mode","id"]):
        if set(g.variant)!=set(range(6)) or len(g)!=6:raise RuntimeError(f"Incomplete consistency variants: {model}/{ds}/{mode}/{qid}")
        preds=g.sort_values("variant").predicted.tolist();parsed=[p for p in preds if p is not None and not pd.isna(p)]
        qrows.append(dict(model=model,dataset=ds,mode=mode,id=qid,axis="consistency",agreement=compute_consistency(preds),
                         agreement_parsed_only=compute_consistency(parsed),n_parsed=len(parsed),n_variants=6,accuracy=float(g.correct.mean())))
    q=pd.DataFrame(qrows)
    if q.empty:return q,pd.DataFrame()
    for (model,ds),g in q.groupby(["model","dataset"]):
        a,b=paired_rows(g,"direct","cot")
        result=paired_metric(a.agreement.to_numpy(),b.agreement.to_numpy(),np.mean,c["n_bootstrap"],c["seed"])
        result["p_value"]=paired_randomization(a.agreement.to_numpy(),b.agreement.to_numpy(),np.mean,c["n_permutations"],c["seed"])
        comparisons.append(dict(model=model,dataset=ds,metric="pairwise_agreement",left="direct",right="cot",left_value=float(a.agreement.mean()),right_value=float(b.agreement.mean()),**result))
    return q,pd.DataFrame(comparisons)

def cru_analysis(df,c):
    rows=[]
    for (model,ds),g in df[df.axis=="ablation"].groupby(["model","dataset"]):
        modes=["trace_empty","trace_filler","trace_structured","trace_shuffled"]
        bymode={mode:g[g["mode"]==mode].set_index("id") for mode in modes}
        ids=set.intersection(*(set(v.index) for v in bymode.values()))
        ids=sorted(i for i in ids if all(pd.notna(v.loc[i,"confidence"]) for v in bymode.values()))
        row=dict(model=model,dataset=ds,n_matched=len(ids),baseline="same continuation template with empty supplied text")
        if not ids:rows.append(row);continue
        values={mode:compute_ece(v.loc[ids,"confidence"].astype(float),v.loc[ids,"correct"].astype(int))["ece"] for mode,v in bymode.items()}
        row.update({"accuracy_"+mode:float(v.loc[ids,"correct"].mean()) for mode,v in bymode.items()})
        row.update({"mean_input_tokens_"+mode:float(v.loc[ids,"input_tokens"].mean()) for mode,v in bymode.items() if "input_tokens" in v})
        row.update(values);d,f,cot=values["trace_empty"],values["trace_filler"],values["trace_structured"]
        gain=d-cot;row["structured_gain"]=gain;row["filler_gain"]=d-f;row["cru"]=np.nan
        if gain<=0:row["regime"]="structured_not_better_than_empty; CRU_not_interpreted"
        elif gain<.01:row["regime"]="small_denominator; CRU_not_interpreted"
        else:
            row["cru"]=(f-cot)/gain
            row["regime"]="filler_matches_or_beats_structured" if f<=cot else "structured_better_than_filler; descriptive_only"
        a=bymode["trace_empty"].loc[ids];b=bymode["trace_structured"].loc[ids]
        aa=np.column_stack([a.confidence.astype(float),a.correct.astype(int)]);bb=np.column_stack([b.confidence.astype(float),b.correct.astype(int)])
        ci=paired_metric(aa,bb,lambda z:compute_ece(z[:,0],z[:,1])["ece"],c["n_bootstrap"],c["seed"])
        row["structured_gain_ci_low"]=-ci["ci_high"];row["structured_gain_ci_high"]=-ci["ci_low"]
        row["gain_supported_by_ci"]=bool(row["structured_gain_ci_low"]>0)
        rows.append(row)
    return pd.DataFrame(rows)

def self_consistency_analysis(df,c):
    rows=[]
    for (model,ds,qid),g in df[df.axis=="self_consistency"].groupby(["model","dataset","id"]):
        if len(g)!=c["self_consistency_k"]:raise RuntimeError("Incomplete self-consistency samples")
        counts=Counter(p for p in g.predicted if p is not None and not pd.isna(p))
        # Invalid generations occupy denominator; no valid answer means abstention.
        answer=sorted(counts,key=lambda p:(-counts[p],p))[0] if counts else None
        rows.append(dict(model=model,dataset=ds,id=qid,mode="vote",axis="self_consistency_vote",size_b=g.size_b.iloc[0],precision="main",
                         predicted=answer,confidence=counts[answer]/len(g) if answer is not None else None,
                         correct=check_correctness(answer,g.gold.iloc[0],ds),n_samples=len(g)))
    q=pd.DataFrame(rows);summary=[]
    if q.empty:return q,pd.DataFrame(),pd.DataFrame()
    comparisons=[]
    for (model,ds),g in q.groupby(["model","dataset"]):
        valid=g[g.confidence.notna()]
        row=dict(model=model,dataset=ds,n_total=len(g),accuracy=float(g.correct.mean()),abstention_rate=float(g.confidence.isna().mean()))
        if len(valid):row.update(calibration_metrics(valid.confidence,valid.correct.astype(int)))
        summary.append(row)
        verbal=df[(df.model==model)&(df.dataset==ds)&(df.axis=="calibration")&(df["mode"]=="cot")].set_index("id")
        vote=g.set_index("id");ids=vote.index.intersection(verbal.index);ids=[i for i in ids if pd.notna(vote.loc[i,"confidence"]) and pd.notna(verbal.loc[i,"confidence"])]
        a=np.column_stack([verbal.loc[ids,"confidence"].astype(float),verbal.loc[ids,"correct"].astype(int)])
        b=np.column_stack([vote.loc[ids,"confidence"].astype(float),vote.loc[ids,"correct"].astype(int)])
        for metric,fn in (("ece",lambda z:compute_ece(z[:,0],z[:,1])["ece"]),("brier",lambda z:compute_brier(z[:,0],z[:,1]))):
            comparisons.append(dict(model=model,dataset=ds,metric=metric,left="greedy_cot_verbalized",right="sampled_vote",**paired_metric(a,b,fn,c["n_bootstrap"],c["seed"])))
    return q,pd.DataFrame(summary),pd.DataFrame(comparisons)

def aggregate_calibration(df,c):
    """Shared question draws across models preserve repeated-question dependence."""
    rows=[]
    base=df[(df.axis=="calibration") & df["mode"].isin(["direct","cot"])]
    for ds in sorted(base.dataset.unique()):
        for size_class in ("sub3","ge3"):
            g=base[(base.dataset==ds) & ((base.size_b<3) if size_class=="sub3" else (base.size_b>=3))]
            pairs=[]
            for model,v in g.groupby("model"):
                a,b=paired_rows(v,"direct","cot");mask=np.isfinite(a.confidence.astype(float))&np.isfinite(b.confidence.astype(float))
                pairs.append((model,a[mask],b[mask]))
            if not pairs:continue
            universe=sorted(set.union(*(set(a.index) for _,a,b in pairs)))
            def delta_for(ids):
                values=[]
                for model,a,b in pairs:
                    selected=[i for i in ids if i in a.index]
                    if not selected:return np.nan
                    values.append(compute_ece(b.loc[selected,"confidence"].astype(float),b.loc[selected,"correct"].astype(int))["ece"]-compute_ece(a.loc[selected,"confidence"].astype(float),a.loc[selected,"correct"].astype(int))["ece"])
                return float(np.mean(values))
            observed=[compute_ece(b.confidence.astype(float),b.correct.astype(int))["ece"]-compute_ece(a.confidence.astype(float),a.correct.astype(int))["ece"] for _,a,b in pairs if len(a)]
            if len(observed)!=len(pairs):continue
            rng=np.random.default_rng(c["seed"]);boot=[delta_for(rng.choice(universe,len(universe),True)) for _ in range(c["n_bootstrap"])]
            lo,hi,nv=interval(boot)
            rows.append(dict(dataset=ds,size_class=size_class,n_models=len(pairs),n_questions=len(universe),mean_delta=float(np.mean(observed)),sd_across_models=float(np.std(observed,ddof=1)) if len(observed)>1 else np.nan,min_delta=min(observed),max_delta=max(observed),ci_low=lo,ci_high=hi,n_boot_valid=nv))
    return pd.DataFrame(rows)

def validate_outputs(c,root,df,require_human=False):
    samples=json.loads((root/"datasets.json").read_text());expected=len(samples)*2
    errors=[]
    expected_ids={s["id"] for s in samples}
    for model in c["models"]:
        for axis in ("accuracy","calibration"):
            g=df[(df.model==model)&(df.axis==axis)]
            if len(g)!=expected:errors.append(f"{model}/{axis}: {len(g)} of {expected} responses")
            for mode in ("direct","cot"):
                if set(g[g["mode"]==mode].id)!=expected_ids:errors.append(f"{model}/{axis}/{mode}: question IDs do not match frozen subset")
        g=df[(df.model==model)&(df.axis=="consistency")]
        n=sum(s["dataset"] in c["consistency_datasets"] for s in samples)*12
        if len(g)!=n:errors.append(f"{model}/consistency: {len(g)} of {n} responses")
    from src.runner import selected,safe_name,score_record
    for model in c["ablation_models"]:
        if model not in c["models"]:continue
        eligibility=root/"raw"/safe_name(model)/"ablation_eligibility.json"
        if not eligibility.exists():errors.append(f"Missing trace eligibility report for {model}")
        else:
            eligible=json.loads(eligibility.read_text());n_eligible=sum(r["eligible"] for r in eligible)
            actual=len(df[(df.model==model)&(df.axis=="ablation")])
            if actual!=n_eligible*4:errors.append(f"Incomplete trace interventions for {model}: {actual}/{n_eligible*4}")
            if n_eligible==0 and not c.get("smoke",False):errors.append(f"No eligible trace interventions for {model}; investigate before reporting this experiment")
        wanted=len(selected(samples,c["ablation_n_per_dataset"]))*c["self_consistency_k"]
        if len(df[(df.model==model)&(df.axis=="self_consistency")])!=wanted:errors.append(f"Incomplete sampled self-consistency for {model}")
    from src.protocol import digest
    for model in c["probe_models"]:
        if model not in c["models"]:continue
        wanted=len(selected(samples,c["probe_n_per_dataset"]))*len(c["probe_modes"])
        records=df[(df.model==model)&(df.axis=="probe")]
        if len(records)!=wanted:errors.append(f"Incomplete probe generations for {model}: {len(records)}/{wanted}")
        missing=sum(not (root/"activations"/safe_name(model)/f"{digest(r['key'])}.npz").exists() for r in records.to_dict("records") if r["generated_tokens"]>0)
        if missing:errors.append(f"Missing {missing} activation checkpoints for {model}")
    if c["quantization_model"] in c["models"]:
        wanted=len(selected(samples,c["quantization_n_per_dataset"]))*2
        for precision in ("nf4","float16"):
            actual=len(df[(df.model==c["quantization_model"])&(df.axis=="quantization")&(df.precision==precision)])
            if actual!=wanted:errors.append(f"Incomplete {precision} quantization comparison: {actual}/{wanted}")
    # Reparse every response and detect stale or manually mismatched correctness/confidence.
    for r in df.to_dict("records"):
        scored=score_record(dict(dataset=r["dataset"],answer=r["gold"],choices=r.get("choices"),subtask=r.get("subtask")),r["raw_response"],r["mode"])
        mismatch=False
        for field in ("predicted","confidence","correct","answer_parsed","confidence_parsed"):
            stored=r[field];fresh=scored[field]
            stored=None if stored is None or (isinstance(stored,float) and np.isnan(stored)) else stored
            if stored!=fresh:mismatch=True
        if mismatch:
            errors.append(f"Stored parser fields differ from raw response: {r['model']}/{r['key']}")
            if len(errors)>50:break
    try:validate_paraphrases(c,root)
    except (RuntimeError,FileNotFoundError) as exc:errors.append(str(exc))
    human_path=root/"paraphrase_human_audit.csv";human={"status":"missing"}
    if human_path.exists():
        h=pd.read_csv(human_path,keep_default_na=False);answers=h.human_equivalent.astype(str).str.strip().str.lower()
        valid=answers.isin(["yes","no"])
        human=dict(status="complete" if valid.all() else "pending",n=len(h),n_reviewed=int(valid.sum()),n_rejected=int((answers=="no").sum()))
        if (answers=="no").any():errors.append("Human audit rejected paraphrases. Replace them and rerun consistency under a new run_name")
    if require_human and human["status"]!="complete":errors.append("Complete the human fidelity sample with yes/no before paper-ready validation")
    report=dict(status="passed" if not errors else "failed",paper_ready=not errors and human["status"]=="complete",errors=errors,human_paraphrase_audit=human,
                n_responses=len(df),token_cap_fraction=float(df.hit_token_cap.mean()),
                note="Passed checks establish data/metric integrity, not acceptance or causal interpretation")
    atomic_json(root/"validation.json",report)
    return report

def analyze(c,require_complete=False,require_human=False):
    root,_=check_manifest(c);df=load_records(root);out=root/"summary";out.mkdir(exist_ok=True)
    summary,bins=summarize_groups(df);summary.to_csv(out/"metrics_by_cell.csv",index=False);bins.to_csv(out/"calibration_bins.csv",index=False)
    # Full rows preserved in JSONL; CSV is the compact analysis view.
    cols=[x for x in df.columns if x not in ("messages","prompt_token_ids","generated_token_ids","choices")]
    df[cols].to_csv(out/"responses.csv",index=False)
    comparisons=[]
    for (axis,model,ds),g in df[df.axis.isin(["accuracy","calibration","ablation"])].groupby(["axis","model","dataset"]):
        if axis=="ablation":
            for mode in ("trace_empty","trace_filler","trace_shuffled"):comparisons+=compare(g,mode,"trace_structured",c)
        else:comparisons+=compare(g,"direct","cot",c,axis=="calibration")
    comp=pd.DataFrame(comparisons)
    if not comp.empty:
        comp["p_holm"]=np.nan
        if "p_value" in comp:
            for _,g in comp.groupby(["axis","metric","left","right"]):comp.loc[g.index,"p_holm"]=holm(g.p_value)
    comp.to_csv(out/"paired_comparisons.csv",index=False)
    q,cons=consistency_analysis(df,c)
    if not cons.empty:cons["p_holm"]=holm(cons.p_value)
    q.to_csv(out/"consistency_by_question.csv",index=False);cons.to_csv(out/"consistency_paired.csv",index=False)
    cru=cru_analysis(df,c)
    cru.to_csv(out/"trace_ablation_comparisons.csv",index=False)
    sq,ss,sc=self_consistency_analysis(df,c)
    sq.to_csv(out/"self_consistency_by_question.csv",index=False);ss.to_csv(out/"self_consistency_metrics.csv",index=False);sc.to_csv(out/"self_consistency_paired.csv",index=False)
    aggregate_calibration(df,c).to_csv(out/"calibration_by_size_and_task.csv",index=False)
    # Quantization compares exactly matching questions under the same prompts.
    quant=[]
    for (model,ds),g in df[(df.axis=="quantization") | ((df.axis=="calibration") & (df.model==c["quantization_model"]))].groupby(["model","dataset"]):
        for mode in ("direct","cot"):
            v=g[g["mode"]==mode].copy();v["mode"]=v.precision
            for precision in ("nf4","float16"):
                if precision in v["mode"].values:
                    fresh=compare(v,"main",precision,c)
                    quant.extend(dict(row,generation_mode=mode) for row in fresh)
    pd.DataFrame(quant).to_csv(out/"quantization_paired.csv",index=False)
    from src.probing import analyze_probes
    analyze_probes(c,root)
    figures(root,summary,cons,comp,cru)
    report=validate_outputs(c,root,df,require_human)
    print(json.dumps(report,indent=2),flush=True)
    if require_complete and report["errors"]:raise RuntimeError("Validation failed; see validation.json")
    return report

def figures(root,summary,cons,comparisons,cru):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=root/"figures";folder.mkdir(exist_ok=True)
    for ds,g in summary[summary.axis=="calibration"].groupby("dataset"):
        fig,ax=plt.subplots(figsize=(9,4))
        for mode in ("direct","cot"):
            v=g[g["mode"]==mode].sort_values(["size_b","model"])
            ax.plot(v.model,v.ece,"o-",label=mode)
        g.to_csv(folder/f"calibration_{ds}.csv",index=False)
        ax.set(ylabel="ECE (10 equal-width bins)",title=ds);ax.tick_params(axis="x",rotation=30);ax.legend();fig.tight_layout();fig.savefig(folder/f"calibration_{ds}.png",dpi=180);plt.close(fig)
    if not cons.empty:
        for ds,g in cons.groupby("dataset"):
            finite=g[np.isfinite(g.delta)].copy();fig,ax=plt.subplots(figsize=(9,4))
            ax.scatter(finite.model,finite.delta)
            for _,r in finite.iterrows():ax.plot([r.model,r.model],[r.ci_low,r.ci_high],color="C0")
            ax.axhline(0,color="grey",ls="--");ax.set(ylabel="CoT − Direct agreement",title=ds);ax.tick_params(axis="x",rotation=30);fig.tight_layout();fig.savefig(folder/f"consistency_{ds}.png",dpi=180);plt.close(fig)
            g.to_csv(folder/f"consistency_{ds}.csv",index=False)
    # Intervention table and plot use the identical summary cells, never an unmatched baseline.
    for _,r in cru.iterrows():
        if not r.n_matched:continue
        model,ds=r.model,r.dataset
        modes=["trace_empty","trace_filler","trace_structured","trace_shuffled"]
        data=pd.DataFrame([dict(mode=mode,ece=r[mode],accuracy=r["accuracy_"+mode],n_matched=r.n_matched) for mode in modes])
        data.to_csv(folder/f"trace_{model}_{ds}.csv",index=False)
        fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(data["mode"],data.accuracy);axes[1].bar(data["mode"],data.ece)
        axes[0].set_ylabel("Accuracy");axes[1].set_ylabel("ECE");fig.suptitle(f"{model} / {ds}: matched continuation controls, N={int(r.n_matched)}")
        for ax in axes:ax.tick_params(axis="x",rotation=35)
        fig.tight_layout();fig.savefig(folder/f"trace_{model}_{ds}.png",dpi=180);plt.close(fig)

"""Canonical metrics. Undefined quantities are NaN, never a fabricated score."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def validate_pairs(confidences, correctness):
    p,y=np.asarray(confidences,float),np.asarray(correctness,float)
    if p.ndim!=1 or y.shape!=p.shape or not len(p):
        raise ValueError("Expected equally sized nonempty one-dimensional arrays")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Apply an explicit common finite mask before computing metrics")
    if np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():
        raise ValueError("Confidence must be in [0,1], correctness binary")
    return p,y

def bin_assignments(p,n_bins=10,strategy="width"):
    if n_bins<1 or strategy not in ("width","quantile"):raise ValueError("Invalid bins")
    edges=np.linspace(0,1,n_bins+1) if strategy=="width" else np.unique(np.r_[0.,np.quantile(p,np.linspace(0,1,n_bins+1)),1.])
    return np.searchsorted(edges[1:-1],p,side="right"),edges

def compute_ece(confidences,correctness,n_bins=10,strategy="width"):
    p,y=validate_pairs(confidences,correctness);ids,edges=bin_assignments(p,n_bins,strategy);rows=[]
    for i in range(len(edges)-1):
        mask=ids==i;n=int(mask.sum())
        if n:rows.append(dict(bin_lo=float(edges[i]),bin_hi=float(edges[i+1]),bin_mid=float((edges[i]+edges[i+1])/2),bin_conf=float(p[mask].mean()),bin_acc=float(y[mask].mean()),bin_count=n))
    assert sum(b["bin_count"] for b in rows)==len(p)
    value=sum(b["bin_count"]*abs(b["bin_conf"]-b["bin_acc"]) for b in rows)/len(p)
    assert value+1e-12>=abs(p.mean()-y.mean())
    return dict(ece=float(value),bin_data=rows)

def compute_brier(confidences,correctness):
    p,y=validate_pairs(confidences,correctness);return float(np.mean((p-y)**2))

def compute_auroc(confidences,correctness):
    p,y=validate_pairs(confidences,correctness)
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan")

def compute_aurc(confidences,correctness):
    """Expected discrete AURC under uniformly random ordering within score ties."""
    p,y=validate_pairs(confidences,correctness);n_prev,errors_prev,risks=0,0.,[]
    for score in np.unique(p)[::-1]:
        outcomes=y[p==score];rate=float(1-outcomes.mean())
        risks.extend((errors_prev+j*rate)/(n_prev+j) for j in range(1,len(outcomes)+1))
        n_prev+=len(outcomes);errors_prev+=len(outcomes)*rate
    return float(np.mean(risks))

def murphy_decomposition(confidences,correctness,n_bins=None):
    """Exact for distinct forecasts. A binned approximation reports its residual."""
    p,y=validate_pairs(confidences,correctness)
    ids=np.unique(p,return_inverse=True)[1] if n_bins is None else bin_assignments(p,n_bins)[0]
    base,rel,res=y.mean(),0.,0.
    for i in np.unique(ids):
        mask=ids==i;weight=mask.mean()
        rel+=weight*(p[mask].mean()-y[mask].mean())**2
        res+=weight*(y[mask].mean()-base)**2
    unc=float(base*(1-base));bs=compute_brier(p,y);residual=float(bs-(rel-res+unc))
    if n_bins is None:assert abs(residual)<1e-10
    return dict(brier=bs,reliability=float(rel),resolution=float(res),uncertainty=unc,residual=residual,base_rate=float(base))

def calibration_metrics(p,y):
    p,y=validate_pairs(p,y)
    out=dict(n_valid=len(p),accuracy_valid=float(y.mean()),mean_confidence=float(p.mean()),brier=compute_brier(p,y),auroc=compute_auroc(p,y),aurc=compute_aurc(p,y))
    for k in (5,10,15,20):out[f"ece_{k}"]=compute_ece(p,y,k)["ece"]
    out["ece"]=out["ece_10"];out["ece_quantile_10"]=compute_ece(p,y,10,"quantile")["ece"]
    out.update(murphy_decomposition(p,y));out["binned_brier_residual"]=murphy_decomposition(p,y,10)["residual"]
    assert out["brier"]<=1-y.mean()+1-p.mean()+1e-12
    return out

def compute_accuracy_by_dataset(df):
    return df.groupby(["model","mode","dataset"],as_index=False).agg(accuracy=("correct","mean"),n=("correct","size"))

def compute_calibration_by_model(df,n_bins=10):
    rows=[]
    for (m,s,d),g in df.groupby(["model","mode","dataset"]):
        valid=g[np.isfinite(g.confidence.astype(float))]
        row=dict(model=m,mode=s,dataset=d,n_total=len(g),confidence_compliance=len(valid)/len(g))
        if len(valid):row.update(calibration_metrics(valid.confidence,valid.correct.astype(int)))
        rows.append(row)
    return pd.DataFrame(rows)

def compute_consistency(answers):
    """All-pairs agreement. Missing answers never agree; singleton is undefined.
    A(q)=2/[K(K-1)] sum_{i<j} 1[a_i=a_j and both parsed].
    Inputs must already be normalized by the dataset-specific parser.
    """
    if len(answers)<2:return float("nan")
    valid=lambda a:a is not None and not pd.isna(a) and str(a)!=""
    hits=sum(valid(a) and valid(b) and a==b for i,a in enumerate(answers) for b in answers[i+1:])
    return float(hits/(len(answers)*(len(answers)-1)/2))

def compute_consistency_by_model(df):
    rows=[]
    for (m,s,d),g in df.groupby(["model","mode","dataset"]):
        values=[compute_consistency(v.predicted.tolist()) for _,v in g.groupby("question_id")]
        finite=np.array(values)[np.isfinite(values)]
        rows.append(dict(model=m,mode=s,dataset=d,consistency=float(finite.mean()) if len(finite) else np.nan,n_questions=len(values),n_eligible=len(finite)))
    return pd.DataFrame(rows)

def compute_consistency_delta(df):
    pivot=df.pivot(index=["model","dataset"],columns="mode",values="consistency")
    return pivot.assign(delta=pivot.cot-pivot.direct)[["delta"]].reset_index()

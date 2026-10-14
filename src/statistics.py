"""Paired uncertainty on questions; nonlinear metrics recomputed per resample."""
import numpy as np
def interval(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    return (float(np.quantile(v,.025)),float(np.quantile(v,.975)),len(v)) if len(v) else (float("nan"),float("nan"),0)
def paired_metric(a,b,metric,n_boot=2000,seed=42,strata=None):
    """Matching rows are questions. Difference is b-a; resampling is paired."""
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or len(a)<2:return dict(delta=float("nan"),ci_low=float("nan"),ci_high=float("nan"),n=len(a),n_boot_valid=0)
    groups=[np.arange(len(a))] if strata is None else [np.flatnonzero(np.asarray(strata)==s) for s in np.unique(strata)]
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n_boot):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);values.append(metric(b[idx])-metric(a[idx]))
    low,high,nvalid=interval(values)
    return dict(delta=float(metric(b)-metric(a)),ci_low=low,ci_high=high,n=len(a),n_boot_valid=nvalid)
def paired_randomization(a,b,metric,n_perm=2000,seed=42):
    """Within-question label swaps. Null hypothesis is condition exchangeability."""
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<2:return float("nan")
    observed=abs(metric(b)-metric(a));rng=np.random.default_rng(seed);extreme=0
    for _ in range(n_perm):
        swap=rng.integers(0,2,len(a)).astype(bool);mask=swap.reshape((-1,)+(1,)*(a.ndim-1))
        left,right=np.where(mask,b,a),np.where(mask,a,b)
        extreme+=int(abs(metric(right)-metric(left))>=observed-1e-12)
    return (extreme+1)/(n_perm+1)
def holm(pvalues):
    p=np.asarray(pvalues,float);out=np.full_like(p,np.nan);idx=np.flatnonzero(np.isfinite(p));idx=idx[np.argsort(p[idx])];adjusted=0.
    for rank,i in enumerate(idx):
        adjusted=max(adjusted,(len(idx)-rank)*p[i]);out[i]=min(1.,adjusted)
    return out

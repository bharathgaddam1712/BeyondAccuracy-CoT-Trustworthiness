"""
Metrics for Paper A — Beyond Accuracy.

Axis 1 (accuracy):     compute_accuracy_by_dataset
Axis 2 (calibration):  compute_ece (dict: ece + bin_data), compute_brier,
                       compute_auroc, compute_aurc, compute_calibration_by_model
Axis 3 (consistency):  compute_consistency, compute_consistency_by_model,
                       compute_consistency_delta
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Axis 1 — Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: model, mode, dataset, correct (bool)
    Returns one row per (model, mode, dataset) with accuracy, and size_b if present.
    """
    agg = df.groupby(["model", "mode", "dataset"])["correct"].mean().reset_index()
    agg = agg.rename(columns={"correct": "accuracy"})
    if "size_b" in df.columns:
        size_lookup = df.groupby("model")["size_b"].first()
        agg["size_b"] = agg["model"].map(size_lookup)
    return agg


# ---------------------------------------------------------------------------
# Axis 2 — Calibration
# ---------------------------------------------------------------------------

def compute_ece(confidences, correctness, n_bins: int = 10) -> dict:
    """
    Standard ECE with equal-width bins.
    Returns {"ece": float, "bin_data": [{"bin_lo","bin_hi","bin_mid","bin_conf","bin_acc","bin_count"}]}
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(conf)
    ece = 0.0
    bin_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(corr[mask].mean())
        mean_conf = float(conf[mask].mean())
        ece += (cnt / n) * abs(acc - mean_conf)
        bin_data.append({
            "bin_lo":   float(lo),
            "bin_hi":   float(hi),
            "bin_mid":  float((lo + hi) / 2),
            "bin_conf": mean_conf,
            "bin_acc":  acc,
            "bin_count": cnt,
        })
    return {"ece": float(ece), "bin_data": bin_data}


def compute_brier(confidences, correctness) -> float:
    """Mean squared error between confidence and binary outcome."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    return float(np.mean((conf - corr) ** 2))


def compute_auroc(confidences, correctness) -> float:
    """AUC-ROC: can confidence discriminate correct from wrong answers? 0.5 if one class only."""
    corr = np.asarray(correctness, dtype=float)
    if len(np.unique(corr)) < 2:
        return 0.5
    return float(roc_auc_score(corr, np.asarray(confidences, dtype=float)))


def compute_aurc(confidences, correctness) -> float:
    """Area under the risk-coverage curve (selective prediction). Lower = better."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=float)
    order = np.argsort(conf)[::-1]
    sorted_correct = corr[order]
    n = len(sorted_correct)
    risks = [1.0 - sorted_correct[:k].mean() for k in range(1, n + 1)]
    return float(np.mean(risks))


def compute_calibration_by_model(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    df must have columns: model, mode, dataset, confidence, correct (rows with
    confidence=NaN are dropped per-group).
    Returns one row per (model, mode, dataset) with ece, brier, auroc, aurc,
    mean_confidence, n_valid, and size_b if present.
    """
    rows = []
    for (model, mode, dataset), grp in df.groupby(["model", "mode", "dataset"]):
        valid = grp.dropna(subset=["confidence"])
        if len(valid) == 0:
            continue
        conf = valid["confidence"].astype(float).values
        corr = valid["correct"].astype(int).values
        ece_res = compute_ece(conf, corr, n_bins=n_bins)
        row = {
            "model":           model,
            "mode":            mode,
            "dataset":         dataset,
            "ece":             ece_res["ece"],
            "brier":           compute_brier(conf, corr),
            "auroc":           compute_auroc(conf, corr),
            "aurc":            compute_aurc(conf, corr),
            "mean_confidence": float(conf.mean()),
            "n_valid":         len(valid),
        }
        if "size_b" in grp.columns:
            row["size_b"] = grp["size_b"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Axis 3 — Consistency
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    import re, string
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def compute_consistency(answers: list) -> float:
    """Fraction of answer variants matching the plurality answer."""
    if not answers:
        return 0.0
    from collections import Counter
    counts = Counter(normalize_answer(a) for a in answers)
    plurality_count = counts.most_common(1)[0][1]
    return plurality_count / len(answers)


def compute_consistency_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: model, mode, dataset, question_id, predicted
    Returns one row per (model, mode, dataset) with mean consistency across questions.
    """
    rows = []
    for (model, mode, dataset), grp in df.groupby(["model", "mode", "dataset"]):
        per_question = grp.groupby("question_id")["predicted"].apply(list)
        if len(per_question) == 0:
            continue
        mean_consistency = float(np.mean([compute_consistency(answers) for answers in per_question]))
        row = {"model": model, "mode": mode, "dataset": dataset, "consistency": mean_consistency}
        if "size_b" in grp.columns:
            row["size_b"] = grp["size_b"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_consistency_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    summary_df must have columns: model, mode, dataset, consistency (output of
    compute_consistency_by_model). Returns one row per (model, dataset) with
    delta = consistency[cot] - consistency[direct].
    """
    rows = []
    for (model, dataset), grp in summary_df.groupby(["model", "dataset"]):
        cot_val    = grp[grp["mode"] == "cot"]["consistency"]
        direct_val = grp[grp["mode"] == "direct"]["consistency"]
        if len(cot_val) == 0 or len(direct_val) == 0:
            continue
        row = {
            "model":   model,
            "dataset": dataset,
            "delta":   float(cot_val.iloc[0] - direct_val.iloc[0]),
        }
        if "size_b" in grp.columns:
            row["size_b"] = grp["size_b"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)

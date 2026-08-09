"""
Metrics for Paper A — Beyond Accuracy.

Change 4: Brier score + AUROC/AURC alongside ECE.
Change 6: compute_reliability_frontier() for the Pareto plot.
Change 2: compute_confidence_instability_correlation().
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# 1. Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """
    Standard ECE with equal-width bins.
    confidences: [0,1] float array
    correctness: binary int array (1=correct, 0=wrong)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_reliability_diagram_data(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Returns per-bin (mean_conf, mean_acc, count) for reliability diagram plotting."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lo":    lo,
            "bin_hi":    hi,
            "mean_conf": float(confidences[mask].mean()),
            "mean_acc":  float(correctness[mask].mean()),
            "count":     int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Brier Score  (Change 4)
# ---------------------------------------------------------------------------

def compute_brier(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Mean squared error between confidence and binary outcome."""
    return float(np.mean((confidences - correctness.astype(float)) ** 2))


# ---------------------------------------------------------------------------
# 3. AUROC — confidence as binary classifier of correctness  (Change 4)
# ---------------------------------------------------------------------------

def compute_auroc(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """
    AUC-ROC: can confidence discriminate correct from wrong answers?
    Returns 0.5 if only one class present.
    """
    if len(np.unique(correctness)) < 2:
        return 0.5
    return float(roc_auc_score(correctness, confidences))


# ---------------------------------------------------------------------------
# 4. AURC — Area Under Risk-Coverage Curve  (Change 4)
# ---------------------------------------------------------------------------

def compute_aurc(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """
    AURC: selective prediction quality.
    Sort by descending confidence; compute risk (error rate) at each coverage level.
    Lower AURC = better selective prediction.
    """
    order  = np.argsort(confidences)[::-1]
    sorted_correct = correctness[order].astype(float)
    n = len(sorted_correct)
    risks = []
    for k in range(1, n + 1):
        risk = 1.0 - sorted_correct[:k].mean()
        risks.append(risk)
    return float(np.mean(risks))


# ---------------------------------------------------------------------------
# 5. Paraphrase consistency  (Change 2 extension)
# ---------------------------------------------------------------------------

def compute_consistency(answers: list[str]) -> float:
    """
    Fraction of answer variants that match the plurality answer.
    answers: list of strings (one per paraphrase variant)
    """
    if not answers:
        return 0.0
    from collections import Counter
    counts = Counter(normalize_answer(a) for a in answers)
    plurality_count = counts.most_common(1)[0][1]
    return plurality_count / len(answers)


def compute_consistency_delta(cot_answers: list[str], direct_answers: list[str]) -> float:
    """CoT consistency minus direct consistency. Positive = CoT is more stable."""
    return compute_consistency(cot_answers) - compute_consistency(direct_answers)


def normalize_answer(s: str) -> str:
    import re, string
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


# ---------------------------------------------------------------------------
# 6. Confidence → instability correlation  (Change 2)
# ---------------------------------------------------------------------------

def compute_confidence_instability_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, mode, dataset): compute Spearman correlation between
    verbalized_confidence and paraphrase_inconsistency (1 - consistency).

    df must have columns: model, mode, dataset, confidence, consistency
    Returns per-(model, mode, dataset) row with spearman_r and p_value.
    """
    from scipy.stats import spearmanr
    rows = []
    for (model, mode, dataset), grp in df.groupby(["model", "mode", "dataset"]):
        if len(grp) < 5:
            continue
        conf  = grp["confidence"].fillna(0.5).values
        insta = 1.0 - grp["consistency"].values
        r, p  = spearmanr(conf, insta)
        rows.append({
            "model": model, "mode": mode, "dataset": dataset,
            "spearman_r": float(r), "p_value": float(p),
            "n": len(grp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Full reliability summary per (model, mode, dataset)  (Change 3 + 4)
# ---------------------------------------------------------------------------

def compute_reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns:
      model, mode, dataset, size_b, confidence, correct, consistency

    Returns one row per (model, mode, dataset) with:
      accuracy, ece, brier, auroc, aurc, mean_consistency, cot_delta
    """
    rows = []
    for (model, mode, dataset), grp in df.groupby(["model", "mode", "dataset"]):
        conf  = grp["confidence"].fillna(0.5).values
        corr  = grp["correct"].astype(int).values
        cons  = grp["consistency"].values if "consistency" in grp.columns else np.ones(len(grp))
        size  = grp["size_b"].iloc[0]

        rows.append({
            "model":            model,
            "mode":             mode,
            "dataset":          dataset,
            "size_b":           size,
            "accuracy":         float(corr.mean()),
            "ece":              compute_ece(conf, corr),
            "brier":            compute_brier(conf, corr),
            "auroc":            compute_auroc(conf, corr),
            "aurc":             compute_aurc(conf, corr),
            "mean_consistency": float(cons.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Accuracy–Reliability divergence score  (Change 1 + 3)
# ---------------------------------------------------------------------------

def compute_divergence(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Divergence = accuracy_rank − reliability_rank (within dataset+mode group).
    High positive divergence: model looks accurate but is poorly calibrated.
    High negative divergence: model is well-calibrated but less accurate.

    reliability_score = 0.5*(1-ECE) + 0.3*AUROC + 0.2*(1-Brier)  [all in [0,1]]
    """
    df = summary_df.copy()
    df["reliability_score"] = (
        0.5 * (1 - df["ece"]) +
        0.3 * df["auroc"] +
        0.2 * (1 - df["brier"])
    )
    for (mode, dataset), grp in df.groupby(["mode", "dataset"]):
        idx = grp.index
        df.loc[idx, "accuracy_rank"]     = grp["accuracy"].rank(ascending=False)
        df.loc[idx, "reliability_rank"]  = grp["reliability_score"].rank(ascending=False)
    df["divergence"] = df["accuracy_rank"] - df["reliability_rank"]
    return df


# ---------------------------------------------------------------------------
# 9. Reliability–Pareto frontier  (Change 6)
# ---------------------------------------------------------------------------

def compute_reliability_frontier(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Marks Pareto-dominant points on the (accuracy, reliability_score) plane.
    A point is Pareto-dominant if no other point beats it on BOTH axes.
    """
    df = summary_df.copy()
    if "reliability_score" not in df.columns:
        df = compute_divergence(df)
    dominated = []
    for i, row in df.iterrows():
        is_dominated = any(
            (df.loc[j, "accuracy"] >= row["accuracy"] and
             df.loc[j, "reliability_score"] >= row["reliability_score"] and
             (df.loc[j, "accuracy"] > row["accuracy"] or
              df.loc[j, "reliability_score"] > row["reliability_score"]))
            for j in df.index if j != i
        )
        dominated.append(is_dominated)
    df["pareto_dominant"] = [not d for d in dominated]
    return df


# ---------------------------------------------------------------------------
# 10. Scaling trend (Change 5)
# ---------------------------------------------------------------------------

def compute_scaling_trend(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (mode, dataset): fit log-linear regression of metric vs log(size_b).
    Returns slope + r² for accuracy, ece, brier, auroc, mean_consistency.
    """
    from scipy.stats import pearsonr
    results = []
    metrics = ["accuracy", "ece", "brier", "auroc", "mean_consistency"]
    for (mode, dataset), grp in summary_df.groupby(["mode", "dataset"]):
        if len(grp) < 3:
            continue
        log_size = np.log(grp["size_b"].values)
        for metric in metrics:
            vals = grp[metric].values
            if np.std(vals) < 1e-9:
                continue
            r, p = pearsonr(log_size, vals)
            slope = np.polyfit(log_size, vals, 1)[0]
            results.append({
                "mode": mode, "dataset": dataset, "metric": metric,
                "slope": float(slope), "pearson_r": float(r), "p_value": float(p),
            })
    return pd.DataFrame(results)
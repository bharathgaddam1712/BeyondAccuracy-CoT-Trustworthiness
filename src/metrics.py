"""
Evaluation metrics for all three axes.

Axis 1 — Accuracy:  exact-match per dataset
Axis 2 — Calibration: Expected Calibration Error (ECE), reliability diagrams
Axis 3 — Consistency: agreement rate across paraphrases, consistency delta (CoT vs direct)
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Axis 1: Accuracy
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Lowercase, strip whitespace and punctuation for loose matching."""
    return answer.lower().strip().rstrip(".").strip()


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """Exact-match accuracy after normalization."""
    correct = sum(
        normalize_answer(p) == normalize_answer(r)
        for p, r in zip(predictions, references)
    )
    return correct / len(references) if references else 0.0


def compute_accuracy_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must have columns: model, mode (direct/cot), dataset, predicted, answer
    Returns accuracy grouped by (model, mode, dataset).
    """
    rows = []
    for (model, mode, dataset), group in df.groupby(["model", "mode", "dataset"]):
        acc = compute_accuracy(group["predicted"].tolist(), group["answer"].tolist())
        rows.append({"model": model, "mode": mode, "dataset": dataset, "accuracy": acc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Axis 2: Calibration
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: list[float],
    corrects: list[bool],
    n_bins: int = 10,
) -> dict:
    """
    Expected Calibration Error (ECE) — equal-width bins.

    Returns:
        ece: scalar ECE value
        bin_data: list of dicts with bin_mid, bin_acc, bin_conf, bin_count (for reliability diagram)
    """
    confidences = np.array(confidences)
    corrects = np.array(corrects, dtype=float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_conf  = confidences[mask].mean()
        bin_acc   = corrects[mask].mean()
        bin_count = mask.sum()
        ece += (bin_count / len(confidences)) * abs(bin_conf - bin_acc)
        bin_data.append({
            "bin_mid":   (lo + hi) / 2,
            "bin_conf":  float(bin_conf),
            "bin_acc":   float(bin_acc),
            "bin_count": int(bin_count),
        })

    return {"ece": float(ece), "bin_data": bin_data}


def compute_calibration_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df: model, mode, dataset, confidence (float 0-1), correct (bool)
    Returns ECE per (model, mode, dataset).
    """
    rows = []
    for (model, mode, dataset), group in df.groupby(["model", "mode", "dataset"]):
        valid = group.dropna(subset=["confidence"])
        if len(valid) < 10:
            print(f"[metrics] Warning: <10 valid confidence samples for {model}/{mode}/{dataset}")
            continue
        result = compute_ece(valid["confidence"].tolist(), valid["correct"].tolist())
        rows.append({
            "model":   model,
            "mode":    mode,
            "dataset": dataset,
            "ece":     result["ece"],
            "n":       len(valid),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Axis 3: Paraphrase Consistency
# ---------------------------------------------------------------------------

def compute_consistency(answers: list[str]) -> float:
    """
    Consistency = fraction of paraphrases that agree with the majority answer.
    answers: list of model answers for the same question across 5 paraphrases.
    """
    if not answers:
        return 0.0
    normalized = [normalize_answer(a) for a in answers]
    # majority answer
    from collections import Counter
    majority = Counter(normalized).most_common(1)[0][0]
    return sum(a == majority for a in normalized) / len(normalized)


def compute_consistency_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df: model, mode, dataset, question_id, paraphrase_idx (0-4), predicted
    Returns mean consistency per (model, mode, dataset).
    """
    rows = []
    for (model, mode, dataset), group in df.groupby(["model", "mode", "dataset"]):
        # Group by question, compute consistency per question
        per_q = group.groupby("question_id")["predicted"].apply(list)
        consistencies = [compute_consistency(answers) for answers in per_q]
        rows.append({
            "model":       model,
            "mode":        mode,
            "dataset":     dataset,
            "consistency": float(np.mean(consistencies)),
            "n_questions": len(consistencies),
        })
    return pd.DataFrame(rows)


def compute_consistency_delta(df_consistency: pd.DataFrame) -> pd.DataFrame:
    """
    Consistency delta = consistency(cot) - consistency(direct).
    Positive = CoT helps consistency. Negative = CoT hurts.
    Returns per (model, dataset).
    """
    pivot = df_consistency.pivot_table(
        index=["model", "dataset"], columns="mode", values="consistency"
    ).reset_index()
    if "cot" in pivot.columns and "direct" in pivot.columns:
        pivot["delta"] = pivot["cot"] - pivot["direct"]
    return pivot

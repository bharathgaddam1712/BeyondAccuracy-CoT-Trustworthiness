"""
Dataset loaders for all three evaluation axes.
Returns standardized dicts: {id, question, answer, dataset}
"""

import re
import json
from pathlib import Path
from datasets import load_dataset


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def load_gsm8k(n: int = 200, split: str = "test") -> list[dict]:
    """
    Load n GSM8K questions.
    Answer is the number after #### in the ground-truth solution.
    """
    ds = load_dataset("gsm8k", "main", split=split)
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        answer_match = re.search(r"####\s*([\d,\-\.]+)", row["answer"])
        answer = answer_match.group(1).replace(",", "") if answer_match else ""
        samples.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": answer,
            "dataset": "gsm8k",
        })
    return samples


# ---------------------------------------------------------------------------
# BIG-Bench Hard (BBH) — use 3 subtasks that have clear answer extraction
# ---------------------------------------------------------------------------

BBH_SUBTASKS = [
    "logical_deduction_three_objects",
    "causal_judgement",
    "tracking_shuffled_objects_three_objects",
]

def load_bbh(n_per_subtask: int = 60) -> list[dict]:
    """
    Load BBH samples from selected subtasks.
    Answers are always single-option strings like "(A)", "(B)", etc.
    """
    samples = []
    for task in BBH_SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"[warn] BBH subtask {task} failed: {e}")
            continue
        for i, row in enumerate(ds):
            if i >= n_per_subtask:
                break
            samples.append({
                "id": f"bbh_{task}_{i}",
                "question": row["input"],
                "answer": row["target"].strip(),
                "dataset": "bbh",
                "subtask": task,
            })
    return samples


# ---------------------------------------------------------------------------
# TruthfulQA — MC1 (single correct answer, multiple choice)
# ---------------------------------------------------------------------------

def load_truthfulqa(n: int = 200) -> list[dict]:
    """
    Load TruthfulQA MC1 questions.
    Answer is the single correct choice string.
    Uses multiple-choice format for deterministic evaluation (avoids LLM judge).
    """
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        # MC1: exactly one correct answer
        mc1_targets = row["mc1_targets"]
        correct_idx = mc1_targets["labels"].index(1)
        correct_answer = mc1_targets["choices"][correct_idx]

        # Build an A/B/C/D formatted question
        choices = mc1_targets["choices"][:4]  # cap at 4 options
        choice_text = "\n".join(
            f"({chr(65+j)}) {c}" for j, c in enumerate(choices)
        )
        question = f"{row['question']}\n\n{choice_text}"
        answer_letter = chr(65 + correct_idx) if correct_idx < 4 else None

        if answer_letter is None:
            continue  # skip if correct answer is beyond first 4 choices

        samples.append({
            "id": f"tqa_{i}",
            "question": question,
            "answer": f"({answer_letter})",      # e.g. "(A)"
            "answer_text": correct_answer,
            "dataset": "truthfulqa",
        })
    return samples[:n]


# ---------------------------------------------------------------------------
# Paraphrase cache — load / save
# ---------------------------------------------------------------------------

def save_paraphrases(paraphrases: dict[str, list[str]], path: str | Path) -> None:
    """Save {question_id -> [paraphrase_1, ..., paraphrase_5]} to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(paraphrases, f, indent=2)
    print(f"[datasets] Saved paraphrases to {path}")


def load_paraphrases(path: str | Path) -> dict[str, list[str]]:
    """Load paraphrase cache."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Combined loader — used by experiment scripts
# ---------------------------------------------------------------------------

def load_all_datasets(
    n_gsm8k: int = 200,
    n_bbh_per_subtask: int = 60,
    n_truthfulqa: int = 200,
) -> dict[str, list[dict]]:
    print("[datasets] Loading GSM8K...")
    gsm8k = load_gsm8k(n_gsm8k)
    print(f"  {len(gsm8k)} samples")

    print("[datasets] Loading BBH...")
    bbh = load_bbh(n_bbh_per_subtask)
    print(f"  {len(bbh)} samples")

    print("[datasets] Loading TruthfulQA...")
    tqa = load_truthfulqa(n_truthfulqa)
    print(f"  {len(tqa)} samples")

    return {"gsm8k": gsm8k, "bbh": bbh, "truthfulqa": tqa}

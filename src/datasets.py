"""
Dataset loaders for Paper A — Beyond Accuracy.
All datasets loaded from HuggingFace. Paraphrase cache saved to JSON.
"""

import json
import random
from pathlib import Path
from datasets import load_dataset


def load_gsm8k(n: int = 200, seed: int = 42) -> list[dict]:
    """Load n questions from GSM8K test split."""
    ds = load_dataset("gsm8k", "main", split="test")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    return [
        {
            "id":       f"gsm8k_{i}",
            "question": ds[i]["question"],
            "answer":   ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
            "dataset":  "gsm8k",
        }
        for i in indices
    ]


def load_bbh(n_per_subtask: int = 60, seed: int = 42) -> list[dict]:
    """
    Load n_per_subtask questions from 3 BBH subtasks:
      - logical_deduction_three_objects
      - causal_judgement
      - date_understanding
    """
    subtasks = [
        "logical_deduction_three_objects",
        "causal_judgement",
        "date_understanding",
    ]
    random.seed(seed)
    all_samples = []
    for task in subtasks:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test")
            indices = random.sample(range(len(ds)), min(n_per_subtask, len(ds)))
            for i in indices:
                all_samples.append({
                    "id":       f"bbh_{task}_{i}",
                    "question": ds[i]["input"],
                    "answer":   ds[i]["target"].strip(),
                    "dataset":  "bbh",
                    "subtask":  task,
                })
        except Exception as e:
            print(f"[datasets] Warning: could not load BBH subtask {task}: {e}")
    return all_samples


def load_truthfulqa(n: int = 200, seed: int = 42) -> list[dict]:
    """
    Load n questions from TruthfulQA as MCQ.
    Uses mc1_targets (single correct answer).
    """
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    samples = []
    for i in indices:
        row     = ds[i]
        choices = row["mc1_targets"]["choices"]
        labels  = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        paired  = list(zip(choices, labels))
        random.shuffle(paired)
        shuffled_choices, shuffled_labels = zip(*paired)
        correct_letter = chr(65 + shuffled_labels.index(1))
        samples.append({
            "id":             f"tqa_{i}",
            "question":       row["question"],
            "choices":        list(shuffled_choices),
            "answer":         correct_letter,
            "dataset":        "truthfulqa",
        })
    return samples


def load_paraphrase_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_paraphrase_cache(cache: dict, path: Path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def get_paraphrases(sample_id: str, cache: dict) -> list[str] | None:
    return cache.get(sample_id)


def set_paraphrases(sample_id: str, paraphrases: list[str], cache: dict):
    cache[sample_id] = paraphrases


def save_samples(samples: list[dict], path: Path):
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)


def load_samples(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)

"""
Prompt templates for Paper A — Beyond Accuracy.
Three evaluation axes:
  1. Accuracy      — direct answer vs chain-of-thought
  2. Calibration   — same prompts + confidence elicitation suffix
  3. Consistency   — paraphrase variants of the same question
"""

# ---------------------------------------------------------------------------
# Confidence elicitation suffix (appended for calibration axis)
# ---------------------------------------------------------------------------

CONFIDENCE_SUFFIX = "\nConfidence: <0-100>"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_DIRECT = (
    "You are a helpful assistant. Answer the question concisely. "
    "Give only the final answer with no explanation."
)

SYSTEM_COT = (
    "You are a helpful assistant. Think step by step before giving your final answer. "
    "End your response with 'Answer: <your answer>'."
)


# ---------------------------------------------------------------------------
# Dataset-specific message builders
# ---------------------------------------------------------------------------

def _gsm8k_messages(question: str, mode: str) -> list[dict]:
    if mode == "cot":
        return [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user",   "content": f"Question: {question}"},
        ]
    return [
        {"role": "system", "content": SYSTEM_DIRECT},
        {"role": "user",   "content": f"Question: {question}\nAnswer:"},
    ]


def _bbh_messages(question: str, mode: str) -> list[dict]:
    if mode == "cot":
        return [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user",   "content": f"Question: {question}"},
        ]
    return [
        {"role": "system", "content": SYSTEM_DIRECT},
        {"role": "user",   "content": f"Question: {question}\nAnswer:"},
    ]


def _truthfulqa_messages(question: str, mode: str, choices: list[str] | None) -> list[dict]:
    choices = choices or []
    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    if mode == "cot":
        return [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user",   "content": f"Question: {question}\n{choices_str}"},
        ]
    return [
        {"role": "system", "content": SYSTEM_DIRECT},
        {"role": "user",   "content": f"Question: {question}\n{choices_str}\nAnswer (letter only):"},
    ]


_DATASET_BUILDERS = {
    "gsm8k":      lambda q, mode, choices=None: _gsm8k_messages(q, mode),
    "bbh":        lambda q, mode, choices=None: _bbh_messages(q, mode),
    "truthfulqa": lambda q, mode, choices=None: _truthfulqa_messages(q, mode, choices),
}


# ---------------------------------------------------------------------------
# Public API — called directly by the notebooks
# ---------------------------------------------------------------------------

def make_accuracy_prompt(question: str, mode: str, dataset: str, choices: list[str] | None = None) -> list[dict]:
    """Direct or CoT prompt for the accuracy axis (Step 2)."""
    return _DATASET_BUILDERS[dataset](question, mode, choices)


def make_calibration_prompt(question: str, mode: str, dataset: str, choices: list[str] | None = None) -> list[dict]:
    """Accuracy prompt + verbalized confidence elicitation suffix (Step 3)."""
    msgs = [m.copy() for m in make_accuracy_prompt(question, mode, dataset, choices)]
    for i in reversed(range(len(msgs))):
        if msgs[i]["role"] == "user":
            msgs[i]["content"] = msgs[i]["content"] + CONFIDENCE_SUFFIX
            break
    return msgs


def make_consistency_prompt(question: str, mode: str, dataset: str, choices: list[str] | None = None) -> list[dict]:
    """Same as accuracy prompt — used to re-run paraphrase variants (Step 4)."""
    return make_accuracy_prompt(question, mode, dataset, choices)


def make_paraphrase_gen_prompt(question: str) -> list[dict]:
    """Ask a model to generate 5 numbered paraphrases of a question (Step 1)."""
    return [
        {"role": "system", "content": (
            "You are a paraphrasing assistant. Given a question, generate 5 paraphrases "
            "that preserve the exact meaning but vary the wording, style, and phrasing. "
            "Output exactly 5 lines, numbered 1-5, one paraphrase per line, nothing else."
        )},
        {"role": "user", "content": f"Original question: {question}\n\nGenerate 5 paraphrases:"},
    ]

"""
Prompt templates for Paper A — Beyond Accuracy.
Three evaluation axes:
  1. Accuracy      — direct answer vs chain-of-thought
  2. Calibration   — same prompts + confidence elicitation suffix
  3. Consistency   — paraphrase variants of the same question
"""

CONFIDENCE_SUFFIX = "\nConfidence: <0-100>"


SYSTEM_DIRECT = (
    "You are a helpful assistant. Answer the question concisely. "
    "Give only the final answer with no explanation."
)

SYSTEM_COT = (
    "You are a helpful assistant. Think step by step before giving your final answer. "
    "End your response with 'Answer: <your answer>'."
)


def make_gsm8k_direct(question: str) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_DIRECT},
        {"role": "user",    "content": f"Question: {question}\nAnswer:"},
    ]


def make_gsm8k_cot(question: str) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_COT},
        {"role": "user",    "content": f"Question: {question}"},
    ]


def make_bbh_direct(question: str) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_DIRECT},
        {"role": "user",    "content": f"Question: {question}\nAnswer:"},
    ]


def make_bbh_cot(question: str) -> list[dict]:
    return [
        {"role": "system",  "content": SYSTEM_COT},
        {"role": "user",    "content": f"Question: {question}"},
    ]


def make_truthfulqa_direct(question: str, choices: list[str]) -> list[dict]:
    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return [
        {"role": "system",  "content": SYSTEM_DIRECT},
        {"role": "user",    "content": f"Question: {question}\n{choices_str}\nAnswer (letter only):"},
    ]


def make_truthfulqa_cot(question: str, choices: list[str]) -> list[dict]:
    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return [
        {"role": "system",  "content": SYSTEM_COT},
        {"role": "user",    "content": f"Question: {question}\n{choices_str}"},
    ]


def make_calibration_prompt(base_messages: list[dict]) -> list[dict]:
    """Takes any base prompt and appends confidence elicitation to last user turn."""
    msgs = [m.copy() for m in base_messages]
    for i in reversed(range(len(msgs))):
        if msgs[i]["role"] == "user":
            msgs[i]["content"] = msgs[i]["content"] + CONFIDENCE_SUFFIX
            break
    return msgs


def make_paraphrase_prompt(question: str, variant_idx: int) -> list[dict]:
    styles = [
        "Rephrase this question using different words but preserve the exact meaning.",
        "Rewrite this question in a more formal style.",
        "Rewrite this question in a simpler, more casual style.",
        "Rephrase this question starting with a different subject.",
        "Rewrite this question as if explaining to a student.",
        "Paraphrase this question using synonyms where possible.",
    ]
    style = styles[variant_idx % len(styles)]
    return [
        {"role": "system", "content": "You are a paraphrasing assistant. Output only the rewritten question, nothing else."},
        {"role": "user",   "content": f"{style}\n\nOriginal: {question}\n\nParaphrase:"},
    ]

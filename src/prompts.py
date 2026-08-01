"""
Prompt templates for all three evaluation axes.

Design principles:
- Direct prompts ask for answer only — no reasoning.
- CoT prompts use "Let's think step by step." and end with "Therefore, the answer is:"
- Calibration prompts append a confidence elicitation after the answer.
- All prompts use chat-style messages (role / content) — converted to model-specific
  format by the tokenizer's apply_chat_template method.
"""

from typing import Literal

PromptMode = Literal["direct", "cot"]


# ---------------------------------------------------------------------------
# Axis 1 — Accuracy (GSM8K, BBH, TruthfulQA)
# ---------------------------------------------------------------------------

SYSTEM_ACCURACY = (
    "You are a helpful and precise assistant. "
    "Answer the question correctly and concisely."
)

def make_accuracy_prompt(question: str, mode: PromptMode, dataset: str) -> list[dict]:
    """
    Returns a chat messages list ready for apply_chat_template.
    dataset: 'gsm8k' | 'bbh' | 'truthfulqa'
    """
    if dataset == "gsm8k":
        answer_instruction = (
            "At the end, write your final numerical answer on a new line "
            "in the format: #### <number>"
        )
    elif dataset == "bbh":
        answer_instruction = (
            "At the end, write your final answer on a new line "
            "in the format: The answer is: <option>"
        )
    else:  # truthfulqa
        answer_instruction = (
            "Answer with a single factually correct sentence. "
            "Do not add qualifications or caveats."
        )

    if mode == "direct":
        user_content = f"{question}\n\n{answer_instruction}"
    else:  # cot
        user_content = (
            f"{question}\n\n"
            "Let's think step by step.\n\n"
            f"{answer_instruction}"
        )

    return [
        {"role": "system", "content": SYSTEM_ACCURACY},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Axis 2 — Calibration (verbalized confidence after answering)
# ---------------------------------------------------------------------------

CONFIDENCE_SUFFIX = (
    "\n\nNow rate your confidence in this answer on a scale from 0 to 100, "
    "where 0 = completely uncertain and 100 = completely certain. "
    "Write only an integer on the last line in this format: Confidence: <number>"
)

def make_calibration_prompt(question: str, mode: PromptMode, dataset: str) -> list[dict]:
    """Same as accuracy prompt but with confidence elicitation appended."""
    messages = make_accuracy_prompt(question, mode, dataset)
    messages[-1]["content"] += CONFIDENCE_SUFFIX
    return messages


# ---------------------------------------------------------------------------
# Axis 3 — Paraphrase Consistency
# ---------------------------------------------------------------------------

PARAPHRASE_GENERATION_PROMPT = """Generate exactly 5 semantically equivalent paraphrases of the following question.
Each paraphrase must:
- Ask exactly the same question (same answer must be correct)
- Use different wording, sentence structure, or phrasing
- Be numbered 1–5

Original question: {question}

Output format:
1. <paraphrase>
2. <paraphrase>
3. <paraphrase>
4. <paraphrase>
5. <paraphrase>"""


def make_paraphrase_gen_prompt(question: str) -> list[dict]:
    """Prompt to generate 5 paraphrases. Run once, results cached."""
    return [
        {"role": "system", "content": "You are a linguistic expert who rewrites questions precisely."},
        {"role": "user",   "content": PARAPHRASE_GENERATION_PROMPT.format(question=question)},
    ]


def make_consistency_prompt(question: str, mode: PromptMode, dataset: str) -> list[dict]:
    """Same as accuracy prompt — applied to each of the 5 paraphrases."""
    return make_accuracy_prompt(question, mode, dataset)

"""
Inference utilities — batched generation with entropy computation.

Key design:
- generate_responses: standard text generation, returns decoded strings
- generate_with_entropy: returns text + per-token entropy of the first generated token
  (used to cross-validate verbalized confidence against token-level uncertainty)
"""

import re
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def format_messages(messages: list[dict], tokenizer: PreTrainedTokenizerBase) -> str:
    """Apply chat template. Falls back to simple concatenation if unsupported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[dict]],
    max_new_tokens: int = 512,
    batch_size: int = 4,
    temperature: float = 0.0,       # greedy by default for reproducibility
) -> list[str]:
    """
    Generate text responses for a list of chat-message prompts.
    Returns decoded strings (model response only, prompt stripped).
    """
    all_responses = []
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch_messages = prompts[i : i + batch_size]
        formatted = [format_messages(m, tokenizer) for m in batch_messages]

        encodings = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            if temperature == 0.0:
                outputs = model.generate(
                    **encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

        input_lengths = encodings["input_ids"].shape[1]
        for out in outputs:
            new_tokens = out[input_lengths:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            all_responses.append(decoded)

        if i % (batch_size * 5) == 0:
            print(f"  [inference] {i + len(batch_messages)}/{len(prompts)} done")

    return all_responses


# ---------------------------------------------------------------------------
# Entropy-aware generation (for cross-validation with verbalized confidence)
# ---------------------------------------------------------------------------

def generate_with_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[dict]],
    max_new_tokens: int = 512,
    batch_size: int = 2,            # smaller batch — storing logits is memory-heavy
) -> list[dict]:
    """
    Generate responses AND compute first-token entropy.
    Returns list of dicts: {response: str, first_token_entropy: float, mean_entropy: float}

    first_token_entropy: entropy of P(first generated token) — captures initial uncertainty
    mean_entropy: mean token entropy across all generated tokens
    """
    results = []
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch_messages = prompts[i : i + batch_size]
        formatted = [format_messages(m, tokenizer) for m in batch_messages]

        encodings = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,         # get logits per step
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = encodings["input_ids"].shape[1]
        # scores: tuple of (batch, vocab) tensors, one per generated token
        scores = torch.stack(outputs.scores, dim=1)  # (batch, seq, vocab)
        probs = torch.softmax(scores, dim=-1)
        entropies = -(probs * (probs + 1e-9).log()).sum(dim=-1)  # (batch, seq)

        for j, seq in enumerate(outputs.sequences):
            new_tokens = seq[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            token_entropies = entropies[j].cpu().numpy()
            results.append({
                "response": decoded,
                "first_token_entropy": float(token_entropies[0]) if len(token_entropies) > 0 else 0.0,
                "mean_entropy": float(token_entropies.mean()) if len(token_entropies) > 0 else 0.0,
            })

    return results


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_gsm8k_answer(response: str) -> str | None:
    """Extract number after #### marker."""
    match = re.search(r"####\s*([\d,\-\.]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in response
    numbers = re.findall(r"[\d,\-\.]+", response)
    return numbers[-1].replace(",", "") if numbers else None


def parse_bbh_answer(response: str) -> str | None:
    """Extract answer after 'The answer is:' marker."""
    match = re.search(r"[Tt]he answer is[:\s]+(.+?)[\.\n]", response)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else None


def parse_truthfulqa_answer(response: str) -> str:
    """Return the full response — TruthfulQA uses judge-based scoring."""
    return response.strip()


def parse_confidence(response: str) -> float | None:
    """
    Extract verbalized confidence score (0–100).
    Returns None if not found — treated as NaN in calibration analysis.
    """
    match = re.search(r"[Cc]onfidence[:\s]+(\d+)", response)
    if match:
        val = int(match.group(1))
        return max(0.0, min(100.0, val)) / 100.0  # normalize to [0, 1]
    # Fallback: last standalone integer in response
    matches = re.findall(r"\b(\d{1,3})\b", response)
    if matches:
        val = int(matches[-1])
        if 0 <= val <= 100:
            return val / 100.0
    return None

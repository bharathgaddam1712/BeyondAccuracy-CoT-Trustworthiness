"""
Inference utilities for Paper A — Beyond Accuracy.

generate_responses()        — batched greedy, returns text only
generate_with_entropy()     — single item, returns text + logit features
parse_answer_*()            — dataset-specific answer extractors
parse_confidence()          — verbalized confidence extractor
"""

import re
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def format_messages(messages: list[dict], tokenizer: PreTrainedTokenizerBase) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[dict]],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> list[str]:
    all_responses = []
    device = next(model.parameters()).device

    for i in range(0, len(prompts), batch_size):
        batch     = prompts[i : i + batch_size]
        formatted = [format_messages(m, tokenizer) for m in batch]
        enc = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            decoded = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            all_responses.append(decoded)

        if i % (batch_size * 5) == 0:
            print(f"  [inference] {i + len(batch)}/{len(prompts)}")

    return all_responses


def generate_with_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    max_new_tokens: int = 256,
) -> dict:
    """
    Returns:
        response            decoded text
        first_token_entropy H(first token distribution)
        mean_entropy        mean token entropy across response
        top1_prob           P(argmax) of first token
        margin_top1_top2    P(top1) - P(top2) of first token
    """
    device    = next(model.parameters()).device
    formatted = format_messages(messages, tokenizer)
    enc = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    response  = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True).strip()

    if out.scores:
        first_logits = out.scores[0][0]
        probs        = torch.softmax(first_logits, dim=-1).cpu().float().numpy()
        sorted_probs = np.sort(probs)[::-1]

        first_entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        top1_prob     = float(sorted_probs[0])
        margin        = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.0

        all_entropies = []
        for score in out.scores:
            p = torch.softmax(score[0], dim=-1).cpu().float().numpy()
            all_entropies.append(float(-(p * np.log(p + 1e-9)).sum()))
        mean_entropy = float(np.mean(all_entropies))
    else:
        first_entropy = mean_entropy = top1_prob = margin = 0.0

    return {
        "response":            response,
        "first_token_entropy": first_entropy,
        "mean_entropy":        mean_entropy,
        "top1_prob":           top1_prob,
        "margin_top1_top2":    margin,
    }


def normalize_answer(s: str) -> str:
    import string
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def parse_gsm8k_answer(response: str) -> str | None:
    """Extract final numeric answer. Looks for #### marker or last number."""
    m = re.search(r'####\s*([+-]?\d[\d,\.]*)', response)
    if m:
        return m.group(1).replace(',', '').strip()
    nums = re.findall(r'Answer:\s*([+-]?\d[\d,\.]*)', response, re.IGNORECASE)
    if nums:
        return nums[-1].replace(',', '').strip()
    all_nums = re.findall(r'[+-]?\d[\d,\.]*', response)
    return all_nums[-1].replace(',', '') if all_nums else None


def parse_bbh_answer(response: str) -> str | None:
    """Extract answer from BBH response. Looks for Answer: or last line."""
    m = re.search(r'[Aa]nswer[:\s]+(.+)', response)
    if m:
        return normalize_answer(m.group(1).split('\n')[0])
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return normalize_answer(lines[-1]) if lines else None


def parse_truthfulqa_answer(response: str) -> str | None:
    """Extract MCQ letter (A/B/C/D) from response."""
    m = re.search(r'\b([A-D])\b', response.upper())
    return m.group(1) if m else None


def parse_confidence(response: str) -> float | None:
    """Extract verbalized confidence 0–100, normalize to [0,1]."""
    m = re.search(r'[Cc]onfidence[:\s]+(\d+)', response)
    if m:
        return max(0.0, min(100.0, int(m.group(1)))) / 100.0
    nums = re.findall(r'\b(\d{1,3})\b', response)
    if nums:
        v = int(nums[-1])
        if 0 <= v <= 100:
            return v / 100.0
    return None

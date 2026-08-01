# LIGHT Paper — Complete Runbook

## Paper Title
"Beyond Accuracy: Calibration, Confidence, and Paraphrase Robustness of CoT Reasoning in Sub-3B Language Models"

**Workshop:** LIGHT @ NeurIPS 2026  
**Deadline:** August 29, 2026

---

## Project Structure

```
light_paper/
├── src/
│   ├── models.py       — model registry, loader (7 models, 0.5B→8B)
│   ├── prompts.py      — all prompt templates (accuracy / calibration / consistency)
│   ├── inference.py    — batched generation, entropy, answer parsers
│   ├── datasets.py     — GSM8K, BBH, TruthfulQA loaders + paraphrase cache
│   └── metrics.py      — ECE, reliability diagram, consistency, delta metrics
│
├── notebooks/
│   ├── step0_kaggle_setup.ipynb        — GPU check + smoke test (run once)
│   ├── step1_setup_and_data.ipynb      — Download datasets, generate paraphrases
│   ├── step2_accuracy.ipynb            — Axis 1: Accuracy (Direct vs CoT)
│   ├── step3_calibration.ipynb         — Axis 2: ECE + reliability diagrams
│   ├── step4_consistency.ipynb         — Axis 3: Paraphrase consistency
│   └── step5_analysis_and_plots.ipynb  — All figures + hypothesis test
│
├── data/
│   ├── datasets.json     — all questions + answers (generated in step 1)
│   └── paraphrases.json  — 5 paraphrases per question (generated in step 1)
│
├── results/              — all CSVs (generated in steps 2–4)
├── figures/              — all PNG figures (generated in step 5)
└── paper/                — LaTeX files (write after step 5)
```

---

## Step-by-Step Execution on Kaggle

### Before you start
1. Upload the entire `light_paper/` folder as a **Kaggle Dataset** named `light-paper`
2. Create a new Kaggle Notebook, attach GPU (T4 × 2), set accelerator to GPU
3. Add `light-paper` as a Dataset input to your notebook
4. Set internet ON (needed for HuggingFace model downloads and datasets)

---

### STEP 0 — Smoke Test (~5 min)
**Notebook:** `step0_kaggle_setup.ipynb`

What it does:
- Verifies both T4s are available and VRAM is correct
- Copies `src/` to working directory
- Loads Qwen2.5-0.5B and runs one test prompt

Expected output:
```
GPU 0: Tesla T4, 15.8 GB
GPU 1: Tesla T4, 15.8 GB
Test response: #### 4
✅ Smoke test passed
```

---

### STEP 1 — Dataset + Paraphrases (~15 min)
**Notebook:** `step1_setup_and_data.ipynb`

What it does:
- Downloads GSM8K (200), BBH (~180), TruthfulQA (200) from HuggingFace
- Generates 5 paraphrases per GSM8K + TruthfulQA question using Qwen2.5-3B locally
- Saves everything to `data/`

**Checkpoint behavior:** If `data/paraphrases.json` already exists, skips generation.

Outputs:
- `data/datasets.json`
- `data/paraphrases.json`

**Inspect before continuing:** Check paraphrase quality in the last cell. If fewer than 90% of questions have exactly 5 paraphrases, re-run.

---

### STEP 2 — Accuracy Experiments (~4–6 hours)
**Notebook:** `step2_accuracy.ipynb`

What it does:
- Loops over 7 models × 2 modes (direct/cot) × 3 datasets
- Saves after every (model, mode, dataset) combo — safe to interrupt and resume
- Prints accuracy preview after each combo

**Resume:** Just re-run — it detects completed combos from `results/accuracy_raw.csv`

Outputs:
- `results/accuracy_raw.csv`
- `results/accuracy_summary.csv`

**Check:** The accuracy pivot table at the end should show clear differences between models. If all values are 0 or NaN, the parser is broken — check `raw_response` column in the CSV.

---

### STEP 3 — Calibration (~3–4 hours)
**Notebook:** `step3_calibration.ipynb`

What it does:
- Same loop but appends confidence elicitation to every prompt
- Parses verbalized confidence (0–100) from response
- Computes ECE per (model, mode, dataset)

**Watch for:** `n_valid` in the print output. If < 80% of responses have a parseable confidence, the prompt needs adjustment. Common failure: model responds without the "Confidence: X" line.

Outputs:
- `results/calibration_raw.csv`
- `results/calibration_ece.csv`
- `results/calibration_bin_data.json`

---

### STEP 4 — Consistency (~8–10 hours, longest step)
**Notebook:** `step4_consistency.ipynb`

What it does:
- For each question: original + 5 paraphrases = 6 inference calls per model per mode
- 400 questions × 6 × 7 models × 2 modes = ~33,600 inference calls
- Checkpoints after every (model, mode, dataset) combo

**Tip:** Run this as a Kaggle session with maximum allowed time (12h). If it times out, just restart — it will resume from where it left off.

Outputs:
- `results/consistency_raw.csv`
- `results/consistency_summary.csv`
- `results/consistency_delta.csv`

---

### STEP 5 — Analysis & Figures (~15 min)
**Notebook:** `step5_analysis_and_plots.ipynb`

What it does:
- Loads all 3 result files
- Generates 5 figures + runs hypothesis test

**The hypothesis test at the end tells you the main result:**
- H1: Does CoT worsen calibration below 3B?
- H2: Does CoT hurt consistency below 3B?
- H3: Does the pattern reverse at ≥3B?

If all 3 are CONFIRMED → main paper claim is validated.
If H1/H2 are NOT confirmed → check the data and adjust claim.

Outputs: All `.png` files in `figures/`

---

## Models Used

| Name | HF ID | Size | Load mode |
|---|---|---|---|
| Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | fp16 |
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | fp16 |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B-Instruct | 1B | fp16 |
| Gemma-2-2B | google/gemma-2-2b-it | 2B | fp16 |
| Qwen2.5-3B | Qwen/Qwen2.5-3B-Instruct | 3B | fp16 |
| Llama-3.2-3B | meta-llama/Llama-3.2-3B-Instruct | 3B | fp16 |
| Llama-3.1-8B | meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | 4-bit |

All models are public on HuggingFace. No paid API keys needed.

---

## Compute Budget

| Step | Est. GPU hours | Notes |
|---|---|---|
| Step 1 | 0.3h | Paraphrase generation |
| Step 2 | 5h | 7 models × 3 datasets × 2 modes |
| Step 3 | 4h | 7 models × 2 datasets × 2 modes |
| Step 4 | 9h | 7 models × 2 datasets × 2 modes × 6 variants |
| Step 5 | 0.2h | CPU-only analysis |
| **Total** | **~18h** | Kaggle gives 30h/week |

---

## Datasets

| Dataset | HF Path | n | What's measured |
|---|---|---|---|
| GSM8K | `gsm8k`, `main` | 200 | Math reasoning accuracy + calibration + consistency |
| BIG-Bench Hard | `lukaemon/bbh` | ~180 | Logical reasoning accuracy only |
| TruthfulQA | `truthful_qa`, `multiple_choice` | 200 | Factual accuracy + calibration + consistency |

---

## Key Files to Check After Each Step

| After step | File to inspect | What to verify |
|---|---|---|
| 1 | `data/paraphrases.json` | Paraphrases are semantically equivalent (spot-check 5) |
| 2 | `results/accuracy_summary.csv` | Accuracy increases with model size; CoT helps at 8B |
| 3 | `results/calibration_ece.csv` | ECE values between 0.05–0.3 (outside this range = parser issue) |
| 4 | `results/consistency_summary.csv` | Consistency above 0.5 for all models |
| 5 | `figures/fig5_combined_trustworthiness.png` | Red cells at small sizes = main finding |

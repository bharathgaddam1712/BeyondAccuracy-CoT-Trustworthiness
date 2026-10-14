# Beyond Accuracy: Locating Trust, Calibration, and Internal Signals in Sub-3B Language Models

This repository contains the complete experimental code, dataset loaders, internal activation probing pipeline, statistical analyses, and publication figures for **"Beyond Accuracy: Calibration, Confidence, and Internal Trust Signals of Chain-of-Thought in Sub-3B Models"** (LIGHT Workshop @ NeurIPS 2026).

---

## 📖 Key Documentation
* **Comprehensive Experimental Report:** [`EXPERIMENTS_AND_RESULTS_DETAILED.md`](EXPERIMENTS_AND_RESULTS_DETAILED.md) — Detailed rationale, mathematical formulations, and empirical results across all 11 behavioral experiments, causal step/length ablations, and the 13,600-trajectory internal activation probing pipeline.

---

## 📁 Repository Structure

```
HAdes 2.0/
├── data/                                 # Evaluation benchmarks (GSM8K, BBH, TruthfulQA, ARC)
│   ├── datasets.json                     # Pooled question bank
│   └── paraphrases.json                  # Semantic paraphrases
├── experiments/                          # Core experiment execution pipelines
│   ├── internal_signals/                 # Hidden state extraction & linear probing
│   │   ├── extract_v2.py                 # Scaled extraction across 680 questions & 4 modes
│   │   ├── probe_v2.py                   # 5-fold CV linear probing & cross-dataset transfer
│   │   ├── extract_internal_signals.py   # Baseline extraction pipeline
│   │   └── probe_analysis.py             # Layer-wise probe diagnostics
│   ├── scaling_and_diagnostics/          # Multi-scale ablation and consistency scripts
│   │   ├── e5_self_consistency_calibration.py
│   │   ├── e9_shuffled_cot.py
│   │   ├── overnight_consistency_master.py
│   │   └── overnight_gpu_run.py
│   └── kaggle/                           # Multi-GPU / Kaggle pipelines
│       ├── kaggle_consistency_pipeline.py
│       ├── kaggle_e7_e8_pipeline.py
│       └── kaggle_gated_e1_e3_e4_pipeline.py
├── analysis/                             # Statistical evaluation and audit scripts
│   ├── deep_rigorous_analysis.py         # Full statistical test battery
│   ├── e11_brier_decomposition.py        # Reliability vs. resolution Brier decomposition
│   ├── e2_trace_quality_analysis.py      # Reasoning trace validity analysis
│   ├── e3_analysis.py                    # Entropy dynamics analysis
│   ├── e4_analysis.py                    # Filler-CoT & CRU index evaluation
│   ├── e10_scaling_law.py                # Trust metrics scaling laws
│   ├── matched_subset_calibration.py     # Strict intersection ECE & survivorship analysis
│   └── consistency_audit.py              # Semantic clustering consistency verification
├── results/                              # Master experimental outputs and CSVs
│   ├── summary/                          # Audited summary CSVs
│   │   ├── probe_v2_summary.csv          # Scale-up probe AUROC & confidence intervals
│   │   ├── probe_v2_transfer.csv         # Cross-dataset OOD generalization metrics
│   │   ├── probe_v2_layer_curves.csv     # Layer-wise AUROC trajectories
│   │   ├── master_cru_summary.csv        # Cross-scale CRU table
│   │   ├── matched_subset_calibration.csv# Strict intersection calibration metrics
│   │   ├── e11_brier_decomposition.csv   # Murphy Brier score decomposition
│   │   └── calibration_ece.csv           # Master ECE evaluations
│   └── raw/                              # Raw generation and calibration logs
├── figures/                              # Publication figures
│   ├── fig_probe_v2_summary.png          # Verbalized vs. Probe AUROC across model scale
│   ├── fig_probe_v2_gap.png              # The decodability gap ("Knows But Can't Say")
│   ├── fig_probe_v2_transfer.png         # Cross-dataset OOD probe transfer
│   ├── fig_reasoning_structure_ladder.png# 4-rung ladder across Direct → Filler → Scratchpad → CoT
│   ├── fig_cru_scaling_transition.png    # CRU index scaling transition
│   └── fig_shuffled_cot_Qwen2.5-0.5B.png # Shuffled-CoT causal ablation
├── paper/                                # LaTeX publication drafts
│   ├── main.tex                          # Primary LaTeX submission draft
│   ├── arxiv.tex                         # Complete arXiv manuscript
│   ├── neurips_2026.sty                  # Style sheet
│   └── references.bib                    # BibTeX bibliography
├── src/                                  # Shared core library
│   ├── models.py                         # HF Model loader with 4-bit / fp16 / SDPA
│   ├── prompts.py                        # Prompt templates & confidence elicitations
│   ├── inference.py                      # Delimiter parsers (scratchpad, CoT, filler)
│   ├── datasets.py                       # GSM8K / BBH / TruthfulQA / ARC loaders
│   └── metrics.py                        # ECE, Brier score, AUROC implementations
├── requirements.txt                      # Project dependencies
└── README.md                             # Repository overview
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation & Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Scaled Linear Probing & Reasoning Ladder
Run 5-fold cross-validated linear probing on extracted layer activations across 680 pooled questions:
```bash
python experiments/internal_signals/probe_v2.py
```
This produces:
* `results/summary/probe_v2_summary.csv`
* `results/summary/probe_v2_transfer.csv`
* `results/summary/probe_v2_layer_curves.csv`
* Publication figures in `figures/`

### 3. Causal Ablations & Statistical Decompositions
* **Filler-CoT & CRU Evaluation:** `python analysis/e4_analysis.py`
* **Information-Theoretic Murphy Decomposition:** `python analysis/e11_brier_decomposition.py`
* **Trace Dynamics & Entropy:** `python analysis/e3_analysis.py`

---

## 🔬 Core Scientific Findings

1. **"Knows But Can't Say" Decodability Gap:** Sub-3B models fail to verbalize calibrated confidence in natural language ($\text{AUROC} \approx 0.30 - 0.51$), but an $\ell_2$-regularized linear probe on intermediate residual streams decodes correctness with high fidelity ($\text{AUROC} \approx 0.68 - 0.83$). The failure is an **emission bottleneck**, not an epistemic one.
2. **The Token-Padding Regime ($\text{CRU} \le 0$):** In sub-1.5B models, inserting unstructured filler text ("Filler-CoT") produces equal or superior calibration gains compared to step-by-step math derivations ($p = 0.1456$ on Llama-3.2-1B).
3. **Emergence of Scratchpad Meta-Control at 3B:** Across the 4-rung deliberation ladder (`direct` $\to$ `filler` $\to$ `scratchpad` $\to$ `cot`), sub-2B models suffer 100% parse collapse on open-ended `<scratchpad>` reasoning. At 3B parameters (`Llama-3.2-3B`), scratchpad mastery sharply emerges, reaching an internal probe $\text{AUROC} = \mathbf{0.8968}$.
4. **Information-Theoretic Resolution Bottleneck:** Murphy decomposition ($N=10,000$ bootstrap resamples) confirms CoT significantly improves bin Reliability ($\Delta = -0.1618, p = 0.0243$) but yields strictly zero gain in discrimination Resolution ($\Delta = +0.0005, p = 0.8178$).

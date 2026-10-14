# Beyond Accuracy: Calibration, Confidence, and Consistency in Small Language Models

Reproducible evaluation of how chain-of-thought prompting affects accuracy, confidence calibration, and paraphrase consistency across seven instruction-tuned language models (0.5B–8B parameters).

**Research status:** corrected evaluation pipeline for a planned WLLFM 2026 submission. Fresh GPU results and the human paraphrase audit are pending; this repository does not claim acceptance or validated new findings.

## Quick start

```bash
git clone https://github.com/bharathgaddam1712/BeyondAccuracy-CoT-Trustworthiness.git
cd BeyondAccuracy-CoT-Trustworthiness
bash run_everything.sh
```

Requires Linux, Python 3.11/3.12, two suitable NVIDIA GPUs, and approved Hugging Face access to the configured Llama and Gemma models. The default configuration targets two 96 GB Blackwell GPUs.

## Documentation

| Document | Purpose |
| --- | --- |
| [Server commands](SERVER_COMMANDS.md) | Setup, background execution, monitoring, and recovery |
| [Evaluation protocol](PROTOCOL.md) | Metrics, datasets, prompts, pairing, and limitations |
| [Verification](VERIFICATION.md) | Completed local checks and remaining GPU checks |
| [Review corrections](REVIEW_FIXES.md) | Scientific and implementation corrections |
| [File audit](FILE_AUDIT.md) | Compatibility entry points and historical code |

## Reproduction workflow

For the complete ordered server command sequence, see [SERVER_COMMANDS.md](SERVER_COMMANDS.md).

To run the entire workflow from the project folder on the Linux GPU server:

```bash
bash run_everything.sh
```

This handles environment setup, tests, authentication checking, the GPU smoke test, all full experiments, tables, figures, numerical validation and timing. It logs to `logs/` and resumes saved experiment checkpoints when rerun unchanged. Initial model-access approval and the human paraphrase audit still require your input.

This project now has one canonical experiment/analysis pipeline. Original experiment and analysis file paths remain in place and delegate to it. Historical source is retained as text under `legacy_reference/`. Existing generated results and figures have been removed as requested. No new scientific results are included: run the smoke test and full evaluation on the server.

Original outputs are preserved outside this project in `../historical-results/original-review-results-2026-10-12.zip`. Every archived file was checked against its original SHA-256 before clearing the active folders. Use this archive to document the original review discrepancies; never merge its measurements into a corrected run. The old manuscript and data files are historical references, not input to the new results pipeline.

## Server setup

Recommended: Linux, Python 3.11 or 3.12, two RTX PRO 6000 Blackwell GPUs (96 GB each). Your NVIDIA driver reports CUDA 13.3; this is driver capability, not the CUDA runtime embedded in PyTorch. The pinned CUDA 12.8 PyTorch 2.8.0 wheel supports Blackwell. The preflight actually executes a GPU matrix multiplication and checks free VRAM.

Wait for your existing training jobs to finish. One independent worker uses each GPU; weights are not unnecessarily split across cards. Default batch size is 8. No existing server process is killed or reconfigured. An out-of-memory failure preserves completed raw checkpoints; changing batch size requires a new run name because the config is frozen.

From the uploaded project directory:

```bash
bash scripts/setup_blackwell.sh
source .venv/bin/activate
hf auth login
```

Accept model access terms in your HF account for Llama 3.1, Llama 3.2, and Gemma 2. Authenticate using a read token; never paste it into a config or logfile. Metadata and gated config access are checked before the long run. If access fails, fix access and rerun; models are not silently skipped.

Blackwell support reference: https://pytorch.org/blog/pytorch-2-7/
Pinned wheel installation reference: https://pytorch.org/get-started/previous-versions/

## Smoke test on the server

```bash
source .venv/bin/activate
python run_pipeline.py plan --config configs/blackwell_smoke.json
bash scripts/run_smoke.sh
```

The smoke test exercises all seven model loaders, fresh subsets, paraphrase generation/screening, strict parsing, both GPU workers, consistency, trace controls, sampled voting, activation extraction, NF4/FP16, summaries and figures. Small samples may produce uncomputed AUROC/probe comparisons or no eligible shuffled traces. That is reported, not replaced with a fabricated score. The smoke test has its own output directory and cannot contaminate the full experiment.

Run in a persistent terminal (`tmux` if already available). The command prints progress and exits nonzero on a worker failure; do not interpret a partial run as complete. You can capture output using `bash scripts/run_smoke.sh 2>&1 | tee smoke.log`; with a pipeline, enable `set -o pipefail` in your shell to retain the run's failure exit code.

## Full evaluation after the smoke test succeeds

```bash
bash scripts/run_full.sh
```

Outputs live under `results/wllfm2026_full_v3/`. The full default sample is 500 GSM8K, 500 TruthfulQA, 300 BBH, and 200 ARC questions. All seven models use BF16 in the main study. The 8B quantization ablation compares this baseline with FP16 and NF4 on matched subsets. Five paraphrases plus the original are evaluated on GSM8K/TruthfulQA. See `PROTOCOL.md` for every metric, mask, prompt, subset and interpretation restriction.

Resume the same command after an interrupted run. Do not edit code or configuration within a run: source/config mismatches stop resumption. Select a new `run_name` for any changed protocol. JSONL is append-only with unique result keys; a partial final line is detected explicitly rather than silently ignored. Preserve the original file before manually repairing an interrupted write.

Budget approximately 24–72 hours for the full GPU evaluation, or 30–90 minutes for the smoke test after model downloads are cached, with both GPUs available. These are planning estimates, not measured throughput on your server. Downloads, paraphrase screening retries, long generations, other GPU jobs and CPU bootstrap/probe analysis can extend elapsed time. Inspect measured stage throughput after the smoke test:

```bash
python run_pipeline.py timing --config configs/blackwell_smoke.json
```

About 177,200 evaluation responses are scheduled at the maximum trace eligibility, plus paraphrase construction/verification and activation forward passes. Do not estimate the complete run solely by multiplying the tiny smoke sample: model loading and analysis have different fixed costs.

Optional stages after `preflight`/`paraphrases`:

```bash
python run_pipeline.py preflight --config configs/blackwell_full.json --gpus 0 1
python run_pipeline.py paraphrases --config configs/blackwell_full.json --device cuda:0
python run_pipeline.py run --config configs/blackwell_full.json --gpus 0 1 --stages behavioral consistency
python run_pipeline.py run --config configs/blackwell_full.json --gpus 0 1 --stages ablations probes quantization
```

Behavioral calibration must finish before trace ablations. `all` runs dependencies in the correct order. Analysis-only runs:

```bash
python run_pipeline.py analyze --config configs/blackwell_full.json
python run_pipeline.py validate --config configs/blackwell_full.json
```

## Human fidelity check and paper-ready validation

Review `results/wllfm2026_full_v3/paraphrase_human_audit.csv` (100 sampled variants per task). Enter `yes` or `no` in `human_equivalent` and notes. Automated semantic screening alone is not proof of fidelity. Rejected rewrites require a revised paraphrase protocol and fresh affected evaluation, not silently deleting failed questions.

```bash
python run_pipeline.py validate --config configs/blackwell_full.json --require-human-review
```

## Files to use for the revised paper

- `manifest.json`, `hardware.json`, `datasets.json`: exact protocol, revisions, environment and frozen examples.
- `raw/<model>/*.jsonl`: full prompts, generated responses, token IDs, scores and completion diagnostics.
- `paraphrase_generation.jsonl`, `paraphrase_verification.jsonl`, `paraphrases.json`: construction and screening evidence.
- `summary/metrics_by_cell.csv`: all task/model/mode metrics and sample masks/counts.
- `summary/paired_comparisons.csv`: paired accuracy, compliance, ECE, Brier, AUROC, AURC and Murphy component differences, intervals and adjusted tests where applicable.
- `summary/calibration_by_size_and_task.csv`: macro effects, across-model variability and shared-question bootstrap.
- `summary/consistency_by_question.csv`, `consistency_paired.csv`: six-variant pairwise agreement and question-level uncertainty.
- `summary/trace_ablation_comparisons.csv`: common-mask continuation controls and sign-aware CRU.
- `summary/self_consistency_*.csv`: sample-and-vote results and matched comparisons.
- `summary/quantization_paired.csv`: matched BF16/FP16/NF4 comparisons.
- `summary/probes.csv`, `probe_oof_predictions.csv`, `probe_transfer.csv`: token positions, matched probe gaps, sample counts, folds and transfer.
- `figures/*.png` and adjacent `*.csv`: figures and their exact numerical source.
- `validation.json`: completeness, integrity and human-review status.

The old paper sources remain under `paper/` as historical reference. Their numbers and claims must be rewritten from validated new outputs before WLLFM submission. The 10-page IEEE paper should foreground the behavioral results; mechanistic claims remain conditional on the evidence. This code does not optimize for acceptance by selecting favourable runs.

## Local verification

```bash
python -m pytest -q
python run_pipeline.py plan --config configs/blackwell_full.json
```

Tests cover reviewer metric counterexamples, parsing, missingness, ties, consistency definitions, pairing, checkpoint corruption, probe positions, fold construction, and a synthetic raw-to-table/figure integration run. Actual CUDA, remote model loading and bitsandbytes are verified by your server smoke test; local CPU tests do not substitute for that.

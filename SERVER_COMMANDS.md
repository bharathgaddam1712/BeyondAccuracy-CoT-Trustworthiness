# Server commands: corrected WLLFM 2026 evaluation

## Single command for the entire workflow

For a fresh checkout:

```bash
git clone https://github.com/bharathgaddam1712/BeyondAccuracy-CoT-Trustworthiness.git
cd BeyondAccuracy-CoT-Trustworthiness
```

For your existing server folder, use `cd ~/Desktop/bharath/llm-internal-signals-master` instead. Adjust later examples using `$HOME/llm-internal-signals-master` to your actual checkout path.

### Background execution without tmux

Complete setup and authentication interactively, then start the launcher:

```bash
bash scripts/setup_blackwell.sh
source .venv/bin/activate
hf auth login
hf auth whoami
mkdir -p logs
nohup bash run_everything.sh > logs/background.log 2>&1 < /dev/null &
echo $! > logs/launcher.pid
```

Monitor with `tail -f logs/background.log` and `nvidia-smi`. Ctrl+C exits the log viewer. Check the launcher with `ps -p "$(cat logs/launcher.pid)" -o pid,etime,cmd`. After it exits, inspect the final log lines to distinguish completion from failure. Resolve any failure and rerun the same launcher to resume; do not start a second copy while it is active.

Keep an active server experiment on its current code revision. Pulling changed evaluation source into a running or resumable experiment can invalidate its frozen manifest. Use a separate checkout for new revisions.

From the project directory on the Linux GPU server, run:

```bash
bash run_everything.sh
```

This installs the pinned environment, runs CPU tests and dependency checks, checks Hugging Face authentication, runs every GPU smoke-test stage, then runs every full experiment and generates the final tables, figures, validation report and timing summary. It uses GPUs 0 and 1, writes a timestamped log under `logs/`, prevents a second launcher in the same folder, and stops on errors. Rerun the same command to resume completed experiment checkpoints with unchanged code/configuration.

On the first run it prompts for Hugging Face login if no token is configured. Your account must already have access to the Llama/Gemma model repositories. Use an interactive persistent terminal such as tmux so an SSH disconnect does not stop the job. Model access approval and the final human paraphrase fidelity check require your input; all automated experiment and figure generation stages are included in this command.

Final tables: `results/wllfm2026_full_v3/summary/`. Final figures: `results/wllfm2026_full_v3/figures/`. The detailed commands below are optional references for inspection and troubleshooting.

Run these commands on the Linux server with the two RTX PRO 6000 Blackwell GPUs. Copy the corrected project to `~/llm-internal-signals-master` first. If you use a different location, change the first `cd` command. Do not upload `.test-deps`, `.test-tmp`, `.test-mpl`, or a Windows `.venv`.

## 1. Inspect the server

```bash
cd "$HOME/llm-internal-signals-master"
python3 --version
nvidia-smi
df -h .
free -h
```

Use Python 3.11 or 3.12. Leave substantial disk space for Hugging Face model caches, raw responses and activation files; approximately 150 GB free is a practical starting budget, not a measured storage requirement. Wait for the existing training jobs on both GPUs to finish. The pipeline does not stop other processes.

## 2. Install the pinned environment and run CPU tests

```bash
bash scripts/setup_blackwell.sh
source .venv/bin/activate
python -m pip check
python -m pytest -q
```

The setup script installs PyTorch 2.8.0 with CUDA 12.8 wheels, then the pinned project dependencies, and saves `environment.lock.txt`. The CUDA version shown by `nvidia-smi` is the driver's capability; it need not equal the wheel runtime. If `python3 -m venv` is unavailable, have your server administrator enable the venv package for your Python version before continuing.

## 3. Authenticate for gated models

```bash
hf auth login
hf auth whoami
```

Use your own Hugging Face account and read token. Accept access terms for `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, and `google/gemma-2-2b-it` on Hugging Face. Do not put the token in a script, config or shared log. A login alone does not grant gated-model access.

## 4. Keep the run alive across SSH disconnects

If `tmux` is already available:

```bash
tmux new -s wllfm
cd "$HOME/llm-internal-signals-master"
source .venv/bin/activate
set -o pipefail
mkdir -p logs
```

Detach using **Ctrl+B, then D**. Reconnect using:

```bash
tmux attach -t wllfm
```

Without `tmux`, use another persistent terminal supported by your server. Keep `set -o pipefail` enabled whenever using `tee`; otherwise a failed Python process can appear to succeed because `tee` succeeded.

## 5. Run the smoke test first

```bash
cd "$HOME/llm-internal-signals-master"
source .venv/bin/activate
set -o pipefail
mkdir -p logs
python run_pipeline.py plan --config configs/blackwell_smoke.json --gpus 0 1
bash scripts/run_smoke.sh 2>&1 | tee -a logs/smoke.log
```

This command performs preflight, freezes the small evaluation subsets, constructs and verifies paraphrases, runs every experiment stage, and produces summaries/figures. Preflight runs a real CUDA kernel and checks GPU memory. The first execution downloads models and datasets; allow extra time. The two output roots are separate:

- Smoke: `results/wllfm2026_smoke_v3/`
- Full: `results/wllfm2026_full_v3/`

After the smoke command exits successfully:

```bash
python run_pipeline.py validate --config configs/blackwell_smoke.json
python run_pipeline.py timing --config configs/blackwell_smoke.json
cat results/wllfm2026_smoke_v3/validation.json
```

Small smoke samples cannot establish significant scientific results. Some probe/AUROC estimates will be unavailable because a class is missing or too small. Human paraphrase review is initially pending; numerical validation may pass while `paper_ready` remains false. Resolve any actual error before starting the full run. Preserve the smoke log if troubleshooting is needed.

## 6. Run the full experiment

```bash
python run_pipeline.py plan --config configs/blackwell_full.json --gpus 0 1
bash scripts/run_full.sh 2>&1 | tee -a logs/full.log
```

Plan for roughly **24–72 hours of full GPU evaluation**, or **30–90 minutes for the smoke test after downloads are cached**, with both GPUs available. These are provisional estimates, not benchmark measurements on this server. Paraphrase retries, long responses, other GPU jobs and CPU statistical/probe analysis can extend elapsed time.

To inspect saved throughput in another SSH terminal while the full run proceeds:

```bash
cd "$HOME/llm-internal-signals-master"
source .venv/bin/activate
python run_pipeline.py timing --config configs/blackwell_full.json
tail -n 40 logs/full.log
nvidia-smi
```

## 7. Resume an interrupted run

With the same code and unchanged configuration, rerun the same command:

```bash
cd "$HOME/llm-internal-signals-master"
source .venv/bin/activate
set -o pipefail
bash scripts/run_full.sh 2>&1 | tee -a logs/full.log
```

Completed records are checkpointed and skipped on resume. Do not edit configuration or source files within a run, copy old outputs into the new output root, or delete individual failed examples. Code/config/data mismatches deliberately stop resumption. If a write was interrupted mid-line, the pipeline reports the corrupt JSONL line; preserve the file and investigate before repairing it. A changed protocol requires a new `run_name` and a fresh evaluation.

## 8. Validate all outputs

```bash
python run_pipeline.py validate --config configs/blackwell_full.json
cat results/wllfm2026_full_v3/validation.json
```

Review `results/wllfm2026_full_v3/paraphrase_human_audit.csv`: enter `yes` or `no` in `human_equivalent` and explanatory notes. The audit has 100 sampled variants per consistency dataset. Rejecting a variant requires investigating and revising the paraphrase protocol, followed by a fresh affected evaluation; do not silently discard it.

Once the fidelity audit is complete:

```bash
python run_pipeline.py validate --config configs/blackwell_full.json --require-human-review
```

Use only a complete run with successful validation and `paper_ready: true` to populate the revised manuscript. Successful validation does not guarantee that the scientific findings are favourable or that the paper will be accepted.

## 9. Regenerate tables and figures from the same saved raw responses

```bash
python run_pipeline.py analyze --config configs/blackwell_full.json --require-human-review
```

This is CPU analysis; it does not rerun model generation. Tables are under `results/wllfm2026_full_v3/summary/` and figures with adjacent source CSVs under `results/wllfm2026_full_v3/figures/`. Full generated text is stored under `raw/`. Preserve the manifest, frozen datasets, paraphrase evidence, hardware report, validation, environment lock and logs with the results.

## 10. Optional: run stages individually

Use this sequence only if you prefer separate stage commands. The `all` command in the normal smoke/full scripts already handles this order.

```bash
python run_pipeline.py preflight --config configs/blackwell_full.json --gpus 0 1
python run_pipeline.py paraphrases --config configs/blackwell_full.json --device cuda:0
python run_pipeline.py run --config configs/blackwell_full.json --gpus 0 1 --stages behavioral consistency
python run_pipeline.py run --config configs/blackwell_full.json --gpus 0 1 --stages ablations probes quantization
python run_pipeline.py validate --config configs/blackwell_full.json
```

Behavioral calibration must finish before trace ablations. Running only some stages intentionally leaves a partial experiment; the final validation must check all stages. See `PROTOCOL.md` for the exact experiment definitions and `REVIEW_FIXES.md` for the changes addressing the reviewers.

## Historical results

The original outputs are preserved outside the corrected project in `../historical-results/original-review-results-2026-10-12.zip`. All 90 archived files were verified against their original SHA-256. They support the historical audit only; the corrected experiment starts from scratch and never imports these measurements.

# File audit and replacement map

Every former Python entry point was read and statically inspected for scoring, masks, prompts, inference, storage and failure handling. Original experiment and analysis paths remain in place as functional entry points into the corrected shared pipeline. Their full source is retained under `legacy_reference/` as `.txt` files. New canonical modules live in `src/`.

| Original file | Lines inspected | Findings / risk | Disposition |
|---|---:|---|---|
| `analysis/analyze_final_results.py` | 80 | hard-coded regime interpretation | Replaced by canonical entry point; historical source retained as text |
| `analysis/consistency_audit.py` | 197 | hard-coded regime interpretation | Replaced by canonical entry point; historical source retained as text |
| `analysis/deep_rigorous_analysis.py` | 174 | duplicated metric implementation; hard-coded regime interpretation | Replaced by canonical entry point; historical source retained as text |
| `analysis/e10_scaling_law.py` | 279 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `analysis/e11_brier_decomposition.py` | 233 | separate confidence parser | Replaced by canonical entry point; historical source retained as text |
| `analysis/e1_analysis.py` | 224 | duplicated metric implementation; separate confidence parser | Replaced by canonical entry point; historical source retained as text |
| `analysis/e2_trace_quality_analysis.py` | 83 | legacy data/analysis interface; incompatible run provenance | Replaced by canonical entry point; historical source retained as text |
| `analysis/e3_analysis.py` | 269 | legacy data/analysis interface; incompatible run provenance | Replaced by canonical entry point; historical source retained as text |
| `analysis/e4_analysis.py` | 274 | duplicated metric implementation | Replaced by canonical entry point; historical source retained as text |
| `analysis/e4_consistency_eval.py` | 174 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `analysis/e7_e8_analysis.py` | 105 | legacy data/analysis interface; incompatible run provenance | Replaced by canonical entry point; historical source retained as text |
| `analysis/make_tables.py` | 145 | legacy data/analysis interface; incompatible run provenance | Replaced by canonical entry point; historical source retained as text |
| `analysis/matched_subset_calibration.py` | 152 | legacy data/analysis interface; incompatible run provenance | Replaced by canonical entry point; historical source retained as text |
| `analysis/merge_and_analyze_all_diagnostics.py` | 166 | duplicated bootstrap implementation; duplicated metric implementation | Replaced by canonical entry point; historical source retained as text |
| `analysis/statistical_tests_and_metrics.py` | 239 | duplicated bootstrap implementation; duplicated metric implementation | Replaced by canonical entry point; historical source retained as text |
| `experiments/internal_signals/extract_internal_signals.py` | 354 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/internal_signals/extract_v2.py` | 380 | separate confidence parser; broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/internal_signals/probe_analysis.py` | 402 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/internal_signals/probe_v2.py` | 587 | incorrect handling of AUROC ties; broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/kaggle/kaggle_consistency_pipeline.py` | 219 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/kaggle/kaggle_e7_e8_pipeline.py` | 607 | response truncation; duplicated metric implementation; separate confidence parser; broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/kaggle/kaggle_gated_e1_e3_e4_pipeline.py` | 536 | separate confidence parser; broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/scaling_and_diagnostics/e5_self_consistency_calibration.py` | 218 | duplicated metric implementation | Replaced by canonical entry point; historical source retained as text |
| `experiments/scaling_and_diagnostics/e9_shuffled_cot.py` | 221 | substring correctness; response truncation; duplicated metric implementation; separate confidence parser; broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/scaling_and_diagnostics/overnight_consistency_master.py` | 336 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `experiments/scaling_and_diagnostics/overnight_gpu_run.py` | 459 | separate confidence parser; broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `src/datasets.py` | 162 | broad exception handling / potential dropped rows | Replaced by canonical entry point; historical source retained as text |
| `src/session_init.py` | 66 | restores stale outputs | Replaced by canonical entry point; historical source retained as text |
| `src/session_save.py` | 33 | legacy data/analysis interface; incompatible run provenance | Replaced by canonical entry point; historical source retained as text |

Core replacements: `src/metrics.py`, `src/inference.py`, `src/models.py`, `src/prompts.py`, `src/protocol.py`, `src/runner.py`, `src/statistics.py`, `src/probing.py`, `src/analysis.py`. Historical paper sources are retained but must not be submitted with their old numeric claims.

# Review corrections implemented in protocol v3

The previous document made claims from unvalidated CSVs. It is retained only as historical text in `legacy_reference/`.

| Reviewer concern | Correction | Canonical implementation |
|---|---|---|
| ECE incompatible with Brier/mean confidence | Include p=1; assert full bin assignment and metric inequalities | `src/metrics.py` |
| Prompt conflict and response truncation | Consistent explicit fields; store full strings/IDs and EOS/cap diagnostics | `src/prompts.py`, `src/inference.py`, `src/runner.py` |
| Confidence/answer parser contamination | Strict separate fields; no arbitrary-number confidence fallback | `src/inference.py` |
| Accuracy scoring / A–D restriction | Decimal-preserving numeric equality; full offered letter range; no substring credit | `src/inference.py` |
| Quantization confound | Uniform BF16 main runs; matched FP16 and NF4 ablations | `src/models.py`, `src/runner.py` |
| Unclear confidence protocol / BBH exclusion | Confidence on a separate line on every benchmark; several calibration metrics/binnings | `PROTOCOL.md` |
| Inconsistent consistency definition | All-pairs agreement; no singleton or missing-parse agreement | `src/metrics.py` |
| Incomplete paraphrase cache | Exactly five screened variants or stop; choices unchanged; human fidelity sample | `src/protocol.py` |
| CRU sign problem | Ratio uninterpreted without a positive substantive calibration gain; raw comparisons retained | `src/analysis.py` |
| Shuffled figure/table disagree | Same continuation arms, matched masks and saved plot source CSVs | `src/analysis.py` |
| Invalid shuffled inference | Actual question/choices; remove final fields; equivalent continuation prompts | `src/runner.py` |
| Wrong target of significance test | Recompute nonlinear metrics on paired resamples; actual metric randomization tests | `src/statistics.py` |
| Dependent pooled bootstrap | Per-question pairing; shared question draw across models for macro calibration effects | `src/analysis.py` |
| Brier decomposition overclaim | Exact distinct-forecast decomposition; raw Brier retained; binned residual reported | `src/metrics.py` |
| Probe comparison / token positions / folds | Single unpadded IDs; pre-answer primary; fold-local scaling; matched gaps/counts/OOF scores | `src/probing.py` |
| AUROC confidence ties | sklearn half-credit tie handling throughout | `src/metrics.py`, `src/probing.py` |
| Reproducibility / inaccessible implementation | One CLI, versioned configs, source hash, HF revisions, frozen subsets, complete raw outputs | `run_pipeline.py`, `src/protocol.py` |
| Old duplicate pipelines can reintroduce bugs | Original entry-point paths retained; shared implementation replaces their duplicate logic | `FILE_AUDIT.md` |

Presentation, related work, limitation implications and IEEE paper formatting are the later manuscript task. No old scientific conclusion is treated as confirmed by these code changes. See PROTOCOL.md for limits of probes, trace interventions, confidence parsing and paraphrase screening.

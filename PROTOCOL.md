# Corrected evaluation protocol (v3)

This is a new evaluation, not a reconstruction of the rejected manuscript's numbers. Set the protocol before running; no data-dependent edits to models, subsets, prompts, binning or claims are permitted within a run. An ordinary JSON config is a frozen protocol, not evidence of formal public preregistration. Record a public timestamp separately if claiming preregistration.

## Main study

Seven original instruction-tuned models, all in BF16 by default, including the 8B model. BF16 is floating-point inference, not quantization. Choose FP16 in a new config/run if desired; never mix precisions within the main comparison. The secondary 8B experiment compares BF16, FP16 and NF4 with identical prompts and matched questions.

Fresh test subsets: 500 GSM8K, 500 TruthfulQA MC1, 100 each of three BBH subtasks, and 200 ARC-Challenge. ARC is additional evidence compared with the original three-benchmark paper. All sources, revisions, row indices, shuffled choice orders and correct labels are frozen in the run directory. BBH confidence elicitation is included; short final answers do not inherently prevent asking for confidence on a separate line.

Accuracy and confidence elicitation are distinct axes. Direct and CoT share answer formatting within each axis. Every model uses exactly the same items and prompts, except Gemma's required folding of a system message into user text. Confidence prompts request Answer first, then Confidence on a separate line. Greedy decoding has the same 2,048-token upper bound for both modes. Token-cap and EOS rates are reported. A token budget is not a guarantee that all responses finish; inspect these diagnostics before interpreting results. Inputs are never silently truncated.

Strict parsers accept explicit answer fields or a complete bare short answer; they never mine a confidence number as an answer. Decimal points and minus signs are preserved; numeric equivalence uses Decimal. MCQ letters include all offered options. Confidence is an explicitly labelled percentage; invalid and missing confidence remains missing. Answer parse failures count as incorrect for operational accuracy. Report parse coverage alongside it.

## Calibration and uncertainty

ECE uses 10 equal-width bins [0,.1), ..., [.9,1], each valid example assigned exactly once. Sensitivity outputs use 5, 15 and 20 width bins and quantile bins (ties may reduce the number of quantile bins). Brier is mean (p-y)^2. AUROC is undefined for a one-class sample, not 0.5; tied scores receive the standard half-credit through sklearn. AURC averages over random permutations within score ties.

Every calibration row reports sample size, mean confidence, correctness accuracy on precisely that same sample, ECE and Brier. Integrity checks verify ECE >= abs(mean confidence - accuracy) and Brier <= (1-accuracy)+(1-mean confidence). Per-condition tables use that condition's parsed-confidence mask. Paired comparisons explicitly use the intersection of questions with confidence in BOTH conditions. Accuracy and compliance comparisons use all paired questions, independently of confidence parsing. Missing-confidence selection remains a limitation; a matched intersection does not recover the missing predictions.

Paired percentile bootstrap recomputes each nonlinear metric on resampled questions. Randomization tests swap conditions within each question; they do not test a mean absolute-error surrogate and call it an ECE test. Holm adjustment is applied within the documented axis/metric/comparison family across model-task cells. AUROC CIs report valid resample counts. Global size/task ECE averages are macro averages over the fixed evaluated models, with the same question draw applied across models; uncertainty describes the finite question sample, not a population of unseen model families. Across-model SD and ranges are separately reported. No shared-threshold conclusion is hard-coded.

Murphy decomposition groups identical confidence values to obtain an exact identity. A coarse 10-bin version reports the residual between raw Brier and the binned expression. Reliability/resolution changes are descriptive: accuracy and base-rate changes can influence them. A nonsignificant resolution change is not proof of no gain or a formatting-only mechanism.

## Paraphrase agreement

GSM8K and TruthfulQA each use the original plus FIVE distinct rewrites. MCQ choices and order remain untouched. Generation uses Qwen2.5-7B, absent from the evaluated registry; Llama-3.1-8B screens bidirectional equivalence. The verifier overlaps the evaluated family; automated screening is not a ground-truth guarantee. Numbers must match exactly, duplicate/original text is rejected, and questions with incomplete variants stop the consistency stage. Generation and verification text are saved.

For each question with K=6 variants:

`agreement(q) = sum(i<j, parsed_i AND parsed_j AND answer_i==answer_j) / choose(K,2)`.

Missing/failed parses never agree. A singleton is undefined and is never included in the production study. A parsed-only sensitivity value and parse count are also stored. Consistency delta is mean agreement(CoT)-mean agreement(Direct) on matching questions. Positive values mean more answer agreement. This measures answers, not identical reasoning or correctness. Question-level paired bootstrap preserves dependence among the 15 pairs.

The reproducible human audit samples 100 variants per task. Fill `human_equivalent` with `yes` or `no` and notes. Paper-ready validation requires complete review and zero rejected variants; a rejection requires fixing the paraphrase protocol and rerunning affected evaluation under a new run. Do not silently drop difficult questions. The primary sample is evaluated after automated screening; report human fidelity findings explicitly.

## Trace interventions and sample-and-vote

Fixed hash-selected secondary question subsets, balanced across BBH subtasks, are selected without looking at outcomes. Eligibility requires a completed source CoT trace with an explicit final field and a genuinely changed step permutation. The final answer/confidence fields are removed. The exact original question and MCQ choices are supplied in every continuation prompt. Four arms use the SAME continuation instructions: empty text, original structured trace, inert token-matched filler, and token-matched shuffled trace. Original and shuffled traces have equal standalone token counts; full prompt token counts are recorded separately, so boundary-token differences are visible. Eligibility reports include ALL preselected questions and exclusion reasons. Conclusions apply to eligible traces. These interventions probe the use of supplied traces, not a direct causal intervention on the model's original hidden reasoning.

Accuracy, ECE and paired comparisons use the same strict scoring. No substring credit, missing-confidence imputation, or answer-key input is allowed. The matched intervention table and plotted bars use the same four-way confidence intersection, with source CSVs beside every plot. CRU is descriptive and uninterpreted when structured CoT does not improve over empty continuation or the improvement is below .01. A positive ratio alone never implies reasoning is required.

Sample-and-vote uses five CoT draws at T=.7, top-p=.95 with fixed per-item seeds. Confidence is the winning answer's vote share; failed parses occupy the denominator. No valid answer means abstention. Vote confidence and greedy verbalized confidence are compared on matching questions but are predictions from different inference procedures with different compute cost.

## Probes

Separate confidence-elicitation trajectories for direct, CoT and scratchpad modes, all four datasets, five models. This deliberately replaces the previous four-mode ladder and its unsupported regime interpretation. Selected layers are 25%, 50%, 70%, 90%, and 100% of transformer depth; 70% is the preselected primary layer, not the best layer found after evaluation.

Extraction uses original generated token IDs and a SINGLE unpadded sequence with all-ones attention mask. Recorded positions: prompt end, immediately before explicit answer content, end of answer content, and sequence end. The primary pre-answer position precedes the final answer and confidence fields; prompt-end is a pre-generation control. Answer-end and sequence-end are exploratory; the latter may encode verbalized confidence. No batch padding positions are sampled. Actual absolute indices are stored in each NPZ. Absent fields are absent features, never replaced with another position. Missing explicit fields condition eligibility and must be reported.

Each model/mode/position contains one row per original question. StandardScaler is fit inside each training fold, followed by balanced logistic regression (C=.5), five-fold stratified CV. Fold IDs and out-of-fold predictions are saved. No layer maximization is used for the primary claim. Probe-vs-confidence comparisons use identical questions and outcomes; both matched AUROCs, paired gap CIs and positive/negative counts are saved. Bootstrap intervals are CONDITIONAL on fitted OOF predictions: they do not include full model-training variability. Probes are supervised; verbalized confidence is zero-shot. Neither decodability nor an AUROC gap proves metacognitive knowledge. Cross-dataset transfer trains solely on source questions and tests solely on target questions. Insufficient class counts yield an explicit status, not a made-up AUROC.

## Reproducibility boundaries

Run directories carry config, source hash, exact HF revisions, dataset hash, package versions, hardware data, exact messages, generated IDs, full output strings, and append-only JSONL checkpoints. Resuming with changed source/config/subset fails rather than mixing protocols. Sampled generation is per item so checkpoints and batch size do not shift its random stream. Greedy GPU computations can still vary across hardware/software and batch composition; record environments, and do not claim bitwise reproducibility across devices.

All scientific interpretation follows the new data. Historical LaTeX and reference code are retained for comparison; they are not evidence that old claims survived the rerun.

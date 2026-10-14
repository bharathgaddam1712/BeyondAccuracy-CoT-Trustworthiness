# Local verification, 12 October 2026

The single-command launcher `run_everything.sh` also passed Bash syntax checking and two mocked integration tests: complete setup-to-final-output orchestration, and a smoke-test failure that preserves exit code 23 and prevents the full run. These tests exercise the actual launcher, replacing external programs with local stubs. GPU execution remains unverified locally.

The corrected pipeline passed all 39 CPU tests in 6.52 seconds. This includes a regression test checking that an incomplete generation batch cannot save misaligned responses. Syntax checks passed for all 46 Python files, and every shell script has Linux-compatible LF line endings. The verification result is recorded in `verification_report.json`.

Checks exercise ECE bin endpoints including confidence exactly one, the reviewers' metric inequalities, Brier identities, tied AUROC/AURC, strict answer/confidence parsing, question pairing, missingness, pairwise consistency, CRU sign interpretation, corrupted checkpoints, actual out-of-fold probe predictions, token positions, and synthetic raw-response-to-table/figure generation. Mock generation tests verify left-padding removal, per-sequence EOS handling and rejection of context overflow.

Both full and smoke configurations load successfully and their CLI plans complete. These checks do not download models or generate scientific observations.

The local test environment uses Python 3.12.14 with project-local CPU test dependencies. The server's pinned dependency environment is installed by `scripts/setup_blackwell.sh`. Actual CUDA kernels, seven model downloads/loaders, BF16/NF4 execution, model-specific chat templates and real paraphrase screening have not been executed on the user's server. Run the server smoke test before the full experiment; no local test can guarantee freedom from hardware, dependency or model-output failures.

Original generated outputs are archived separately under `../historical-results/`. The active pipeline begins a fresh evaluation and does not import the archived measurements. Only newly generated, validated outputs may populate the revised paper.

Do not upload `.test-deps`, `.test-tmp`, `.test-mpl` or a Windows virtual environment to the Linux server. They are local verification artifacts, not runtime dependencies.

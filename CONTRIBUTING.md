# Contributing

Use the canonical implementation in `src/` and `run_pipeline.py`. Existing experiment and analysis entry points delegate to that implementation.

Before proposing a change, run `python -m pytest -q` in the configured environment. Shell orchestration tests use stubs and require Bash; they do not run models. Follow `SERVER_COMMANDS.md` for GPU validation.

Changes to prompts, parsing, metrics, sampling, or generation settings require a new run name. Preserve the previous manifest and raw records. Include the relevant regression test and explain how the change affects reported results.

Never commit tokens, model weights, generated experiment outputs, virtual environments, or machine-specific dependency directories. Report the configuration, commit, validation status, and sample counts when sharing results. Do not present synthetic test fixtures as experimental evidence.

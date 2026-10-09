# CZSC operations

Use the configured .venv-win and app working directory. Analyze with --symbol and optional --start/--end/--as-of. start clips the visible range; prior raw history remains warmup. Scan uses --stocks; omit to select all local symbols, and disclose failures. Analyze/scan/backtest default --start to 2026-01-01 and --end to latest local data; explicit historical dates remain supported. Batch commands accept --device, --cpu-workers and --batch-size; all generated results carry current data/config/source identity and actual compute provenance.

Update first when requested: python -m my_strategy.cli update-data --mode local. Verify actual target-day raw coverage, not only exit status. Preserve source failure details; do not present incomplete dates as complete.

Backtests persist daily.csv, ledger.csv, rejections.csv and RunContext metadata. Actual execution is next-open long-only main-board, 100-share lots, T+1, costs and prior-volume participation. Unknown historical ST/limit prices use a conservative 5% guard, not a claim of official exchange limits. Unadjusted data does not account for dividends or split holdings. Multi-stock totals use fixed equal accounts and keep failed-account cash.

Unified research is the default for analyze/scan/backtest/backtest-batch. Examples:

```
python -m my_strategy.cli analyze --symbol 600027.SH --end 2026-09-30 --usage-mode historical --model-policy auto --device cuda:1
python -m my_strategy.cli scan --end 2026-09-30 --device cuda:1 --cpu-workers 4
python -m my_strategy.cli backtest --stocks 600027.SH --start 2026-01-01 --end 2026-09-30 --usage-mode historical --model-policy auto --device cuda:1
python -m my_strategy.cli research-status --json
```

Use --structure-only for native structure baseline. --model-run-id with --model-fold selects a pinned checkpoint; its data availability must precede the applicable signal dates. Retrospective analysis requires --usage-mode retrospective --model-policy pinned --model-run-id RUN --model-fold production and never filters entry. It is rejected for scan/backtest. Automatic historical routing preserves one Position prefix across checkpoint boundaries; cash accounts begin at the requested backtest start and execute at next verified openings. Chart display start does not discard warmup or redefine personal holdings.

Only certified models can be published with `research-promote --model-run-id RUN --checkpoint production --reason TEXT`. Its immutable `reports/production_certification.json` must bind actual completed training, full-universe reports, independent execution audit and at least three untouched windows frozen in a separate RunContext before evaluation. `research-rollback --reason TEXT` withdraws the alias; supplying --release-id restores a verified prior release. Current release dates never become historical data-availability dates. Models that fail gates stay shadow, with probabilities separately displayed and combination rules driving intents. Exits remain rule-based.

Executable-entry options: `--entry-policy legacy` preserves the original prefix/target execution; `fresh` starts a flat research Position at the selected start and only executes a new valid plan on the next verified session; `risk` additionally tests fixed MA20/ATR price limits, point confirmation age, frozen structure invalidation and risk sizing. Risk is an unvalidated historical experiment and cannot run with production mode. Earlier feature warmup remains intact. Changing the start therefore changes the new-policy account history; chart zoom alone does not. Old saved requests without a policy restore legacy.

For the actual-exit-label development experiment, run `python -m my_strategy.scripts.study_czsc_executable_entries --parent-run-id FROZEN_RESEARCH_RUN --device cuda:1 --workers 8`. This verifies all parent hashes/raw inputs, builds actual full-roundtrip labels once per account, purges immature labels at training cutoffs, trains CUDA candidates and evaluates legacy/fresh/risk/fresh_ml on the complete frozen mainboard pool. It writes its own RunContext and preserves rejected and missing accounts. Its models are isolated from automatic routing and promotion. Previously inspected quarters and short windows relative to the 120-bar holding dependence cannot supply independent certification.

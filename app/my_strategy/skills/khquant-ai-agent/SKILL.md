---
name: khquant-ai-agent
description: "Operate, inspect, update, test, and maintain KHQuant's CZSC structure workbench, SQLite raw prices, chronological analysis, BrokerSimulator backtests, six-view dashboard, saved personal data, results, and audit records."
---

# KHQuant AI Agent

The current architecture is CZSC-only as authorized on 2026-10-01. Old V2, chip, Z1, ML training, registry, evidence, dashboard and task pipelines are retired. Historical notes do not authorize restoring them.

The user's additional 2026-10-01 authorization adds an independent CZSC/MA-volume-price/PyTorch research layer, not the retired ML pipeline. Use services/czsc_research*.py, configs/czsc_research.json and research-train; freeze the current universe and share the native Position/Broker execution contract. Models remain shadow until the complete-universe chronological ledger gate passes. Classifier probability estimates a fixed-horizon feasible net-positive label and only filters entries; exits remain rule-based. The deterministic four evidence roles are distinct from actual LLM review.

The executable-entry research added on 2026-10-02 has `legacy`, `fresh`, and experimental `risk` policies. CLI/API default to legacy for compatibility; new dashboard requests default to fresh. Fresh/risk Position starts flat at the selected start while keeping earlier feature warmup. A new point creates a single next-session plan; HOLD, rejected plans and repeated identities do not create later buys. Actual account stops use filled cost and planned risk exits retain pending sales until all shares close. Risk is rejected in production mode. Old fixed-10-day models stay shadow for fresh/risk because their exit contract differs. `czsc_strategy_ledger_v1` labels mature only at actual full exit, preserve unfilled/unclosed rows, and are trained in an isolated development study; these candidates are not production releases.

## Locate and verify

Find the app root containing AGENTS.md and my_strategy. Use Python 3.12 with repository .env KHQUANT_VENV_DIR, normally .venv-win. Do not infer the runtime from a historical absolute interpreter path. Read AGENTS.md and relevant source/config/tests; start and finish ai_change_logger.py for modifications. Preserve unrelated changes and append-only audit.

Use scripts/verify_khquant_workspace.py for live workspace and raw-coverage inspection; prefer python -m my_strategy.cli doctor --check-write --strict --json for runtime verification. Verify actual input dates before reporting results.

## Route

- Raw input, update, analyze, scan, backtest: references/operations.md
- Paths and reproducible outputs: references/project-map.md
- Six-view interface and browser validation: references/dashboard.md
- Chronological structure/intent/execution checks: references/governance.md
- Exact source/data retirement: references/maintenance.md
- Chart periods, layers and replay: references/charting.md
- Packaging current CZSC state: references/migration.md

## Boundaries

Use actual local raw OHLCV without replacing historical bars with later quotes. as_of truncates input before calculation. Structure anchors and confirmation times differ. Unfinished structures may change. Higher periods close only once the next bucket is observed when no verified calendar exists.

Close-derived intentions execute through BrokerSimulator at the next observable opening. Persist Ledger.to_frame(), daily equity, fees, rejections and open positions with RunContext. Multi-stock accounts divide capital equally without cash reallocation. Report missing stocks and all execution limitations.

Source integration must validate czsc-source-manifest.json against the frozen attachment. Never silently substitute a PyPI version that only shares its version number. Do not modify upstream algorithms under the original hash.

Cleanup requires exact absolute targets and boundary checks, closing SQLite first. User-authorized retirement can remove old implementations and derived results without compatibility copies. Preserve reusable raw prices and historical AI audit. Do not introduce old-result readers or model-registry dependencies.

## Commands and completion

From app root: python -m my_strategy.cli analyze/scan/backtest/backtest-batch/update-data/dashboard/doctor/verify-db/package. Check command help for flags. Report actual input end/coverage, run IDs, real artifacts, verification, limitations and audit session. Running processes alone are not proof of completion.

Research commands: research-train --end YYYY-MM-DD --calendar-run-id VERIFIED_RUN --device cuda:1 --cpu-workers 8; scan/backtest --research --model-run-id TRAIN_RUN. Verify actual target-date coverage, label exclusions and frozen-budget accounts. Training, scan and research ledgers require the independently verified calendar, source allowlist and quality window; missing expected session bars are rejected rather than silently delayed. A model cannot precede its available_at. Preserve validation-only calibration, training-only preprocessing, semantic schema binding and immutable dataset/checkpoint hashes.

Analysis, scan and backtest CLI/UI now default to the unified research strategy; use --structure-only for the native baseline. Default --usage-mode historical --model-policy auto routes compatible dated checkpoints continuously, with rules when unavailable. Production filtering requires an actual dated release; a production-named checkpoint is only a final candidate. Historical reconstruction cannot borrow later global gates; retrospective use is explicit pinned analysis only and stays shadow. research-status/promote/rollback use append-only release events and an atomic alias in the existing CZSC results database. Promotion requires complete-universe ledger gates plus independent windows frozen before evaluation, real training/certification/release times, and bound artifact hashes. Known development windows never certify a release. See references/operations.md and the current unified research ADR.

Batch compute may use optional pinned PyTorch/CUDA (`install.py --gpu --device auto`). CPU preparation preserves frozen chronological CZSC; parent-only CUDA price/warmup tensors must feed native Position. Record actual CUDA work and fallback reasons separately from hardware availability. Explicit CUDA requests must fail rather than silently use CPU. Charts and batch requests default to 2026-01-01 through latest local market data, retaining pre-start warmup; buy/sell color and shape distinguish signal intention from actual ledger execution.

# Current project map

Git root contains install.py and .env using KHQUANT_ENV_SCHEMA=2; app root contains AGENTS.md and my_strategy. Business paths resolve through core.paths after runtime_env normalization. Config uses core.config_loader.

Input: KHQUANT_RAW_DB, stock_daily_normalized unadjusted daily prices. CZSC adapter validates schema, duplicates, ordering, OHLC, volume and as_of availability. No processed warehouse dependency.

Current pipeline: adapters/czsc_adapter.py → services/czsc_analysis.py → services/czsc_backtest.py → BrokerSimulator/Ledger. web_dashboard/api.py and tasks.py serve six views. storage/czsc_results.py indexes new reports at processed/czsc/results.db, relative to artifacts/runs/<id>. Task state lives in metadata/czsc_tasks.db. storage/personal_portfolio.py stores the watchlist and manual holdings in metadata/personal_portfolio.db.

Original source: app/vendor/czsc with per-file hashes in app/czsc-source-manifest.json. Local chart asset: web_dashboard/static/vendor/lightweight-charts. Historical knowledge_base is audit only, not current runtime policy.

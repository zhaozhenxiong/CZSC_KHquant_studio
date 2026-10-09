# Four-view dashboard

Default http://127.0.0.1:8124. Start python -m my_strategy.cli dashboard; use --port for an isolated QA server. Six views: structure analysis, market scan, strategy backtest, watchlist, manual holdings, data/tasks. Endpoints /api/analysis, /api/symbols, /api/data/status, /api/tasks, /api/runs, /api/watchlist and /api/holdings. Old routes return 404.

Watchlist stars and the watchlist page share persistent storage. Import legacy browser favorites additively and only clear them after saving succeeds. Holdings are manually entered positive integer shares and finite positive average costs, keyed by symbol. Show the actual local close and its date, missing-quote coverage, cost, market value and unrealized PnL. Do not create example positions in the user's database or mix these records with backtest ledgers. Validate CRUD, reload and server restart persistence in isolated QA storage. Check for running tasks before restarting the main server.

Use actual backend JSON for all chart/table data. Validate period switches, layer toggles, as_of replay, market scan filtering, task cancellation/failure, net-value/drawdown chart and Ledger trade location. Use a browser screenshot for visual evidence, check desktop/mobile layouts and console errors. Reload after source changes without hot reload.

Non-loopback binding requires KHQUANT_API_TOKEN; same-origin actions and optional Bearer token are enforced. Browser token resides in sessionStorage khquant_api_token; no endpoint returns it. Scripts and chart library are local assets.

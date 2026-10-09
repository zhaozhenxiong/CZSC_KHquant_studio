# Current CZSC package

Package current code, frozen vendor source/wheel, reusable raw DB, CZSC result/task databases, saved watchlist/manual holdings, and only new CZSC RunContext artifacts. SQLite copies use the backup API; do not copy active WAL files as a database snapshot. Exclude secrets, .scratch, virtual environments, caches, old models and old derived warehouses.

Run `python -m my_strategy.cli package --target ../czsc-package --include-data --include-artifacts all`. At the destination use `python install.py --upgrade` to install the current frozen runtime.

Result report index paths are relative to runs_root. Packages install a fresh environment; never copy a virtual environment between computers. Preserve source hashes and validate the package manifest. No old-result rebinding/compatibility is part of this package.

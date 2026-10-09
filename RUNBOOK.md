# KHQuant CZSC portable package

Run `python install.py` from this directory. Use `python -m my_strategy.cli package --verify .` from this directory to verify the payload. Runtime paths resolve from `app`; the raw market database is an input. Code-only is the default. Data and model/report/calendar state are optional; saved watchlist/holdings require the separate --include-personal export option. Run `python install.py --device auto` to select available CPU, CUDA or Apple MPS support.

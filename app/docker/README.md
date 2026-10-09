# CZSC container

Current image builds frozen attachment source with Rust and installs the CZSC-only runtime. Run `docker compose build` and `docker compose run --rm khquant-cli doctor --check-write --strict --json` from app root. Mount raw input and current CZSC outputs separately. Old GPU/model/registry/toolbox images and compose variants are retired. The native Windows runtime is the locally validated deployment; Docker and other platform builds require their own execution validation.

# Frozen CZSC attachment

`czsc/` contains the runtime/build sources from the user-provided
`czsc-master.zip` (CZSC 1.0.1). The original SHA256 and each retained source
SHA256 are recorded in `../czsc-source-manifest.json`. Apache-2.0 LICENSE and
upstream declarations are retained. Cargo metadata separately declares MIT.
No analysis algorithm has been patched. Guides, prompts, development caches,
and upstream examples are excluded from this runtime distribution.

The published PyPI 1.0.1 source has different `czsc-core` analysis code.
Installing its identically named wheel is therefore not an equivalent build.
`wheels/` contains the wheel built from this frozen attachment; the installer
validates its SHA256 before installing it. Python dependencies are pinned in
`../requirements-czsc-runtime.lock`.

## Rebuild

Use Python 3.12 and a Rust 2024-compatible compiler, then run:

```text
python -m pip install -r app/requirements-czsc-runtime.lock
python app/build_czsc_runtime.py --out app/vendor/wheels
```

This verifies the retained sources, invokes Maturin with Cargo `--locked`,
records build provenance, and installs the resulting wheel into that Python
environment. Cargo obtains the versions and checksums frozen in Cargo.lock.
The script does not install a global compiler. Keep only one CZSC wheel in
the selected output directory when rebuilding for a different platform.
Build intermediates default to `app/.runtime-build/czsc-target`; an explicit
`CARGO_TARGET_DIR` is respected. They are not part of the vendored source or
the installed wheel.

The supplied Windows amd64 wheel was built with task-local Rust GNU and
LLVM-MinGW UCRT. For that toolchain, set `CC` and
`CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER` to LLVM-MinGW's
`x86_64-w64-mingw32-clang.exe`, `AR` to `llvm-ar.exe`, and put its bin directory
on PATH. Set RUSTFLAGS to `-Lnative=<gcc-support> -C dlltool=<llvm-dlltool.exe>`;
gcc-support contains only `libgcc.a`, `libgcc_eh.a`, `libgcc_s.a` from Rust's
GNU self-contained directory. Adding that entire directory would mix its
MSVCRT imports with LLVM's UCRT startup. These settings change the build
toolchain, not CZSC source. The exact successful versions and artifact hashes
are in the manifest. The supplied build uses environment overrides
`CARGO_PROFILE_RELEASE_LTO=false` and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`
to avoid optimizing the entire Polars/signal workspace in a single LTO unit;
optimization remains level 3 and the frozen source files are unchanged.
On other platforms, use an appropriate native compiler
and Rust toolchain; the bundled Windows wheel is not installed there.

"""Build the frozen attachment and record the installed wheel's provenance.

Usage: python build_czsc_runtime.py [--out DIRECTORY]
Requires Python >=3.10, maturin==1.13.1 and an appropriate Rust linker.
On Windows, either MSVC tools or a Rust GNU toolchain with its self-contained
linker is supported. This script never installs or changes a global toolchain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig


def _version(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-install", action="store_true", help="Build a release wheel without installing it")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest_path = root / "czsc-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = root / "vendor" / "czsc"
    for entry in manifest["files"]:
        path = source / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Frozen source mismatch: {entry['path']}")
    os.environ.setdefault("CARGO_TARGET_DIR", str(root / ".runtime-build" / "czsc-target"))
    os.environ.setdefault("CARGO_PROFILE_RELEASE_LTO", "false")
    os.environ.setdefault("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "16")
    output = (arguments.out or root / ".runtime-wheels").resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "maturin", "build", "--release", "--locked", "--manifest-path",
               str(source / "crates" / "czsc-python" / "Cargo.toml"), "--out", str(output)]
    subprocess.run(command, cwd=source, check=True)
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    compatible_tags = set(sys_tags())
    wheels = [wheel for wheel in output.glob("czsc-1.0.1-*.whl")
              if parse_wheel_filename(wheel.name)[3] & compatible_tags]
    if len(wheels) != 1:
        raise SystemExit("Expected exactly one frozen CZSC wheel in output directory")
    wheel = wheels[0]
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    build = {
        "python": sys.version.split()[0], "system": platform.system(), "machine": platform.machine() or sysconfig.get_platform(),
        "wheel_tags": wheel.stem.split("-")[-3:],
        "rustc": _version(["rustc", "--version"]), "cargo": _version(["cargo", "--version"]),
        "maturin": _version([sys.executable, "-m", "maturin", "--version"]),
        "compiler": _version([os.environ["CC"], "--version"]) if os.environ.get("CC") else "system toolchain",
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "release_profile_overrides": {key: os.environ[key] for key in ("CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_CODEGEN_UNITS") if key in os.environ},
        "source_tree_sha256": hashlib.sha256("\n".join(f"{item['path']} {item['sha256']}" for item in manifest["files"]).encode()).hexdigest(),
        "cargo_locked": True, "source_algorithm_patches": False,
        "cargo_target_dir": os.environ["CARGO_TARGET_DIR"],
    }
    # Keep the original attachment manifest unchanged. Platform build evidence
    # is additive and travels beside wheels as well as inside KHQuant wheels.
    index_path = root / "vendor" / "wheels" / "wheel-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {
        "schema_version": 1, "package": "czsc", "version": manifest["version"],
        "source_tree_sha256": build["source_tree_sha256"], "wheels": []}
    if index.get("source_tree_sha256") != build["source_tree_sha256"]:
        raise SystemExit("Existing wheel index belongs to different frozen source")
    index["wheels"] = [entry for entry in index["wheels"] if entry["filename"] != wheel.name]
    index["wheels"].append({"filename": wheel.name, "sha256": wheel_hash,
                            "tags": sorted(str(tag) for tag in parse_wheel_filename(wheel.name)[3]), "build": build})
    for path in dict.fromkeys((index_path, output / "wheel-index.json")):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    if not arguments.no_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
    print(f"Built frozen CZSC wheel {wheel.name} SHA256={wheel_hash}")


if __name__ == "__main__":
    main()

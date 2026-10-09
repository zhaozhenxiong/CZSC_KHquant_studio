"""Assemble a source Release ZIP from the current workspace and verified wheels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from packaging.utils import parse_wheel_filename
from packaging.version import Version

from my_strategy.scripts.migration_manifest import write_integrity_manifest
from my_strategy.scripts.package_project import build_package, verify_package


APP = Path(__file__).resolve().parent
ROOT = APP.parent


def wheel_index(directories: list[Path], manifest: dict) -> tuple[dict, dict[str, Path]]:
    source_hash = hashlib.sha256("\n".join(f"{entry['path']} {entry['sha256']}" for entry in manifest["files"]).encode()).hexdigest()
    merged = {"schema_version": 1, "package": "czsc", "version": manifest["version"],
              "source_tree_sha256": source_hash, "wheels": []}
    files: dict[str, Path] = {}
    for directory in directories:
        index_path = directory / "wheel-index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if (type(index.get("schema_version")) is not int or index["schema_version"] != 1
                    or index.get("source_tree_sha256") != source_hash
                    or index.get("package") != "czsc"
                    or index.get("version") != manifest["version"]):
                raise ValueError(f"Wheel index source mismatch: {index_path}")
            entries = index["wheels"]
        else:
            entries = [{"filename": manifest["built_wheel_name"], "sha256": manifest["built_wheel_sha256"],
                        "tags": ["-".join(manifest["build"]["wheel_tags"])], "build": manifest["build"]}]
        for entry in entries:
            name = entry["filename"]
            if (not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
                    or not name.endswith(".whl")):
                raise ValueError(f"Unsafe wheel filename: {name}")
            package, version, _build, tags = parse_wheel_filename(name)
            if package != "czsc" or version != Version(manifest["version"]):
                raise ValueError(f"Wheel filename package/version mismatch: {name}")
            if set(entry.get("tags", [])) != {str(tag) for tag in tags}:
                raise ValueError(f"Wheel filename/index tags mismatch: {name}")
            wheel = directory / name
            if not wheel.is_file():
                continue  # A combined index may describe another platform's asset.
            if not entry["build"].get("cargo_locked") or entry["build"].get("source_algorithm_patches") is not False:
                raise ValueError(f"Wheel build does not preserve frozen source: {name}")
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Wheel checksum mismatch: {wheel}")
            if name in files:
                previous = next(value for value in merged["wheels"] if value["filename"] == name)
                if previous["sha256"] != entry["sha256"]:
                    raise ValueError(f"Conflicting wheel builds: {name}")
                continue
            files[name] = wheel
            merged["wheels"].append(entry)
    merged["wheels"].sort(key=lambda entry: entry["filename"])
    return merged, files


def release_archive(package: Path) -> Path:
    """Keep script execution permissions in ZIPs assembled on Windows."""
    archive = package.parent / f"{package.name}.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        for path in [package, *sorted(package.rglob("*"))]:
            info = ZipInfo.from_file(path, arcname=path.relative_to(package.parent).as_posix())
            info.create_system = 3
            mode = 0o40755 if path.is_dir() else 0o100755 if path.suffix in {".sh", ".command"} else 0o100644
            info.external_attr = mode << 16 | (0x10 if path.is_dir() else 0)
            info.compress_type = ZIP_DEFLATED
            if path.is_dir():
                output.writestr(info, b"")
            else:
                with path.open("rb") as source, output.open(info, "w") as target:
                    shutil.copyfileobj(source, target)
    return archive


def build_release(output: Path, wheel_directories: list[Path]) -> dict:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    source = json.loads((APP / "czsc-source-manifest.json").read_text(encoding="utf-8"))
    index, wheels = wheel_index(wheel_directories, source)
    if not wheels:
        raise ValueError("A release needs at least one verified native wheel")
    output = output.resolve()
    package = output / f"KHQuant-{version}"
    payload = build_package(source=APP, target=package, include_data=False, include_artifacts="none")
    destination = package / "app" / "vendor" / "wheels"
    destination.mkdir(parents=True, exist_ok=True)
    for name, wheel in wheels.items():
        shutil.copy2(wheel, destination / name)
    (destination / "wheel-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    state = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False)
    release = {"version": version, "git_revision": revision.stdout.strip() or None,
               "workspace_changes_included": bool(state.stdout.strip()), "native_wheels": sorted(wheels),
               "contains_market_data": False, "contains_personal_data": False,
               "platform_verification": json.loads((ROOT / "runtime-manifest.json").read_text(encoding="utf-8")).get("platform_verification", {})}
    (package / "release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
    write_integrity_manifest(package, source_project_root=APP, filename="image-manifest.json",
                             reject_external_symlinks=True, metadata={key: payload[key] for key in
                                 ("architecture", "include_data", "include_artifacts", "include_personal", "run_ids", "artifact_selection", "release_event_count") if key in payload})
    issues = verify_package(package)
    if issues:
        raise RuntimeError("Release verification failed: " + "; ".join(issues))
    archive = release_archive(package)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "SHA256SUMS.txt").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {**release, "archive": str(archive), "archive_sha256": digest, "package": str(package)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new release directory")
    parser.add_argument("--wheel-dir", type=Path, action="append", help="Repeat for CI platform wheel directories")
    arguments = parser.parse_args()
    print(json.dumps(build_release(arguments.output_dir, arguments.wheel_dir or [APP / "vendor" / "wheels"]), indent=2))


if __name__ == "__main__":
    main()

"""Release native assets must match their filename, bytes and frozen source."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.utils import parse_wheel_filename

from build_release import release_archive, wheel_index


@pytest.fixture
def manifest():
    return {"package": "czsc", "version": "1.0.1",
            "files": [{"path": "engine.py", "sha256": hashlib.sha256(b"frozen algorithm").hexdigest()}]}


def indexed_wheel(directory: Path, manifest: dict, name: str = "czsc-1.0.1-cp312-cp312-macosx_14_0_arm64.whl",
                  content: bytes = b"fixture native wheel bytes") -> tuple[dict, dict]:
    directory.mkdir(parents=True)
    (directory / name).write_bytes(content)
    entry = {"filename": name, "sha256": hashlib.sha256(content).hexdigest(),
             "tags": sorted(str(tag) for tag in parse_wheel_filename(name)[3]),
             "build": {"cargo_locked": True, "source_algorithm_patches": False}}
    source_hash = hashlib.sha256("\n".join(f"{item['path']} {item['sha256']}" for item in manifest["files"]).encode()).hexdigest()
    index = {"schema_version": 1, "package": "czsc", "version": manifest["version"],
             "source_tree_sha256": source_hash, "wheels": [entry]}
    save_index(directory, index)
    return index, entry


def save_index(directory: Path, index: dict) -> None:
    (directory / "wheel-index.json").write_text(json.dumps(index), encoding="utf-8")


def test_merge_platform_assets_binds_names_tags_and_bytes(tmp_path, manifest):
    first, second = tmp_path / "mac", tmp_path / "linux"
    mac_index, mac = indexed_wheel(first, manifest)
    linux_index, linux = indexed_wheel(second, manifest, "czsc-1.0.1-cp312-cp312-manylinux_2_17_x86_64.whl")
    # A combined index can refer to an asset downloaded into another directory.
    mac_index["wheels"].append(deepcopy(linux))
    save_index(first, mac_index)
    merged, assets = wheel_index([first, second], manifest)
    assert set(assets) == {mac["filename"], linux["filename"]}
    assert len(merged["wheels"]) == 2
    for item in merged["wheels"]:
        assert hashlib.sha256(assets[item["filename"]].read_bytes()).hexdigest() == item["sha256"]
        assert set(item["tags"]) == {str(tag) for tag in parse_wheel_filename(item["filename"])[3]}


@pytest.mark.parametrize("name", ["other-1.0.1-cp312-cp312-win_amd64.whl", "czsc-9.0.0-cp312-cp312-win_amd64.whl"])
def test_reject_other_package_or_version_even_when_index_and_sha_are_valid(tmp_path, manifest, name):
    directory = tmp_path / "native"
    indexed_wheel(directory, manifest, name)
    with pytest.raises(ValueError, match="package/version mismatch"):
        wheel_index([directory], manifest)


def test_reject_index_tags_that_misrepresent_filename(tmp_path, manifest):
    directory = tmp_path / "native"
    index, entry = indexed_wheel(directory, manifest)
    entry["tags"] = ["cp312-cp312-win_amd64"]
    save_index(directory, index)
    with pytest.raises(ValueError, match="tags mismatch"):
        wheel_index([directory], manifest)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("schema_version", None),
                                         ("schema_version", True), ("source_tree_sha256", "another-source")])
def test_reject_unsupported_index_or_frozen_source_conflict(tmp_path, manifest, field, value):
    directory = tmp_path / "native"
    index, _entry = indexed_wheel(directory, manifest)
    index[field] = value
    save_index(directory, index)
    with pytest.raises(ValueError, match="source mismatch"):
        wheel_index([directory], manifest)


@pytest.mark.parametrize("prefix", ["../", "..\\", "directory/", "directory\\"])
def test_reject_both_platform_path_separators_before_resolving_assets(tmp_path, manifest, prefix):
    directory = tmp_path / "native"
    index, entry = indexed_wheel(directory, manifest)
    entry["filename"] = prefix + entry["filename"]
    save_index(directory, index)
    with pytest.raises(ValueError, match="Unsafe wheel filename"):
        wheel_index([directory], manifest)


def test_reject_changed_wheel_bytes_and_conflicting_same_name_builds(tmp_path, manifest):
    first, second = tmp_path / "one", tmp_path / "two"
    _index, entry = indexed_wheel(first, manifest)
    indexed_wheel(second, manifest, content=b"different native build")
    with pytest.raises(ValueError, match="Conflicting wheel builds"):
        wheel_index([first, second], manifest)
    (first / entry["filename"]).write_bytes(b"changed after index creation")
    with pytest.raises(ValueError, match="checksum mismatch"):
        wheel_index([first], manifest)


def test_existing_frozen_native_wheel_validates_without_rewriting_manifest():
    app = Path(__file__).resolve().parents[2]
    source_path = app / "czsc-source-manifest.json"
    original = source_path.read_bytes()
    source = json.loads(original)
    directory = app / "vendor/wheels"
    if not (directory / source["built_wheel_name"]).is_file():
        pytest.skip("native asset not present on this test host")
    merged, assets = wheel_index([directory], source)
    name = source["built_wheel_name"]
    assert name in assets
    assert hashlib.sha256(assets[name].read_bytes()).hexdigest() == source["built_wheel_sha256"]
    assert merged["version"] == source["version"]
    assert source_path.read_bytes() == original


def test_release_archive_preserves_unix_script_modes_bytes_and_package_layout(tmp_path):
    package = tmp_path / "KHQuant-0.1.0"
    files = {"khquant-native.sh": b"#!/bin/sh\nprintf 'start\\n'\n",
             "macos/install.command": b"#!/bin/sh\nexec python3 install.py\n",
             "README.md": b"fixture documentation\r\n",
             "app/model.pt": b"\x00\xff\n"}
    for relative, content in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (package / "app/empty-state").mkdir()
    archive = release_archive(package)
    assert archive == tmp_path / "KHQuant-0.1.0.zip"
    with ZipFile(archive) as zipped:
        assert "KHQuant-0.1.0/" in zipped.namelist()
        assert "KHQuant-0.1.0/app/empty-state/" in zipped.namelist()
        assert all(name.startswith("KHQuant-0.1.0/") for name in zipped.namelist())
        for relative, content in files.items():
            name = f"KHQuant-0.1.0/{relative}"
            info = zipped.getinfo(name)
            expected_mode = 0o100755 if Path(relative).suffix in {".sh", ".command"} else 0o100644
            assert info.create_system == 3
            assert info.external_attr >> 16 == expected_mode
            assert zipped.read(name) == content
            assert (package / relative).read_bytes() == content

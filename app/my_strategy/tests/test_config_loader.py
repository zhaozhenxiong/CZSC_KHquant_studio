"""Unit tests for my_strategy.core.config_loader."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from my_strategy.core.config_loader import (
    apply_env_overrides,
    get_config_path,
    load_config,
    load_json_config,
    load_yaml_config,
)


class ConfigLoaderTests(unittest.TestCase):
    def test_get_config_path_defaults_to_yaml(self) -> None:
        path = get_config_path("data_config")
        self.assertEqual(path.suffix, ".yaml")
        self.assertTrue(path.name.startswith("data_config"))

    def test_get_config_path_returns_existing_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "my_strategy" / "configs"
            config_dir.mkdir(parents=True)
            (config_dir / "test.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
            original = get_config_path.__module__
            try:
                import my_strategy.core.config_loader as loader

                original_dir = loader.CONFIGS_DIR
                loader.CONFIGS_DIR = config_dir
                path = get_config_path("test")
                self.assertEqual(path.suffix, ".json")
            finally:
                import my_strategy.core.config_loader as loader

                loader.CONFIGS_DIR = original_dir

    def test_load_yaml_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.yaml"
            path.write_text("root:\n  key: value\nlist:\n  - a\n  - b\n", encoding="utf-8")
            data = load_yaml_config(path)
            self.assertEqual(data["root"]["key"], "value")
            self.assertEqual(data["list"], ["a", "b"])

    def test_load_json_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(json.dumps({"x": 1, "y": [2, 3]}), encoding="utf-8")
            data = load_json_config(path)
            self.assertEqual(data["x"], 1)
            self.assertEqual(data["y"], [2, 3])

    def test_load_config_accepts_name_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.yaml"
            path.write_text("key: value\n", encoding="utf-8")
            data = load_config(path)
            self.assertEqual(data["key"], "value")

    def test_apply_env_overrides_nested(self) -> None:
        config = {"data": {"root": "/default"}, "agents": {"ml": {"enabled": False}}}
        env = {
            "KHQUANT__DATA__ROOT": "/override",
            "KHQUANT__AGENTS__ML__ENABLED": "true",
        }
        original = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            result = apply_env_overrides(config, prefix="KHQUANT", separator="__")
            self.assertEqual(result["data"]["root"], "/override")
            self.assertTrue(result["agents"]["ml"]["enabled"])
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_load_config_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("definitely_missing_config_xyz")

    def test_invalid_strategy_variant_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-variant.yaml"
            path.write_text("decision:\n  strategy_variant: unknown_variant\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strategy_variant"):
                load_yaml_config(path)


if __name__ == "__main__":
    unittest.main()

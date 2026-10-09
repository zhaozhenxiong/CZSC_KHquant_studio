"""Centralized configuration loading for KHQuant.

All YAML/JSON configuration files under ``my_strategy/configs/`` should be
loaded through this module so that path resolution, environment overrides,
and defaults are handled consistently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from my_strategy.core.paths import PROJECT_ROOT


def _resolve_configs_dir() -> Path:
    if os.environ.get("KHQUANT_CONFIG_ROOT"):
        return Path(os.environ["KHQUANT_CONFIG_ROOT"]).resolve()
    return (PROJECT_ROOT / "my_strategy" / "configs").resolve()


CONFIGS_DIR = _resolve_configs_dir()


def get_config_path(name: str) -> Path:
    """Return the absolute path to a config file in ``my_strategy/configs/``.

    ``name`` may be a bare config name (e.g. ``"czsc_strategy"``) or a
    relative path under ``configs/``.
    The appropriate extension (``.yaml``, ``.yml``, or ``.json``) is appended
    if missing and the file exists.
    """
    path = CONFIGS_DIR / name
    if path.suffix in (".yaml", ".yml", ".json"):
        return path.resolve()
    for ext in (".yaml", ".yml", ".json"):
        candidate = path.with_suffix(ext)
        if candidate.exists():
            return candidate.resolve()
    return path.with_suffix(".yaml").resolve()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dict.

    Relative paths are resolved against ``PROJECT_ROOT``. Missing files raise
    ``FileNotFoundError``. Empty files return an empty dict.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) or {}
    _validate_config(config)
    return config


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file and return it as a dict.

    Relative paths are resolved against ``PROJECT_ROOT``.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def load_config(name: str | Path) -> dict[str, Any]:
    """Load a config file from ``my_strategy/configs/`` by name or path.

    ``name`` may be a bare config name (e.g. ``"czsc_strategy"``), a
    relative path under the project root, or an absolute path. JSON configs
    are loaded with ``json.loads``; YAML configs with ``yaml.safe_load``.
    """
    raw = Path(name)
    if raw.is_absolute() or raw.suffix in (".yaml", ".yml", ".json"):
        if raw.suffix == ".json":
            return load_json_config(raw)
        return load_yaml_config(raw)
    path = get_config_path(name)
    if path.suffix == ".json":
        config = load_json_config(path)
    else:
        config = load_yaml_config(path)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError for config values that are structurally invalid.

    This is called automatically by :func:`load_config` after loading so that
    typos in enums fail fast instead of silently falling back to defaults.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    if "decision" in config:
        raise ValueError("decision.strategy_variant has been retired; use czsc_strategy")


def apply_env_overrides(
    config: dict[str, Any],
    prefix: str = "KHQUANT",
    separator: str = "__",
) -> dict[str, Any]:
    """Return a new config with environment-variable overrides applied.

    Variables are expected in the form ``PREFIX__section__key=value``. Nested
    sections are supported via ``separator``. Values are parsed as JSON when
    possible; otherwise kept as strings.

    Example: ``KHQUANT__DATA__ROOT=/tmp/data`` overrides
    ``config["data"]["root"]``.
    """
    result: dict[str, Any] = _deep_copy(config)
    for key, value in os.environ.items():
        if not key.startswith(prefix + separator):
            continue
        parts = [p.lower() for p in key[len(prefix) + len(separator):].split(separator)]
        if not parts:
            continue
        target = result
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _parse_env_value(value)
    return result


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _parse_env_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

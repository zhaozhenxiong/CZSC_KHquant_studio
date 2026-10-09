"""Validate local assets, shell structure and JavaScript without starting jobs."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess


class ShellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.assets: list[str] = []
        self.routes: list[str] = []
        self.number_inputs: list[dict[str, str | None]] = []
        self.date_inputs: dict[str, str | None] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        if attributes.get("data-route"):
            self.routes.append(attributes["data-route"])
        if tag == "script" and attributes.get("src"):
            self.assets.append(attributes["src"])
        if tag == "link" and attributes.get("rel") == "stylesheet":
            self.assets.append(attributes["href"])
        if tag == "input" and attributes.get("type") == "number" and attributes.get("value"):
            self.number_inputs.append(attributes)
        if tag == "input" and attributes.get("type") == "date" and attributes.get("id"):
            self.date_inputs[attributes["id"]] = attributes.get("value")


def validate(node: str | None = None) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1] / "static"
    parser = ShellParser()
    parser.feed((root / "index.html").read_text(encoding="utf-8"))
    if len(parser.ids) != len(set(parser.ids)):
        raise ValueError("Duplicate shell element id")
    if set(parser.routes) != {"analysis", "scan", "backtest", "watchlist", "holdings", "data"}:
        raise ValueError("The workbench must have its six routes")
    for name in ("analysis-start", "scan-start", "backtest-start"):
        if parser.date_inputs.get(name) != "2026-01-01":
            raise ValueError(f"Incorrect research date default: {name}")
    for name in ("analysis-end", "scan-end", "backtest-end"):
        if parser.date_inputs.get(name):
            raise ValueError(f"Research end date must come from actual local data: {name}")
    for field in parser.number_inputs:
        value = Decimal(field["value"])
        minimum = Decimal(field["min"]) if field.get("min") is not None else None
        maximum = Decimal(field["max"]) if field.get("max") is not None else None
        if not value.is_finite() or minimum is not None and value < minimum or maximum is not None and value > maximum:
            raise ValueError(f"Invalid default numeric range: {field.get('id')}")
        if field.get("step") != "any":
            step = Decimal(field.get("step") or "1")
            base = minimum if minimum is not None else value
            if step <= 0 or (value - base) % step != 0:
                raise ValueError(f"Default numeric value has a step mismatch: {field.get('id')}")
    for asset in parser.assets:
        if not asset.startswith("/static/"):
            raise ValueError(f"Remote or unexpected runtime asset: {asset}")
        path = (root / asset.removeprefix("/static/")).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing local runtime asset: {asset}")
    script = root / "js" / "workbench.js"
    literal_ids = set(re.findall(r"\$\('([^']+)'\)", script.read_text(encoding="utf-8")))
    missing = literal_ids - set(parser.ids)
    if missing:
        raise ValueError(f"JavaScript references missing shell elements: {sorted(missing)}")
    vendor = root / "vendor" / "lightweight-charts"
    provenance = json.loads((vendor / "provenance.json").read_text(encoding="utf-8"))
    if provenance["version"] != "4.1.3":
        raise ValueError("Unexpected chart runtime version")
    asset = vendor / "lightweight-charts.standalone.production.js"
    if hashlib.sha256(asset.read_bytes()).hexdigest() != provenance["asset_sha256"]:
        raise ValueError("Chart asset checksum mismatch")
    if not (vendor / "LICENSE").is_file() or not (vendor / "NOTICE").is_file():
        raise ValueError("Missing chart license or attribution")
    executable = node or shutil.which("node")
    if not executable:
        raise ValueError("Pass --node to validate JavaScript syntax")
    for path in (root / item.removeprefix("/static/") for item in parser.assets if item.endswith(".js")):
        subprocess.run([executable, "--check", str(path)], check=True, capture_output=True, text=True)
    return {"status": "passed", "routes": parser.routes, "unique_ids": len(parser.ids),
            "local_assets": len(parser.assets), "literal_js_ids": len(literal_ids), "valid_number_defaults": len(parser.number_inputs), "chart_version": provenance["version"]}


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--node")
    print(json.dumps(validate(arguments.parse_args().node), ensure_ascii=False))

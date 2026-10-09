from __future__ import annotations

import pandas as pd

from my_strategy.scripts.update_daily_data import _metadata_refresh_status


def test_metadata_refresh_status_skips_current_eligible_stocks() -> None:
    metadata = pd.DataFrame([
        {"stock": "000001.SZ", "industry_source_fetched_at": "2026-09-08T16:40:47+08:00", "classification_eligible": "true"},
        {"stock": "600000.SH", "industry_source_fetched_at": "2026-09-08T16:40:47+08:00", "classification_eligible": "yes"},
    ])

    result = _metadata_refresh_status(metadata, ["000001.SZ", "600000.SH"], "2026-09-08")

    assert result == {
        "is_current": True,
        "requested_stocks": 2,
        "current_stocks": 2,
        "missing_or_ineligible_stocks": 0,
    }


def test_metadata_refresh_status_skips_repeat_when_current_snapshot_has_missing_stock() -> None:
    metadata = pd.DataFrame([
        {"stock": "000001.SZ", "industry_source_fetched_at": "2026-09-08T16:40:47+08:00", "classification_eligible": "true"},
    ])

    result = _metadata_refresh_status(metadata, ["000001.SZ", "600000.SH"], "2026-09-08")

    assert result == {
        "is_current": True,
        "requested_stocks": 2,
        "current_stocks": 1,
        "missing_or_ineligible_stocks": 1,
    }


def test_metadata_refresh_status_requires_a_current_source_snapshot() -> None:
    metadata = pd.DataFrame([
        {"stock": "000001.SZ", "industry_source_fetched_at": "2026-09-07T16:40:47+08:00", "classification_eligible": "true"},
    ])

    result = _metadata_refresh_status(metadata, ["000001.SZ"], "2026-09-08")

    assert result["is_current"] is False

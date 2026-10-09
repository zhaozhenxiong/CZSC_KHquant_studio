"""SQLite authority tests for stock industry metadata."""

from __future__ import annotations

import sqlite3

import pandas as pd

from my_strategy.storage.raw_repository import RawMarketRepository
from my_strategy.storage.warehouse_schema import SCHEMA_VERSION


def _metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock": ["000001.SZ", "600837.SH", "UNKNOWN1.SZ"],
            "stock_name": ["平安银行", "海通证券", ""],
            "board": ["sz_main", "sh_main", "other"],
            "industry": ["金融业", "", ""],
            "neutral_group": ["金融业", "sh_main", "other"],
            "industry_l1_code": ["J", "", ""],
            "industry_l1_name": ["金融业", "", ""],
            "industry_l2_code": ["J66", "", ""],
            "industry_l2_name": ["货币金融服务", "", ""],
            "industry_taxonomy": ["csrc_2012_l1", "csrc_2012_l1", ""],
            "industry_source": [
                "baostock:query_stock_industry",
                "baostock:query_stock_industry",
                "khquant:legacy_metadata_migration",
            ],
            "industry_source_provider": ["baostock", "baostock", "khquant"],
            "industry_source_endpoint": [
                "query_stock_industry",
                "query_stock_industry",
                "legacy_metadata_migration",
            ],
            "industry_confidence": ["high", "none", "none"],
            "industry_confidence_score": [1.0, 0.0, 0.0],
            "industry_confidence_method": [
                "exact_stock_code_and_taxonomy_parse",
                "source_returned_no_classification",
                "invalid_stock_code",
            ],
            "industry_source_updated_at": ["2026-08-17", "2026-08-17", ""],
            "industry_source_fetched_at": ["2026-08-19T10:00:00+08:00"] * 3,
            "metadata_updated_at": ["2026-08-19T10:05:00+08:00"] * 3,
            "classification_eligible": [True, False, False],
        }
    )


def test_industry_metadata_sqlite_is_authoritative_and_audited(tmp_path) -> None:
    db_path = tmp_path / "raw.db"
    repo = RawMarketRepository(db_path)
    exclusions = pd.DataFrame(
        {
            "stock": ["000002.SZ", "600837.SH", "UNKNOWN1.SZ"],
            "reason": ["missing_metadata", "no_unified_industry", "invalid_stock_code"],
        }
    )
    summary = {
        "created_at": "2026-08-19T10:04:00+08:00",
        "classification": {
            "taxonomy": "csrc_2012_l1",
            "source": "baostock:query_stock_industry",
            "source_provider": "baostock",
            "source_endpoint": "query_stock_industry",
            "source_updated_at": "2026-08-17",
            "source_fetched_at": "2026-08-19T10:00:00+08:00",
            "source_rows": 5542,
            "coverage": 1 / 3,
        },
    }

    result = repo.replace_industry_metadata(
        _metadata_frame(),
        exclusions,
        run_id="industry-unit-1",
        summary=summary,
    )

    current = repo.read_industry_metadata()
    eligible = repo.read_industry_metadata(eligible_only=True)
    stored_exclusions = repo.read_industry_exclusions()
    latest_run = repo.read_latest_industry_run()
    assert result["quick_check"] == "ok"
    assert len(current) == 3
    assert eligible["stock"].tolist() == ["000001.SZ"]
    assert current.loc[current["stock"] == "000001.SZ", "industry_confidence_score"].iloc[0] == 1.0
    assert current.loc[current["stock"] == "600837.SH", "industry_confidence_method"].iloc[0] == (
        "source_returned_no_classification"
    )
    assert len(stored_exclusions) == 3
    assert current.attrs["khquant_data_source"] == "raw_sqlite:stock_industry_metadata"
    assert latest_run["run_id"] == "industry-unit-1"
    assert latest_run["summary"]["classification"]["source_provider"] == "baostock"

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT value FROM warehouse_meta WHERE key='schema_version'").fetchone()[0] == str(
            SCHEMA_VERSION
        )
        run = conn.execute(
            "SELECT source_provider, source_endpoint, source_updated_at, source_fetched_at "
            "FROM industry_classification_runs WHERE run_id='industry-unit-1'"
        ).fetchone()
        assert run == (
            "baostock",
            "query_stock_industry",
            "2026-08-17",
            "2026-08-19T10:00:00+08:00",
        )

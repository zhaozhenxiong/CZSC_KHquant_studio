"""Tests for the unified CSRC level-one industry backfill."""

from __future__ import annotations

import pandas as pd
import pytest

from my_strategy.data_manager.industry_enricher import (
    CSRC_SOURCE,
    CSRC_TAXONOMY,
    apply_csrc_industry_snapshot,
    audit_csrc_source_snapshot,
    build_industry_exclusions,
    parse_csrc_industry,
)


def _snapshot(rows: list[tuple[str, str]]) -> pd.DataFrame:
    records = []
    for stock, raw_industry in rows:
        parsed = parse_csrc_industry(raw_industry)
        records.append(
            {
                "stock": stock,
                "code_name": stock,
                "industry": raw_industry,
                "industry_source": CSRC_SOURCE,
                "industry_taxonomy": CSRC_TAXONOMY,
                "industry_source_updated_at": "2026-08-17",
                "industry_source_fetched_at": "2026-08-19T10:00:00+08:00",
                **parsed,
            }
        )
    return pd.DataFrame(records)


def test_parse_csrc_industry_keeps_l1_and_l2() -> None:
    parsed = parse_csrc_industry("J66货币金融服务")

    assert parsed == {
        "industry_l1_code": "J",
        "industry_l1_name": "金融业",
        "industry_l2_code": "J66",
        "industry_l2_name": "货币金融服务",
    }


def test_apply_snapshot_unifies_taxonomy_without_adding_source_only_stocks() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ", "600000.SH", "UNKNOWN1.SZ"],
            "board": ["sz_main", "sh_main", "other"],
            "stock_name": ["平安银行", "浦发银行", ""],
            "industry": ["Finance", "银行", "legacy"],
            "neutral_group": ["Finance", "银行", "other"],
        }
    )
    source = _snapshot(
        [
            ("000001.SZ", "J66货币金融服务"),
            ("600000.SH", "J66货币金融服务"),
            ("300750.SZ", "C38电气机械和器材制造业"),
        ]
    )

    updated, report = apply_csrc_industry_snapshot(existing, source)

    assert updated["stock"].tolist() == ["000001.SZ", "600000.SH", "UNKNOWN1.SZ"]
    assert set(updated.loc[updated["stock"].isin(["000001.SZ", "600000.SH"]), "industry"]) == {"金融业"}
    assert set(updated.loc[updated["stock"].isin(["000001.SZ", "600000.SH"]), "industry_taxonomy"]) == {
        CSRC_TAXONOMY
    }
    assert set(updated.loc[updated["stock"].isin(["000001.SZ", "600000.SH"]), "industry_source_provider"]) == {
        "baostock"
    }
    assert set(updated.loc[updated["stock"].isin(["000001.SZ", "600000.SH"]), "industry_confidence_score"]) == {
        1.0
    }
    assert updated.loc[updated["stock"] == "UNKNOWN1.SZ", "classification_eligible"].iloc[0] == "false"
    assert report["target_stocks"] == 2
    assert report["classified_stocks"] == 2
    assert report["coverage"] == 1.0


def test_apply_snapshot_rejects_low_coverage_without_partial_result() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ", "600000.SH"],
            "board": ["sz_main", "sh_main"],
            "stock_name": ["平安银行", "浦发银行"],
            "industry": ["Finance", "Finance"],
        }
    )
    source = _snapshot([("000001.SZ", "J66货币金融服务")])

    with pytest.raises(ValueError, match="coverage"):
        apply_csrc_industry_snapshot(existing, source, min_coverage=0.98)


def test_apply_snapshot_is_repeatable_after_csv_boolean_inference() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ", "600000.SH"],
            "board": ["sz_main", "sh_main"],
            "stock_name": ["平安银行", "浦发银行"],
            "industry": ["金融业", "金融业"],
            "classification_eligible": [True, True],
        }
    )
    source = _snapshot(
        [
            ("000001.SZ", "J66货币金融服务"),
            ("600000.SH", "J66货币金融服务"),
        ]
    )

    updated, report = apply_csrc_industry_snapshot(existing, source)

    assert updated["classification_eligible"].tolist() == ["true", "true"]
    assert report["coverage"] == 1.0


def test_exclusion_audit_includes_every_unclassified_metadata_row() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ", "600837.SH", "UNKNOWN1.SZ"],
        }
    )
    updated = existing.assign(classification_eligible=[True, False, False])

    exclusions = build_industry_exclusions(
        {"000001.SZ", "000002.SZ"},
        updated,
    )

    assert set(map(tuple, exclusions[["stock", "reason"]].to_records(index=False))) == {
        ("000002.SZ", "missing_metadata"),
        ("600837.SH", "no_unified_industry"),
        ("UNKNOWN1.SZ", "invalid_stock_code"),
    }


def test_identical_source_duplicates_are_observable_but_not_conflicts() -> None:
    source = pd.concat(
        [
            _snapshot([("000001.SZ", "J66货币金融服务")]),
            _snapshot([("000001.SZ", "J66货币金融服务")]),
        ],
        ignore_index=True,
    )

    quality = audit_csrc_source_snapshot(source)

    assert quality["duplicate_rows"] == 2
    assert quality["conflict_count"] == 0


def test_conflicting_duplicate_source_classifications_are_blocked() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ"],
            "industry": ["金融业"],
            "classification_eligible": [True],
        }
    )
    source = pd.concat(
        [
            _snapshot([("000001.SZ", "J66货币金融服务")]),
            _snapshot([("000001.SZ", "C38电气机械和器材制造业")]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="source conflicts"):
        apply_csrc_industry_snapshot(existing, source)


def test_coverage_regression_is_blocked_even_above_minimum() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ", "600000.SH"],
            "industry": ["金融业", "金融业"],
            "industry_l1_code": ["J", "J"],
            "classification_eligible": [True, True],
        }
    )
    source = _snapshot(
        [
            ("000001.SZ", "J66货币金融服务"),
            ("600000.SH", ""),
        ]
    )

    with pytest.raises(ValueError, match="coverage regression"):
        apply_csrc_industry_snapshot(
            existing,
            source,
            min_coverage=0.4,
            max_coverage_drop=0.0,
        )


def test_authority_change_requires_newer_source_effective_date() -> None:
    existing = pd.DataFrame(
        {
            "stock": ["000001.SZ"],
            "industry": ["金融业"],
            "industry_l1_code": ["J"],
            "industry_l1_name": ["金融业"],
            "industry_source_updated_at": ["2026-08-17"],
            "classification_eligible": [True],
        }
    )
    stale = _snapshot([("000001.SZ", "C38电气机械和器材制造业")])

    with pytest.raises(ValueError, match="source conflicts"):
        apply_csrc_industry_snapshot(existing, stale)

    newer = stale.copy()
    newer["industry_source_updated_at"] = "2026-08-18"
    updated, report = apply_csrc_industry_snapshot(existing, newer)

    assert updated.loc[0, "industry_l1_code"] == "C"
    assert report["source_quality"]["classification_change_count"] == 1
    assert report["source_quality"]["conflict_count"] == 0
    assert report["quality_gates"]["status"] == "passed"

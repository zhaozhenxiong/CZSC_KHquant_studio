"""Industry/board/stock_name enricher for stock metadata.

Reads and updates SQLite-authoritative stock metadata from multiple upstream
sources with automatic fallback. ``stock_metadata.csv`` is a regenerated
compatibility export.

Priority order for online enrichment:
1. AkShare (Eastmoney industry boards + individual stock info)
2. Baostock basic stock info (stock name only)
3. TuShare Pro ``stock_basic`` (industry + stock name)
4. Existing CSV row (last resort)

The module exposes both a stateful ``IndustryEnricher`` class and module-level
convenience helpers ``refresh_metadata``, ``lookup`` and ``enrich``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.core.paths import STOCK_METADATA_CSV_PATH
from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.market_data_provider import (
    BaostockProvider,
    TushareProvider,
    _akshare_call,
)

logger = logging.getLogger(__name__)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_NONE = "none"

CSRC_TAXONOMY = "csrc_2012_l1"
CSRC_SOURCE = "baostock:query_stock_industry"
CSRC_SOURCE_PROVIDER = "baostock"
CSRC_SOURCE_ENDPOINT = "query_stock_industry"
CSRC_CONFIDENCE_METHOD = "exact_stock_code_and_taxonomy_parse"
DEFAULT_INDUSTRY_MIN_COVERAGE = 0.995
DEFAULT_INDUSTRY_MAX_COVERAGE_DROP = 0.0
DEFAULT_INDUSTRY_MAX_SOURCE_CONFLICTS = 0
DEFAULT_INDUSTRY_DASHBOARD_WARN_COVERAGE = 0.999
DEFAULT_INDUSTRY_MAX_SOURCE_AGE_DAYS = 14
CSRC_L1_NAMES = {
    "A": "农、林、牧、渔业",
    "B": "采矿业",
    "C": "制造业",
    "D": "电力、热力、燃气及水生产和供应业",
    "E": "建筑业",
    "F": "批发和零售业",
    "G": "交通运输、仓储和邮政业",
    "H": "住宿和餐饮业",
    "I": "信息传输、软件和信息技术服务业",
    "J": "金融业",
    "K": "房地产业",
    "L": "租赁和商务服务业",
    "M": "科学研究和技术服务业",
    "N": "水利、环境和公共设施管理业",
    "O": "居民服务、修理和其他服务业",
    "P": "教育",
    "Q": "卫生和社会工作",
    "R": "文化、体育和娱乐业",
    "S": "综合",
}
_VALID_STOCK_RE = re.compile(r"^\d{6}\.(?:SH|SZ)$")
_CSRC_INDUSTRY_RE = re.compile(r"^([A-S])(?:(\d{2}))?(.*)$")

# Columns required in the output metadata CSV.
_REQUIRED_COLUMNS = [
    "stock",
    "board",
    "stock_name",
    "industry",
    "industry_source",
    "industry_source_provider",
    "industry_source_endpoint",
    "industry_confidence",
    "industry_confidence_score",
    "industry_confidence_method",
    "industry_l1_code",
    "industry_l1_name",
    "industry_l2_code",
    "industry_l2_name",
    "industry_taxonomy",
    "industry_source_updated_at",
    "industry_source_fetched_at",
    "classification_eligible",
    "metadata_updated_at",
]


def _normalize_baostock_code(code: str) -> str:
    """Convert ``sh.600000``/``sz.000001`` into KHQuant stock codes."""
    text = str(code).strip().lower()
    match = re.fullmatch(r"(sh|sz)\.(\d{6})", text)
    if not match:
        return ""
    market, digits = match.groups()
    return f"{digits}.{market.upper()}"


def parse_csrc_industry(value: str) -> dict[str, str]:
    """Split a Baostock CSRC value such as ``J66货币金融服务``.

    The compatibility ``industry`` column uses the stable CSRC level-one
    section (A-S).  The more detailed source value is retained as level two so
    future screens can opt into it without mixing classification systems.
    """
    text = str(value or "").strip()
    match = _CSRC_INDUSTRY_RE.fullmatch(text)
    if not match:
        return {
            "industry_l1_code": "",
            "industry_l1_name": "",
            "industry_l2_code": "",
            "industry_l2_name": "",
        }
    level_one, digits, level_two_name = match.groups()
    level_one_name = CSRC_L1_NAMES.get(level_one, "")
    if not level_one_name:
        return {
            "industry_l1_code": "",
            "industry_l1_name": "",
            "industry_l2_code": "",
            "industry_l2_name": "",
        }
    return {
        "industry_l1_code": level_one,
        "industry_l1_name": level_one_name,
        "industry_l2_code": f"{level_one}{digits}" if digits else level_one,
        "industry_l2_name": str(level_two_name or "").strip(),
    }


def industry_quality_settings(config: dict[str, Any] | None = None) -> dict[str, float | int]:
    """Return validated industry refresh and dashboard quality thresholds."""
    source = config if config is not None else load_data_config()
    section = source.get("industry_classification", {}) if isinstance(source, dict) else {}
    settings: dict[str, float | int] = {
        "min_coverage": float(section.get("min_coverage", DEFAULT_INDUSTRY_MIN_COVERAGE)),
        "max_coverage_drop": float(
            section.get("max_coverage_drop", DEFAULT_INDUSTRY_MAX_COVERAGE_DROP)
        ),
        "max_source_conflicts": int(
            section.get("max_source_conflicts", DEFAULT_INDUSTRY_MAX_SOURCE_CONFLICTS)
        ),
        "dashboard_warn_coverage": float(
            section.get(
                "dashboard_warn_coverage",
                DEFAULT_INDUSTRY_DASHBOARD_WARN_COVERAGE,
            )
        ),
        "max_source_age_days": int(
            section.get("max_source_age_days", DEFAULT_INDUSTRY_MAX_SOURCE_AGE_DAYS)
        ),
    }
    if not 0 < float(settings["min_coverage"]) <= 1:
        raise ValueError("industry_classification.min_coverage must be in (0, 1]")
    if not 0 <= float(settings["max_coverage_drop"]) <= 1:
        raise ValueError("industry_classification.max_coverage_drop must be in [0, 1]")
    if int(settings["max_source_conflicts"]) < 0:
        raise ValueError("industry_classification.max_source_conflicts must be >= 0")
    if not 0 < float(settings["dashboard_warn_coverage"]) <= 1:
        raise ValueError("industry_classification.dashboard_warn_coverage must be in (0, 1]")
    if int(settings["max_source_age_days"]) < 0:
        raise ValueError("industry_classification.max_source_age_days must be >= 0")
    return settings


def _text_column(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str).str.strip()


def _text_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _prepare_csrc_source(snapshot: pd.DataFrame, refreshed_at: str) -> pd.DataFrame:
    source = snapshot.copy()
    source["stock"] = _text_column(source, "stock")
    defaults = {
        "industry_l1_code": "",
        "industry_l1_name": "",
        "industry_l2_code": "",
        "industry_l2_name": "",
        "industry_taxonomy": CSRC_TAXONOMY,
        "industry_source": CSRC_SOURCE,
        "industry_source_provider": CSRC_SOURCE_PROVIDER,
        "industry_source_endpoint": CSRC_SOURCE_ENDPOINT,
        "industry_source_updated_at": "",
        "industry_source_fetched_at": refreshed_at,
    }
    for column, default in defaults.items():
        source[column] = _text_column(source, column, default).replace("", default)
    return source


def audit_csrc_source_snapshot(snapshot: pd.DataFrame) -> dict[str, Any]:
    """Detect contradictory identities and classifications in one source snapshot.

    Identical duplicate rows are reported for observability but are not treated
    as conflicts. A conflicting provider/endpoint, taxonomy, L1 code/name pair,
    or multiple L1 assignments for the same stock is a blocking issue by
    default.
    """
    if snapshot.empty or "stock" not in snapshot.columns:
        return {
            "source_rows": int(len(snapshot)),
            "unique_stocks": 0,
            "duplicate_rows": 0,
            "conflict_count": 1,
            "conflict_stocks": 0,
            "conflicts": [{"stock": "", "type": "empty_source", "detail": "source snapshot is empty"}],
        }

    source = _prepare_csrc_source(snapshot, _now_text())
    conflicts: list[dict[str, str]] = []
    duplicate_rows = int(source["stock"].duplicated(keep=False).sum())

    for row in source.to_dict("records"):
        stock = str(row["stock"])
        provider = str(row["industry_source_provider"])
        endpoint = str(row["industry_source_endpoint"])
        composite = str(row["industry_source"])
        taxonomy = str(row["industry_taxonomy"])
        code = str(row["industry_l1_code"])
        name = str(row["industry_l1_name"])
        expected_name = CSRC_L1_NAMES.get(code, "") if code else ""
        if stock and not _VALID_STOCK_RE.fullmatch(stock):
            conflicts.append({"stock": stock, "type": "invalid_stock_code", "detail": stock})
        if (provider, endpoint, composite) != (
            CSRC_SOURCE_PROVIDER,
            CSRC_SOURCE_ENDPOINT,
            CSRC_SOURCE,
        ):
            conflicts.append(
                {
                    "stock": stock,
                    "type": "source_identity_mismatch",
                    "detail": f"{provider}:{endpoint}|{composite}",
                }
            )
        if taxonomy != CSRC_TAXONOMY:
            conflicts.append(
                {"stock": stock, "type": "taxonomy_mismatch", "detail": taxonomy}
            )
        if bool(code) != bool(name) or (code and (not expected_name or name != expected_name)):
            conflicts.append(
                {
                    "stock": stock,
                    "type": "l1_code_name_mismatch",
                    "detail": f"{code}|{name}",
                }
            )

    for stock, group in source[source["stock"] != ""].groupby("stock", sort=True):
        assignments = {
            (str(code), str(name))
            for code, name in zip(group["industry_l1_code"], group["industry_l1_name"])
            if str(code) or str(name)
        }
        identities = set(
            zip(
                group["industry_source_provider"],
                group["industry_source_endpoint"],
                group["industry_source"],
            )
        )
        taxonomies = set(group["industry_taxonomy"])
        if len(assignments) > 1:
            conflicts.append(
                {
                    "stock": stock,
                    "type": "duplicate_classification_conflict",
                    "detail": ";".join(f"{code}|{name}" for code, name in sorted(assignments)),
                }
            )
        if len(identities) > 1:
            conflicts.append(
                {
                    "stock": stock,
                    "type": "duplicate_source_identity_conflict",
                    "detail": ";".join(":".join(values) for values in sorted(identities)),
                }
            )
        if len(taxonomies) > 1:
            conflicts.append(
                {
                    "stock": stock,
                    "type": "duplicate_taxonomy_conflict",
                    "detail": ";".join(sorted(taxonomies)),
                }
            )

    return {
        "source_rows": int(len(source)),
        "unique_stocks": int(source.loc[source["stock"] != "", "stock"].nunique()),
        "duplicate_rows": duplicate_rows,
        "conflict_count": len(conflicts),
        "conflict_stocks": len({row["stock"] for row in conflicts if row["stock"]}),
        "conflicts": conflicts[:50],
    }


def fetch_baostock_industry_snapshot() -> pd.DataFrame:
    """Fetch the complete CSRC industry snapshot in one Baostock query."""
    try:
        import baostock as bs
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"baostock unavailable: {exc}") from exc

    login = bs.login()
    if getattr(login, "error_code", "1") != "0":
        raise RuntimeError(getattr(login, "error_msg", "baostock login failed"))
    try:
        result = bs.query_stock_industry()
        if result is None or getattr(result, "error_code", "1") != "0":
            raise RuntimeError(getattr(result, "error_msg", "baostock industry query failed"))
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()

    if frame.empty:
        raise RuntimeError("baostock industry snapshot is empty")
    required = {"updateDate", "code", "code_name", "industry", "industryClassification"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"baostock industry snapshot missing columns: {sorted(missing)}")

    frame = frame.copy()
    fetched_at = _now_text()
    frame["stock"] = frame["code"].map(_normalize_baostock_code)
    parsed = frame["industry"].map(parse_csrc_industry).apply(pd.Series)
    frame = pd.concat([frame, parsed], axis=1)
    frame["industry_taxonomy"] = CSRC_TAXONOMY
    frame["industry_source"] = CSRC_SOURCE
    frame["industry_source_provider"] = CSRC_SOURCE_PROVIDER
    frame["industry_source_endpoint"] = CSRC_SOURCE_ENDPOINT
    frame["industry_source_updated_at"] = frame["updateDate"].fillna("").astype(str).str.strip()
    frame["industry_source_fetched_at"] = fetched_at
    frame = frame[frame["stock"].map(lambda value: bool(_VALID_STOCK_RE.fullmatch(str(value))))]
    # Preserve duplicate source rows until the quality audit runs. Identical
    # duplicates are observable; contradictory duplicates block the refresh.
    return frame.reset_index(drop=True)


def apply_csrc_industry_snapshot(
    existing: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    stocks: list[str] | None = None,
    refreshed_at: str | None = None,
    min_coverage: float = DEFAULT_INDUSTRY_MIN_COVERAGE,
    max_coverage_drop: float = DEFAULT_INDUSTRY_MAX_COVERAGE_DROP,
    max_source_conflicts: int = DEFAULT_INDUSTRY_MAX_SOURCE_CONFLICTS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one taxonomy to existing metadata without adding new stocks.

    Stocks absent from ``existing`` are intentionally ignored.  This preserves
    the operator-approved metadata universe while allowing the raw market
    warehouse to retain additional historical or newly listed securities.
    """
    metadata = _prepare_metadata(existing)
    if metadata.empty:
        raise ValueError("stock metadata is empty")
    if snapshot.empty or "stock" not in snapshot.columns:
        raise ValueError("industry snapshot is empty or missing stock")
    if not 0 < float(min_coverage) <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0 <= float(max_coverage_drop) <= 1:
        raise ValueError("max_coverage_drop must be in [0, 1]")
    if int(max_source_conflicts) < 0:
        raise ValueError("max_source_conflicts must be >= 0")

    refreshed_at = refreshed_at or _now_text()
    source = _prepare_csrc_source(snapshot, refreshed_at)
    source_quality = audit_csrc_source_snapshot(source)
    target = metadata["stock"].map(lambda value: bool(_VALID_STOCK_RE.fullmatch(str(value))))
    if stocks:
        requested = {_normalize_code(stock) for stock in stocks}
        target &= metadata["stock"].isin(requested)
    target_stocks = metadata.loc[target, "stock"]
    if target_stocks.empty:
        return metadata, {
            "target_stocks": 0,
            "classified_stocks": 0,
            "coverage": 1.0,
            "excluded_metadata_rows": int((~target).sum()),
            "source_quality": source_quality,
        }

    source = (
        source.sort_values(
            ["stock", "industry_source_updated_at", "industry_source_fetched_at"],
            kind="stable",
        )
        .drop_duplicates("stock", keep="last")
        .set_index("stock")
    )
    target_source = source.reindex(target_stocks.tolist())
    eligible = target_source["industry_l1_name"].fillna("").astype(str).str.strip().ne("")
    coverage = float(eligible.mean())
    baseline_target = metadata.loc[target].set_index("stock")
    baseline_eligible = (
        baseline_target["classification_eligible"]
        .fillna("")
        .map(lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"})
    )
    baseline_coverage = float(baseline_eligible.mean()) if len(baseline_eligible) else 0.0
    coverage_drop = max(0.0, baseline_coverage - coverage)

    classification_changes: list[dict[str, str]] = []
    authority_conflicts: list[dict[str, str]] = []
    for stock in target_stocks:
        old_code = _text_value(baseline_target.at[stock, "industry_l1_code"])
        new_code = _text_value(target_source.at[stock, "industry_l1_code"])
        if not old_code or not new_code or old_code == new_code:
            continue
        old_effective = _text_value(
            baseline_target.at[stock, "industry_source_updated_at"]
        )
        new_effective = _text_value(target_source.at[stock, "industry_source_updated_at"])
        change = {
            "stock": stock,
            "old_l1_code": old_code,
            "new_l1_code": new_code,
            "old_effective_date": old_effective,
            "new_effective_date": new_effective,
        }
        classification_changes.append(change)
        if not new_effective or (old_effective and new_effective <= old_effective):
            authority_conflicts.append(
                {
                    "stock": stock,
                    "type": "authoritative_disagreement_without_newer_effective_date",
                    "detail": f"{old_code}@{old_effective}->{new_code}@{new_effective}",
                }
            )

    source_quality["classification_change_count"] = len(classification_changes)
    source_quality["classification_changes"] = classification_changes[:50]
    source_quality["authority_conflict_count"] = len(authority_conflicts)
    source_quality["conflict_count"] = int(source_quality["conflict_count"]) + len(
        authority_conflicts
    )
    source_quality["conflict_stocks"] = len(
        {
            str(row.get("stock", ""))
            for row in [*source_quality["conflicts"], *authority_conflicts]
            if str(row.get("stock", ""))
        }
    )
    source_quality["conflicts"] = [
        *source_quality["conflicts"],
        *authority_conflicts,
    ][:50]

    if coverage < float(min_coverage):
        raise ValueError(
            f"CSRC industry coverage {coverage:.2%} is below required {float(min_coverage):.2%}; metadata unchanged"
        )
    if coverage_drop > float(max_coverage_drop) + 1e-12:
        raise ValueError(
            "CSRC industry coverage regression "
            f"{coverage_drop:.2%} exceeds allowed {float(max_coverage_drop):.2%} "
            f"(baseline={baseline_coverage:.2%}, incoming={coverage:.2%}); metadata unchanged"
        )
    if int(source_quality["conflict_count"]) > int(max_source_conflicts):
        first = source_quality["conflicts"][0] if source_quality["conflicts"] else {}
        raise ValueError(
            "CSRC source conflicts "
            f"{source_quality['conflict_count']} exceed allowed {int(max_source_conflicts)}; "
            f"first={first}; metadata unchanged"
        )

    metadata = metadata.set_index("stock")
    target_index = target_stocks.tolist()
    mapped_index = target_source.index[eligible].tolist()
    unmapped_index = target_source.index[~eligible].tolist()

    classification_assignments = {
        "industry": "industry_l1_name",
        "industry_l1_code": "industry_l1_code",
        "industry_l1_name": "industry_l1_name",
        "industry_l2_code": "industry_l2_code",
        "industry_l2_name": "industry_l2_name",
    }
    for destination, source_column in classification_assignments.items():
        metadata.loc[mapped_index, destination] = target_source.loc[mapped_index, source_column].fillna("").values
    provenance_assignments = {
        "industry_taxonomy": "industry_taxonomy",
        "industry_source": "industry_source",
        "industry_source_provider": "industry_source_provider",
        "industry_source_endpoint": "industry_source_endpoint",
        "industry_source_updated_at": "industry_source_updated_at",
        "industry_source_fetched_at": "industry_source_fetched_at",
    }
    for destination, source_column in provenance_assignments.items():
        values = target_source[source_column].fillna("")
        fallback = {
            "industry_taxonomy": CSRC_TAXONOMY,
            "industry_source": CSRC_SOURCE,
            "industry_source_provider": CSRC_SOURCE_PROVIDER,
            "industry_source_endpoint": CSRC_SOURCE_ENDPOINT,
            "industry_source_fetched_at": refreshed_at,
        }.get(destination, "")
        metadata.loc[target_index, destination] = values.replace("", fallback).values
    metadata.loc[mapped_index, "industry_confidence"] = CONFIDENCE_HIGH
    metadata.loc[mapped_index, "industry_confidence_score"] = 1.0
    metadata.loc[mapped_index, "industry_confidence_method"] = CSRC_CONFIDENCE_METHOD
    metadata.loc[mapped_index, "classification_eligible"] = "true"
    metadata.loc[mapped_index, "neutral_group"] = metadata.loc[mapped_index, "industry_l1_name"]

    # Clear legacy mixed-taxonomy values for the few unmapped rows. They remain
    # visible in the audit file but cannot enter industry-level aggregation.
    clear_columns = [
        "industry",
        "industry_l1_code",
        "industry_l1_name",
        "industry_l2_code",
        "industry_l2_name",
    ]
    if unmapped_index:
        metadata.loc[unmapped_index, clear_columns] = ""
        metadata.loc[unmapped_index, "industry_confidence"] = CONFIDENCE_NONE
        metadata.loc[unmapped_index, "industry_confidence_score"] = 0.0
        metadata.loc[unmapped_index, "industry_confidence_method"] = "source_returned_no_classification"
        metadata.loc[unmapped_index, "classification_eligible"] = "false"
        metadata.loc[unmapped_index, "neutral_group"] = metadata.loc[unmapped_index, "board"]

    metadata.loc[target_index, "metadata_updated_at"] = refreshed_at
    invalid_index = metadata.index[~metadata.index.map(lambda value: bool(_VALID_STOCK_RE.fullmatch(str(value))))]
    if len(invalid_index):
        metadata.loc[invalid_index, "industry_source"] = "khquant:legacy_metadata_migration"
        metadata.loc[invalid_index, "industry_source_provider"] = "khquant"
        metadata.loc[invalid_index, "industry_source_endpoint"] = "legacy_metadata_migration"
        metadata.loc[invalid_index, "industry_confidence"] = CONFIDENCE_NONE
        metadata.loc[invalid_index, "industry_confidence_score"] = 0.0
        metadata.loc[invalid_index, "industry_confidence_method"] = "invalid_stock_code"
        metadata.loc[invalid_index, "industry_source_fetched_at"] = refreshed_at
        metadata.loc[invalid_index, "metadata_updated_at"] = refreshed_at
        metadata.loc[invalid_index, "classification_eligible"] = "false"

    updated = _prepare_metadata(metadata.reset_index())
    report = {
        "target_stocks": len(target_stocks),
        "classified_stocks": int(eligible.sum()),
        "coverage": coverage,
        "baseline_coverage": baseline_coverage,
        "coverage_drop": coverage_drop,
        "unclassified_target_stocks": int((~eligible).sum()),
        "excluded_metadata_rows": int((~target).sum() + (~eligible).sum()),
        "taxonomy": CSRC_TAXONOMY,
        "source": CSRC_SOURCE,
        "source_provider": CSRC_SOURCE_PROVIDER,
        "source_endpoint": CSRC_SOURCE_ENDPOINT,
        "source_rows": len(snapshot),
        "source_updated_at": str(snapshot.get("industry_source_updated_at", pd.Series(dtype=str)).max() or ""),
        "source_fetched_at": str(snapshot.get("industry_source_fetched_at", pd.Series(dtype=str)).max() or refreshed_at),
        "quality_gates": {
            "status": "passed",
            "min_coverage": float(min_coverage),
            "max_coverage_drop": float(max_coverage_drop),
            "max_source_conflicts": int(max_source_conflicts),
            "coverage_passed": coverage >= float(min_coverage),
            "coverage_regression_passed": coverage_drop <= float(max_coverage_drop) + 1e-12,
            "source_conflicts_passed": int(source_quality["conflict_count"])
            <= int(max_source_conflicts),
        },
        "source_quality": source_quality,
    }
    return updated, report


def build_industry_exclusions(raw_universe: set[str], metadata: pd.DataFrame) -> pd.DataFrame:
    """Build the complete current exclusion audit from one metadata snapshot."""
    prepared = _prepare_metadata(metadata)
    metadata_codes = set(prepared["stock"].fillna("").astype(str).str.strip())
    eligible_codes = set(
        prepared.loc[
            prepared["classification_eligible"].map(
                lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"}
            ),
            "stock",
        ]
    )
    valid_metadata = {
        stock for stock in metadata_codes if _VALID_STOCK_RE.fullmatch(str(stock))
    }
    rows = [
        {"stock": stock, "reason": "missing_metadata"}
        for stock in sorted(raw_universe - metadata_codes)
    ]
    rows.extend(
        {"stock": stock, "reason": "no_unified_industry"}
        for stock in sorted(valid_metadata - eligible_codes)
    )
    rows.extend(
        {"stock": stock, "reason": "invalid_stock_code"}
        for stock in sorted(metadata_codes - valid_metadata)
        if stock
    )
    return pd.DataFrame(rows, columns=["stock", "reason"])


def _normalize_code(code: str) -> str:
    """Return a canonical ``000001.SZ`` style code."""
    text = str(code).strip().upper().split(".")[0].zfill(6)
    suffix = "SH" if text.startswith("6") else "SZ"
    return f"{text}.{suffix}"


def _market_board(stock: str) -> str:
    """Map a normalized stock code to a board label."""
    code = stock.split(".")[0]
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith("688"):
        return "star"
    if code.startswith(("8", "4")):
        return "bse"
    if code.startswith(("600", "601", "603", "605", "6")):
        return "sh_main"
    if code.startswith(("000", "001", "002", "003")):
        return "sz_main"
    return "other"


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def _fetch_akshare_industry_map() -> dict[str, str]:
    """Build ``{stock: industry}`` from Eastmoney industry boards.

    This is the most complete AkShare source for bulk enrichment but it issues
    one request per industry board, so it is only used inside ``refresh_metadata``.
    """
    try:
        import akshare as ak
    except Exception as exc:
        logger.warning("akshare unavailable: %s", exc)
        return {}

    try:
        boards = _akshare_call(ak.stock_board_industry_name_em)
    except Exception as exc:
        logger.warning("failed to fetch AkShare industry board list: %s", exc)
        return {}

    if boards is None or boards.empty:
        return {}

    name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
    industry_map: dict[str, str] = {}
    unique_boards = boards[name_col].dropna().astype(str).unique()
    for idx, industry in enumerate(unique_boards, start=1):
        try:
            cons = _akshare_call(ak.stock_board_industry_cons_em, symbol=industry)
        except Exception as exc:
            logger.debug("AkShare industry board failed %s: %s", industry, exc)
            continue
        if cons is None or cons.empty:
            continue
        code_col = "代码" if "代码" in cons.columns else ("code" if "code" in cons.columns else None)
        if not code_col:
            continue
        for raw in cons[code_col].dropna().astype(str):
            norm = _normalize_code(raw)
            if norm not in industry_map:
                industry_map[norm] = industry
        if idx % 20 == 0:
            logger.info("AkShare industry board progress: %d/%d", idx, len(unique_boards))
    return industry_map


def _fetch_akshare_individual_info(stock: str) -> dict[str, str]:
    """Fetch name + industry for a single stock via ``stock_individual_info_em``.

    Returns empty dict when AkShare is missing or the endpoint does not include
    the expected keys.
    """
    try:
        import akshare as ak
    except Exception as exc:
        logger.debug("akshare unavailable: %s", exc)
        return {}

    symbol = stock.split(".")[0]
    try:
        df = _akshare_call(ak.stock_individual_info_em, symbol=symbol)
    except Exception as exc:
        logger.debug("AkShare individual info failed for %s: %s", stock, exc)
        return {}

    if df is None or df.empty or {"item", "value"} - set(df.columns):
        return {}

    mapping = {"股票简称": "stock_name", "行业": "industry"}
    out: dict[str, str] = {}
    for cn_key, en_key in mapping.items():
        rows = df[df["item"] == cn_key]
        if not rows.empty:
            value = str(rows["value"].iloc[0]).strip()
            if value and value not in {"-", "--", "nan", "None"}:
                out[en_key] = value
    return out


def _fetch_tushare_stock_info(stock: str, config: dict | None = None) -> dict[str, str]:
    """Fetch name + industry for a single stock via Tushare Pro ``stock_basic``.

    Tushare requires a configured token.  Returns empty dict on any failure so
    the caller can fall back to the next source.
    """
    try:
        provider = TushareProvider(config)
        pro = provider._pro_api()
    except Exception as exc:
        logger.debug("Tushare provider unavailable: %s", exc)
        return {}

    suffix = "SZ" if stock.endswith(".SZ") else "SH"
    ts_code = f"{stock.split('.')[0]}.{suffix}"
    try:
        df = pro.query(
            "stock_basic",
            ts_code=ts_code,
            fields="ts_code,name,industry",
        )
    except Exception as exc:
        logger.debug("Tushare query failed for %s: %s", stock, exc)
        return {}

    if df is None or df.empty:
        return {}

    row = df.iloc[0]
    out: dict[str, str] = {}
    name = str(row.get("name", "")).strip()
    industry = str(row.get("industry", "")).strip()
    if name:
        out["stock_name"] = name
    if industry:
        out["industry"] = industry
    return out


def _fetch_baostock_stock_info(stock: str) -> dict[str, str]:
    """Fetch stock name for a single stock via Baostock ``query_stock_basic``.

    Baostock does not expose industry classification, so this helper only fills
    ``stock_name``.  After a global login failure is detected, all subsequent
    calls in the same process are skipped to avoid spamming ``bs.login()``.
    """
    if BaostockProvider._is_global_failed():
        return {}
    try:
        import baostock as bs
    except Exception as exc:
        BaostockProvider._mark_global_failed(str(exc))
        logger.debug("baostock unavailable: %s", exc)
        return {}

    market = "sh" if stock.endswith(".SH") else "sz"
    bs_code = f"{market}.{stock.split('.')[0]}"
    try:
        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            error = getattr(lg, "error_msg", "baostock login failed")
            BaostockProvider._mark_global_failed(error)
            logger.debug("baostock login failed: %s", error)
            return {}
        try:
            rs = bs.query_stock_basic(code=bs_code)
            if rs is None or getattr(rs, "error_code", "0") != "0":
                return {}
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 2:
                    name = str(row[1]).strip()
                    if name:
                        return {"stock_name": name}
        finally:
            bs.logout()
    except Exception as exc:
        BaostockProvider._mark_global_failed(str(exc))
        logger.debug("baostock query failed for %s: %s", stock, exc)
    return {}


# ---------------------------------------------------------------------------
# Enricher class
# ---------------------------------------------------------------------------


class IndustryEnricher:
    """Stateful DB-first enricher with a compatibility CSV export.

    Passing an explicit ``metadata_path`` selects compatibility-file mode for
    isolated tests and legacy callers. Production defaults to raw SQLite.
    """

    def __init__(
        self,
        metadata_path: str | Path | None = None,
        config: dict | None = None,
    ):
        self.metadata_path = Path(metadata_path) if metadata_path else STOCK_METADATA_CSV_PATH
        self._sqlite_authority = metadata_path is None
        self.config = config or load_data_config()
        self._metadata_cache: pd.DataFrame | None = None
        self._last_storage_report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Metadata I/O
    # ------------------------------------------------------------------

    def _read_metadata(self) -> pd.DataFrame:
        if self._metadata_cache is not None:
            return self._metadata_cache

        if self._sqlite_authority:
            try:
                from my_strategy.storage.access_layer import get_data_access

                authoritative = get_data_access().read_industry_metadata()
                if not authoritative.empty:
                    df = _prepare_metadata(authoritative)
                    self._metadata_cache = df
                    return df
            except Exception as exc:
                logger.warning(
                    "failed to read authoritative industry SQLite: %s; trying CSV compatibility export",
                    exc,
                )

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.metadata_path.exists():
            df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
        else:
            try:
                df = pd.read_csv(self.metadata_path, dtype={"stock": str})
            except Exception as exc:
                logger.warning("failed to read %s: %s; starting fresh", self.metadata_path, exc)
                df = pd.DataFrame(columns=_REQUIRED_COLUMNS)

        df = _prepare_metadata(df)
        self._metadata_cache = df
        return df

    def _write_metadata(
        self,
        df: pd.DataFrame,
        *,
        run_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        df = _prepare_metadata(df)
        if self._sqlite_authority:
            from my_strategy.storage.access_layer import get_data_access

            storage = get_data_access()
            effective_run_id = run_id or f"industry-metadata-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
            exclusions = build_industry_exclusions(set(storage.list_stocks()), df)
            self._last_storage_report = storage.replace_industry_metadata(
                df,
                exclusions,
                run_id=effective_run_id,
                summary=summary or {"created_at": _now_text()},
            )
            df = _prepare_metadata(storage.read_industry_metadata())
            authoritative_exclusions = storage.read_industry_exclusions()
            if not authoritative_exclusions.empty:
                _atomic_write_csv(
                    authoritative_exclusions[["stock", "reason"]],
                    self.metadata_path.parent / "industry_exclusions.csv",
                )
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_csv(df, self.metadata_path)
        self._metadata_cache = df.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_metadata(self, stocks: list[str] | None = None, force: bool = False) -> pd.DataFrame:
        """Refresh metadata for ``stocks`` and persist to CSV.

        When ``stocks`` is ``None`` the method refreshes every stock already
        present in the metadata file.  ``force=True`` re-enriches rows that
        already have high-confidence data.

        Returns the updated metadata DataFrame.
        """
        existing = self._read_metadata()
        target_stocks: list[str]
        if stocks:
            target_stocks = [_normalize_code(s) for s in stocks]
        elif not existing.empty:
            target_stocks = existing["stock"].dropna().astype(str).unique().tolist()
        else:
            target_stocks = []

        if not target_stocks:
            return existing

        target_stocks = list(dict.fromkeys(target_stocks))  # preserve order, dedupe
        updated_now = _now_text()

        # Determine which stocks actually need refreshing.
        to_refresh = target_stocks if force else [
            s for s in target_stocks
            if self._needs_refresh(existing, s, updated_now)
        ]

        # Build source tables.
        akshare_map = _fetch_akshare_industry_map() if to_refresh else {}

        rows: list[dict[str, Any]] = []
        for stock in to_refresh:
            record = self._empty_record(stock, updated_now)

            # AkShare board map (industry + name if present).
            if akshare_map and stock in akshare_map:
                record["industry"] = akshare_map[stock]
                record["industry_source"] = "akshare:stock_board_industry_cons_em"
                record["industry_confidence"] = CONFIDENCE_HIGH

            # AkShare individual info fallback for missing name/industry.
            if not record["stock_name"] or not record["industry"]:
                info = _fetch_akshare_individual_info(stock)
                if info.get("stock_name"):
                    record["stock_name"] = info["stock_name"]
                    if not record["industry_source"]:
                        record["industry_source"] = "akshare:stock_individual_info_em"
                        record["industry_confidence"] = CONFIDENCE_MEDIUM
                if info.get("industry"):
                    record["industry"] = info["industry"]
                    record["industry_source"] = "akshare:stock_individual_info_em"
                    record["industry_confidence"] = CONFIDENCE_MEDIUM

            # Baostock (name only).
            if not record["stock_name"]:
                bs = _fetch_baostock_stock_info(stock)
                if bs.get("stock_name"):
                    record["stock_name"] = bs["stock_name"]
                    record["industry_source"] = record["industry_source"] or "baostock:query_stock_basic"
                    if record["industry_confidence"] == CONFIDENCE_HIGH:
                        record["industry_confidence"] = CONFIDENCE_MEDIUM

            # Tushare (name + industry).
            if not record["stock_name"] or not record["industry"]:
                ts = _fetch_tushare_stock_info(stock, self.config)
                if ts.get("stock_name"):
                    record["stock_name"] = ts["stock_name"]
                    record["industry_source"] = record["industry_source"] or "tushare:stock_basic"
                    if not record["industry_confidence"]:
                        record["industry_confidence"] = CONFIDENCE_MEDIUM
                if ts.get("industry"):
                    record["industry"] = ts["industry"]
                    record["industry_source"] = record["industry_source"] or "tushare:stock_basic"
                    if not record["industry_confidence"]:
                        record["industry_confidence"] = CONFIDENCE_MEDIUM

            # Existing CSV fallback (preserve stale values only if nothing better found).
            old = existing[existing["stock"] == stock]
            if not old.empty:
                old_row = old.iloc[0]
                if not record["stock_name"]:
                    record["stock_name"] = str(old_row.get("stock_name", "") or "")
                if not record["industry"]:
                    record["industry"] = str(old_row.get("industry", "") or "")
                    record["industry_source"] = str(old_row.get("industry_source", "") or "existing_csv")
                    record["industry_confidence"] = CONFIDENCE_LOW

            # Board is always deterministic.
            record["board"] = _market_board(stock)
            # Neutral group is preserved during the merge step below.
            rows.append(record)

        # Merge refreshed rows into existing metadata, preserving extra columns.
        if rows:
            refreshed = pd.DataFrame(rows)
            updated = _merge_into(existing, refreshed)
        else:
            updated = existing.copy()

        self._write_metadata(
            updated,
            summary={
                "created_at": updated_now,
                "source": "legacy_multi_source_enrichment",
                "metadata_rows": len(updated),
            },
        )
        return updated

    def refresh_csrc_industry(
        self,
        stocks: list[str] | None = None,
        *,
        min_coverage: float | None = None,
        max_coverage_drop: float | None = None,
        max_source_conflicts: int | None = None,
        run_id: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Refresh the unified CSRC level-one classification in one batch.

        Unlike :meth:`refresh_metadata`, this path never appends stocks that
        are absent from the existing metadata file.  It is therefore safe for
        the approved industry-analysis universe while raw market data remains
        untouched.
        """
        existing = self._read_metadata()
        snapshot = fetch_baostock_industry_snapshot()
        settings = industry_quality_settings(self.config)
        updated, report = apply_csrc_industry_snapshot(
            existing,
            snapshot,
            stocks=stocks,
            min_coverage=float(
                settings["min_coverage"] if min_coverage is None else min_coverage
            ),
            max_coverage_drop=float(
                settings["max_coverage_drop"]
                if max_coverage_drop is None
                else max_coverage_drop
            ),
            max_source_conflicts=int(
                settings["max_source_conflicts"]
                if max_source_conflicts is None
                else max_source_conflicts
            ),
        )
        self._write_metadata(
            updated,
            run_id=run_id,
            summary={"created_at": _now_text(), "classification": report},
        )
        report["sqlite"] = self._last_storage_report
        return updated, report

    def lookup(self, stock: str) -> dict[str, Any]:
        """Return the metadata row for ``stock``.

        If the stock is not present, ``enrich`` is called once and the result is
        cached in the CSV before being returned.
        """
        stock = _normalize_code(stock)
        df = self._read_metadata()
        row = df[df["stock"] == stock]
        if not row.empty:
            return row.iloc[0].to_dict()
        return self.enrich(stock)

    def enrich(self, stock: str) -> dict[str, Any]:
        """Fast single-stock enrichment for missing stocks.

        The method updates the metadata CSV with the result and returns a row
        dict containing at minimum ``stock``, ``board``, ``stock_name``,
        ``industry``, ``industry_source``, ``industry_confidence`` and
        ``metadata_updated_at``.
        """
        stock = _normalize_code(stock)
        updated_now = _now_text()
        record = self._empty_record(stock, updated_now)

        # 1. Existing CSV (fastest).
        existing = self._read_metadata()
        old = existing[existing["stock"] == stock]
        if not old.empty:
            old_row = old.iloc[0].to_dict()
            record["board"] = old_row.get("board") or _market_board(stock)
            record["stock_name"] = str(old_row.get("stock_name", "") or "")
            record["industry"] = str(old_row.get("industry", "") or "")
            record["industry_source"] = str(old_row.get("industry_source", "") or "existing_csv")
            record["industry_confidence"] = old_row.get("industry_confidence") or CONFIDENCE_LOW
            record["metadata_updated_at"] = str(old_row.get("metadata_updated_at", "") or updated_now)
            return record

        # 2. AkShare individual info (name + industry).
        if not record["stock_name"] or not record["industry"]:
            info = _fetch_akshare_individual_info(stock)
            if info.get("stock_name"):
                record["stock_name"] = info["stock_name"]
                if not record["industry"]:
                    record["industry_source"] = "akshare:stock_individual_info_em"
                    record["industry_confidence"] = CONFIDENCE_MEDIUM
            if info.get("industry"):
                record["industry"] = info["industry"]
                record["industry_source"] = "akshare:stock_individual_info_em"
                record["industry_confidence"] = CONFIDENCE_MEDIUM

        # 3. Baostock (name only).
        if not record["stock_name"]:
            bs = _fetch_baostock_stock_info(stock)
            if bs.get("stock_name"):
                record["stock_name"] = bs["stock_name"]
                record["industry_source"] = record["industry_source"] or "baostock:query_stock_basic"
                if not record["industry_confidence"]:
                    record["industry_confidence"] = CONFIDENCE_MEDIUM

        # 4. Tushare (name + industry).
        if not record["stock_name"] or not record["industry"]:
            ts = _fetch_tushare_stock_info(stock, self.config)
            if ts.get("stock_name"):
                record["stock_name"] = ts["stock_name"]
                record["industry_source"] = record["industry_source"] or "tushare:stock_basic"
                if not record["industry_confidence"]:
                    record["industry_confidence"] = CONFIDENCE_MEDIUM
            if ts.get("industry"):
                record["industry"] = ts["industry"]
                record["industry_source"] = record["industry_source"] or "tushare:stock_basic"
                if not record["industry_confidence"]:
                    record["industry_confidence"] = CONFIDENCE_MEDIUM

        record["board"] = _market_board(stock)

        # Cache the result in the metadata CSV.
        updated = _merge_into(existing, pd.DataFrame([record]))
        self._write_metadata(updated)
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_record(self, stock: str, updated_at: str) -> dict[str, Any]:
        return {
            "stock": stock,
            "board": _market_board(stock),
            "stock_name": "",
            "industry": "",
            "industry_source": "",
            "industry_confidence": "",
            "metadata_updated_at": updated_at,
        }

    def _needs_refresh(self, df: pd.DataFrame, stock: str, updated_at: str) -> bool:
        row = df[df["stock"] == stock]
        if row.empty:
            return True
        if str(row["industry"].iloc[0]).strip() == "":
            return True
        confidence = str(row["industry_confidence"].iloc[0]).strip() if "industry_confidence" in row.columns else ""
        if confidence == CONFIDENCE_LOW:
            return True
        # Re-check medium-confidence rows once per calendar day.
        last = str(row["metadata_updated_at"].iloc[0]).strip() if "metadata_updated_at" in row.columns else ""
        if last[:10] == updated_at[:10] and confidence == CONFIDENCE_HIGH:
            return False
        return True


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------


def refresh_metadata(
    stocks: list[str] | None = None,
    force: bool = False,
    metadata_path: str | Path | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Convenience wrapper around ``IndustryEnricher.refresh_metadata``."""
    return IndustryEnricher(
        metadata_path=metadata_path,
        config=config,
    ).refresh_metadata(stocks=stocks, force=force)


def lookup(stock: str, metadata_path: str | Path | None = None, config: dict | None = None) -> dict[str, Any]:
    """Convenience wrapper around ``IndustryEnricher.lookup``."""
    return IndustryEnricher(
        metadata_path=metadata_path,
        config=config,
    ).lookup(stock)


def enrich(stock: str, metadata_path: str | Path | None = None, config: dict | None = None) -> dict[str, Any]:
    """Convenience wrapper around ``IndustryEnricher.enrich``.

    The ``stock`` argument is a normalized or raw A-share code such as
    ``000001.SZ`` or ``000001``.
    """
    return IndustryEnricher(
        metadata_path=metadata_path,
        config=config,
    ).enrich(stock)


# ---------------------------------------------------------------------------
# Metadata frame utilities
# ---------------------------------------------------------------------------


def _prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure required columns exist."""
    if df.empty:
        return pd.DataFrame(columns=_REQUIRED_COLUMNS)

    # Strip whitespace and lowercase for lookup robustness.
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Normalize stock codes.
    if "stock" in df.columns:
        df["stock"] = df["stock"].astype(str).str.strip().apply(_normalize_code)
        df = df.drop_duplicates("stock", keep="last")

    # Ensure required columns exist.
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # CSV readers infer a fully populated true/false column as boolean. Keep
    # the persisted representation textual so repeat refreshes can safely
    # assign both eligibility states without pandas dtype conflicts.
    df["classification_eligible"] = (
        df["classification_eligible"].fillna("").astype(str).str.strip().str.lower()
    )
    df["industry_confidence_score"] = pd.to_numeric(
        df["industry_confidence_score"], errors="coerce"
    ).astype(float)

    # Reorder required columns first, preserve any extra columns after.
    extra = [c for c in df.columns if c not in _REQUIRED_COLUMNS]
    return df[[*_REQUIRED_COLUMNS, *extra]].reset_index(drop=True)


def _merge_into(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Overwrite ``existing`` with ``new_rows`` while preserving extra columns."""
    if existing.empty:
        return _prepare_metadata(new_rows)
    if new_rows.empty:
        return _prepare_metadata(existing)

    existing = _prepare_metadata(existing)
    new_rows = _prepare_metadata(new_rows)

    all_cols = list(dict.fromkeys([*existing.columns, *new_rows.columns]))
    existing = existing.reindex(columns=all_cols, fill_value="")
    new_rows = new_rows.reindex(columns=all_cols, fill_value="")

    # Use pandas update so existing-only columns survive for overlapping stocks,
    # and empty new values do not clobber existing values.
    existing_indexed = existing.set_index("stock")
    new_indexed = new_rows.set_index("stock").replace("", pd.NA)
    existing_indexed.update(new_indexed)

    # Append stocks that only exist in new_rows.
    only_new = new_rows[~new_rows["stock"].isin(existing_indexed.index)]
    combined = pd.concat([existing_indexed.reset_index(), only_new], ignore_index=True)

    # Ensure neutral_group stays filled if it existed.
    if "neutral_group" in all_cols:
        combined["neutral_group"] = combined["neutral_group"].replace("", pd.NA).fillna(
            combined["industry"].fillna("")
        ).fillna(combined["board"].fillna("")).fillna("未分类")

    return combined.drop_duplicates("stock", keep="last").reset_index(drop=True)


# Backwards-compatible alias used by some callers.
enrich_stock = enrich

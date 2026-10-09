#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build source-stock metadata independently of retired model training."""

from __future__ import annotations

from pathlib import Path
import sys
from io import StringIO

import pandas as pd
import requests


from my_strategy.core.paths import STOCK_METADATA_CSV_PATH

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from my_strategy.data_manager.industry_enricher import _market_board as market_board  # noqa: E402
from my_strategy.data_manager.stock_pool import normalize_stock_code  # noqa: E402
from my_strategy.storage.access_layer import get_data_access  # noqa: E402


OUTPUT_PATH = STOCK_METADATA_CSV_PATH
HF_SECTOR_URL = "https://huggingface.co/datasets/kjhq/China-Stock-Symbols-and-Metadata/resolve/main/china.csv"
HF_SECTOR_SOURCE = "kjhq/China-Stock-Symbols-and-Metadata"


def base_metadata() -> pd.DataFrame:
    storage = get_data_access()
    out = pd.DataFrame({"stock": storage.list_stocks()})
    out["stock"] = out["stock"].map(normalize_stock_code)
    out["board"] = out["stock"].map(market_board)
    basic = storage.read_stock_basic()
    if not basic.empty and "code" in basic:
        basic = basic.rename(columns={"code": "stock"})
        basic["stock"] = basic["stock"].map(normalize_stock_code)
        extra_cols = [column for column in ("stock_name", "total_market_cap", "turnover") if column in basic]
        out = out.merge(basic[["stock", *extra_cols]].drop_duplicates("stock"), on="stock", how="left")
    return out


def fetch_akshare_industry() -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as exc:
        print(f"warning: akshare unavailable: {exc}")
        return pd.DataFrame(columns=["stock", "industry"])

    try:
        boards = ak.stock_board_industry_name_em()
    except Exception as exc:
        print(f"warning: failed to fetch industry board list: {exc}")
        return pd.DataFrame(columns=["stock", "industry"])

    if boards.empty:
        return pd.DataFrame(columns=["stock", "industry"])

    name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
    rows = []
    for idx, industry in enumerate(boards[name_col].dropna().astype(str).unique(), start=1):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception as exc:
            print(f"warning: failed industry {industry}: {exc}")
            continue
        code_col = "代码" if "代码" in cons.columns else ("code" if "code" in cons.columns else None)
        name_col2 = "名称" if "名称" in cons.columns else ("stock_name" if "stock_name" in cons.columns else None)
        if not code_col:
            continue
        part = pd.DataFrame({"stock": cons[code_col].astype(str).map(normalize_stock_code), "industry": industry})
        if name_col2:
            part["stock_name_industry_source"] = cons[name_col2].astype(str)
        rows.append(part)
        if idx % 20 == 0:
            print(f"industry progress: {idx}/{len(boards)}")
    if not rows:
        return pd.DataFrame(columns=["stock", "industry"])
    industry_df = pd.concat(rows, ignore_index=True).drop_duplicates("stock", keep="first")
    return industry_df


def fetch_huggingface_sector() -> pd.DataFrame:
    """Fetch a broad public sector map when the primary industry source is unavailable."""
    try:
        response = requests.get(HF_SECTOR_URL, timeout=45)
        response.raise_for_status()
        source = pd.read_csv(StringIO(response.text))
    except Exception as exc:
        print(f"warning: failed to fetch Hugging Face sector metadata: {exc}")
        return pd.DataFrame(columns=["stock", "industry", "industry_source"])
    required = {"ticker", "market", "sector"}
    if not required.issubset(source.columns):
        print(f"warning: Hugging Face sector metadata missing columns: {sorted(required - set(source.columns))}")
        return pd.DataFrame(columns=["stock", "industry", "industry_source"])
    out = source[["ticker", "market", "sector"]].copy()
    out["ticker"] = pd.to_numeric(out["ticker"], errors="coerce")
    exchange = out["market"].astype(str).str.upper().map({"SSE": "SH", "SZSE": "SZ"})
    out["stock"] = out["ticker"].dropna().astype(int).astype(str).str.zfill(6) + "." + exchange
    out["industry"] = out["sector"].fillna("").astype(str).str.strip()
    out["industry_source"] = HF_SECTOR_SOURCE
    out = out[(out["stock"].str.endswith((".SH", ".SZ"))) & out["industry"].ne("")]
    return out[["stock", "industry", "industry_source"]].drop_duplicates("stock", keep="last")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    meta = base_metadata()
    industry = fetch_akshare_industry()
    source_name = "eastmoney_industry"
    if industry.empty:
        industry = fetch_huggingface_sector()
        source_name = HF_SECTOR_SOURCE
    if not industry.empty:
        meta = meta.merge(industry, on="stock", how="left")
        meta["neutral_group"] = meta["industry"].fillna(meta["board"])
        if "industry_source" not in meta.columns:
            meta["industry_source"] = source_name
        meta["industry_source"] = meta["industry_source"].fillna("")
    else:
        meta["industry"] = ""
        meta["neutral_group"] = meta["board"]
        meta["industry_source"] = ""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved: {OUTPUT_PATH}")
    print(f"stocks: {len(meta)}, industry filled: {int(meta['industry'].astype(str).str.len().gt(0).sum())}, source: {source_name}")


if __name__ == "__main__":
    main()

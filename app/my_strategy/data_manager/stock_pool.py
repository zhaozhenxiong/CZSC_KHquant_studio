"""Stock pool management for custom, all-A, and index pools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_data_config, project_path
from .local_cache import LocalCache
from my_strategy.storage.access_layer import get_data_access


def normalize_stock_code(code: str) -> str:
    code = str(code).strip().split(".")[0].zfill(6)
    suffix = "SH" if code.startswith("6") else "SZ"
    return f"{code}.{suffix}"


class StockPoolManager:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.cache = LocalCache(self.config)
        # Lazy import to break circular import: stock_pool <-> downloader <-> market_data_provider -> stock_pool
        from .downloader import DataDownloader
        self.downloader = DataDownloader(self.config)

    def ensure_default_custom_pool(self) -> Path:
        pool_path = project_path(self.config.get("stock_pool", {}).get("custom_pool_path", self.cache.custom_pool_dir / "my_pool.csv"))
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        if not pool_path.exists():
            default = pd.DataFrame(
                [
                    {"code": "002436.SZ", "name": "兴森科技", "market": "SZ", "remark": "测试股"},
                    {"code": "002585.SZ", "name": "双星新材", "market": "SZ", "remark": "自选股"},
                    {"code": "000767.SZ", "name": "晋控电力", "market": "SZ", "remark": "电力股"},
                    {"code": "600519.SH", "name": "贵州茅台", "market": "SH", "remark": "大盘蓝筹"},
                    {"code": "300750.SZ", "name": "宁德时代", "market": "SZ", "remark": "成长股"},
                ]
            )
            default.to_csv(pool_path, index=False, encoding="utf-8-sig")
        return pool_path

    def load_custom_pool(self) -> pd.DataFrame:
        pool_path = self.ensure_default_custom_pool()
        df = pd.read_csv(pool_path)
        code_col = "code" if "code" in df.columns else "stock"
        name_col = "name" if "name" in df.columns else "stock_name"
        out = pd.DataFrame()
        out["stock"] = df[code_col].map(normalize_stock_code)
        out["stock_name"] = df[name_col] if name_col in df.columns else ""
        out["source_pool"] = "custom"
        return out.drop_duplicates("stock").reset_index(drop=True)

    def update_all_a_pool(self, min_count: int = 3000) -> pd.DataFrame:
        result = self.downloader.download_stock_basic(min_count=min_count)
        if not result.ok or result.data is None or result.data.empty:
            raise RuntimeError(result.error or "all-A stock pool download failed")
        self.cache.save_stock_basic(result.data)
        return self.load_all_a_pool()

    def load_all_a_pool(self, auto_update: bool = False, min_count: int = 3000) -> pd.DataFrame:
        if auto_update:
            try:
                return self.update_all_a_pool(min_count=min_count)
            except Exception:
                pass
        basic = self.cache.read_stock_basic()
        if basic.empty:
            return self.load_db_pool(source_pool="db_fallback")
        name_col = "name" if "name" in basic.columns else "stock_name"
        out = pd.DataFrame()
        out["stock"] = basic["code"].map(normalize_stock_code)
        out["stock_name"] = basic[name_col] if name_col in basic.columns else ""
        out["source_pool"] = "all_a"
        for col in ["latest_price", "total_market_cap", "turnover"]:
            if col in basic.columns:
                out[col] = basic[col]
        return out.drop_duplicates("stock").reset_index(drop=True)

    def load_db_pool(self, source_pool: str = "db") -> pd.DataFrame:
        stocks = get_data_access().list_stocks()
        if not stocks:
            return pd.DataFrame(columns=["stock", "stock_name", "source_pool"])
        return pd.DataFrame(
            {
                "stock": [normalize_stock_code(stock) for stock in stocks],
                "stock_name": "",
                "source_pool": source_pool,
            }
        ).drop_duplicates("stock").reset_index(drop=True)

    def build_pool(self, mode: str | None = None, stocks: list[str] | None = None) -> pd.DataFrame:
        mode = mode or self.config.get("stock_pool", {}).get("default_mode", "custom")
        if stocks:
            custom = self.load_custom_pool()
            name_map = custom.set_index("stock")["stock_name"].to_dict() if not custom.empty else {}
            rows = [{"stock": normalize_stock_code(code), "stock_name": name_map.get(normalize_stock_code(code), ""), "source_pool": "custom"} for code in stocks]
            return pd.DataFrame(rows).drop_duplicates("stock").reset_index(drop=True)
        if mode == "all_a":
            return self.load_all_a_pool(auto_update=False)
        if mode in {"db", "local", "local_db"}:
            return self.load_db_pool(source_pool=mode)
        return self.load_custom_pool()

"""Public sector catalogs and full, paginated membership snapshots."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from importlib.resources import files
from typing import Any

import requests
from bs4 import BeautifulSoup
from py_mini_racer import MiniRacer

from my_strategy.core.paths import METADATA_ROOT

EASTMONEY_ENDPOINTS = tuple(
    f"https://{host}/api/qt/clist/get"
    for host in ("push2.eastmoney.com", "20.push2.eastmoney.com", "17.push2.eastmoney.com", "82.push2.eastmoney.com")
)
EASTMONEY_ENDPOINT = EASTMONEY_ENDPOINTS[0]
TONGHUASHUN_ENDPOINT = "https://q.10jqka.com.cn"


class TonghuashunAnonymousLimit(RuntimeError):
    pass


class EastmoneySectorProvider:
    def __init__(self, *, direct: bool = False):
        self.direct = direct
        self.session = requests.Session()
        self.session.trust_env = not direct
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})

    def close(self):
        self.session.close()

    def _page(self, fs: str, page: int) -> dict[str, Any]:
        # Large broad-market concepts span dozens of pages.  The public endpoint
        # can drop one connection mid-download, so retry the individual page
        # before discarding an otherwise complete snapshot.
        for attempt in range(8):
            try:
                if not self.direct:
                    self.session.trust_env = True
                response = self.session.get(EASTMONEY_ENDPOINTS[attempt % len(EASTMONEY_ENDPOINTS)], params={
                    "pn": page, "pz": 100, "np": 1, "fltt": 2,
                    "fs": fs, "fields": "f12,f13,f14", "fid": "f12", "po": 0,
                }, timeout=20)
                response.raise_for_status()
                payload = response.json()
                if payload.get("rc") != 0 or not isinstance(payload.get("data"), dict):
                    raise ValueError(f"invalid board response: filter={fs} page={page}")
                return payload["data"]
            except (requests.RequestException, ValueError):
                if attempt == 7:
                    raise
                time.sleep(min(attempt + 1, 4))
        raise AssertionError("unreachable")

    def rows(self, fs: str) -> list[dict]:
        rows = []
        expected = None
        page = 1
        while True:
            data = self._page(fs, page)
            total = int(data["total"])
            if expected is not None and expected != total:
                raise ValueError(f"membership changed during pagination: {fs}")
            expected = total
            chunk = data.get("diff") or []
            if isinstance(chunk, dict):
                chunk = list(chunk.values())
            rows.extend(chunk)
            if len(rows) >= total:
                break
            if not chunk:
                raise ValueError(f"truncated board pagination: {fs}")
            page += 1
            time.sleep(0.1)
        if len(rows) != expected or len({(x.get('f13'), x['f12']) for x in rows}) != len(rows):
            raise ValueError(f"duplicate or incomplete source rows: {fs}")
        return rows

    def catalog(self) -> list[dict]:
        boards = []
        for kind, number in [("industry", 2), ("concept", 3)]:
            rows = self.rows(f"m:90 t:{number} f:!50")
            if len(rows) < 20:
                raise ValueError(f"unexpectedly small {kind} catalog: {len(rows)}")
            for row in rows:
                code, name = str(row["f12"]), str(row["f14"]).strip()
                if not re.fullmatch(r"BK\d+", code) or not name:
                    raise ValueError("invalid board identity")
                boards.append({"code": code, "name": name, "kind": kind})
        if len({x['code'] for x in boards}) != len(boards):
            raise ValueError("overlapping source board identities")
        return boards

    def members(self, code: str) -> tuple[list[dict], int]:
        if not re.fullmatch(r"BK\d+", code):
            raise ValueError("invalid board code")
        raw = self.rows(f"b:{code}")
        members = []
        for row in raw:
            digits = str(row["f12"])
            if not re.fullmatch(r"\d{6}", digits):
                continue
            if digits.startswith(("4", "8", "92")):
                exchange = "BJ"
            elif digits.startswith("6") and int(row["f13"]) == 1:
                exchange = "SH"
            elif digits.startswith(("0", "3")) and int(row["f13"]) == 0:
                exchange = "SZ"
            else:
                continue
            members.append({"stock": f"{digits}.{exchange}", "name": str(row['f14']).strip()})
        return members, len(raw) - len(members)


class TonghuashunSectorProvider:
    """Read the public Tonghuashun industry and concept pages without iFinD."""

    _KINDS = {
        "I": ("industry", "thshy"),
        "C": ("concept", "gn"),
    }

    def __init__(self, *, direct: bool = False):
        self.direct = direct
        self._request_count = 0
        self.session = requests.Session()
        self.session.trust_env = not direct
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
        self._token_runtime = MiniRacer()
        self._token_runtime.eval(files("akshare.data").joinpath("ths.js").read_text(encoding="utf-8"))
        self._refresh_token()

    def close(self):
        self.session.close()

    def _refresh_token(self) -> None:
        token = self._token_runtime.eval("v()")
        self.session.headers["Cookie"] = f"v={token}"
        self.session.headers["hexin-v"] = token

    def _soup(self, path: str) -> BeautifulSoup:
        url = f"{TONGHUASHUN_ENDPOINT}/{path.lstrip('/')}"
        for attempt in range(5):
            try:
                if not self.direct:
                    self.session.trust_env = (self._request_count + attempt) % 2 == 0
                self._refresh_token()
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                text = response.content.decode("gbk", errors="replace")
                if "Nginx forbidden" in text or "chameleon.1.7" in text and len(text) < 2000:
                    raise ValueError("source anti-bot page returned")
                self._request_count += 1
                return BeautifulSoup(text, "html.parser")
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise
                time.sleep(1 + attempt * 2)
        raise AssertionError("unreachable")

    @staticmethod
    def _identity(kind_code: str, source_code: str) -> str:
        return f"THS_{kind_code}_{source_code}"

    def catalog(self) -> list[dict]:
        industry_page = self._soup("thshy/")
        industries: dict[str, str] = {}
        for anchor in industry_page.select('a[href*="/thshy/detail/code/"]'):
            match = re.search(r"/thshy/detail/code/(\d{6})", anchor.get("href", ""))
            name = anchor.get_text(" ", strip=True)
            if match and name:
                industries.setdefault(match.group(1), name)
                if industries[match.group(1)] != name:
                    raise ValueError(f"conflicting Tonghuashun industry name: {match.group(1)}")
        if not 70 <= len(industries) <= 150:
            raise ValueError(f"unexpected Tonghuashun industry catalog size: {len(industries)}")

        concept_page = self._soup("gn/")
        concepts: dict[str, str] = {}
        for anchor in concept_page.select('a[href*="/gn/detail/code/"]'):
            match = re.search(r"/gn/detail/code/(\d{6})", anchor.get("href", ""))
            name = anchor.get_text(" ", strip=True)
            if match and name:
                concepts.setdefault(match.group(1), name)
                if concepts[match.group(1)] != name:
                    raise ValueError(f"conflicting Tonghuashun concept name: {match.group(1)}")
        if not 250 <= len(concepts) <= 500:
            raise ValueError(f"unexpected Tonghuashun concept catalog size: {len(concepts)}")

        boards = [
            {"code": self._identity("I", code), "source_code": code, "name": name,
             "kind": "industry", "source": "tonghuashun_public"}
            for code, name in industries.items()
        ] + [
            {"code": self._identity("C", code), "source_code": code, "name": name,
             "kind": "concept", "source": "tonghuashun_public"}
            for code, name in concepts.items()
        ]
        if len({x["code"] for x in boards}) != len(boards):
            raise ValueError("overlapping Tonghuashun board identities")
        return boards

    @staticmethod
    def _page_info(soup: BeautifulSoup) -> tuple[int, int]:
        node = soup.select_one(".page_info")
        if node is None:
            return 1, 1
        match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", node.get_text() if node else "")
        if not match:
            raise ValueError("Tonghuashun membership pagination is missing")
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _member_rows(soup: BeautifulSoup) -> list[dict]:
        rows = []
        for tr in soup.select("#maincont table tbody tr, table.m-pager-table tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
            if len(cells) < 3 or not re.fullmatch(r"\d{6}", cells[1]):
                continue
            rows.append({"digits": cells[1], "name": cells[2]})
        return rows

    def members(self, code: str) -> tuple[list[dict], int]:
        match = re.fullmatch(r"THS_([IC])_(\d{6})", code)
        if not match:
            raise ValueError("invalid Tonghuashun board code")
        kind_code, source_code = match.groups()
        _, path_kind = self._KINDS[kind_code]
        first = self._soup(
            f"{path_kind}/detail/field/3475914/order/desc/page/1/ajax/1/code/{source_code}/"
        )
        current, expected_pages = self._page_info(first)
        if current != 1:
            raise ValueError(f"invalid Tonghuashun first membership page: {code}")
        if expected_pages > 10:
            raise TonghuashunAnonymousLimit(
                f"Tonghuashun public membership exceeds the 200-row anonymous limit: {code} pages={expected_pages}"
            )

        descending = self._member_rows(first)
        if not descending:
            raise ValueError(f"empty Tonghuashun membership page: {code} page=1")
        for page in range(2, min(expected_pages, 5) + 1):
            soup = self._soup(
                f"{path_kind}/detail/field/3475914/order/desc/page/{page}/ajax/1/code/{source_code}/"
            )
            current, total = self._page_info(soup)
            if current != page or total != expected_pages:
                raise ValueError(f"Tonghuashun membership changed during pagination: {code}")
            chunk = self._member_rows(soup)
            if not chunk:
                raise ValueError(f"empty Tonghuashun membership page: {code} page={page}")
            descending.extend(chunk)
            time.sleep(0.05)
        if len({x["digits"] for x in descending}) != len(descending):
            raise ValueError(f"duplicate Tonghuashun membership: {code}")

        raw_by_code = {row["digits"]: row for row in descending}
        ascending = []
        for page in range(1, max(0, expected_pages - 5) + 1):
            soup = self._soup(
                f"{path_kind}/detail/field/3475914/order/asc/page/{page}/ajax/1/code/{source_code}/"
            )
            current, total = self._page_info(soup)
            if current != page or total != expected_pages:
                raise ValueError(f"Tonghuashun membership changed during reverse pagination: {code}")
            chunk = self._member_rows(soup)
            if not chunk:
                raise ValueError(f"empty Tonghuashun reverse membership page: {code} page={page}")
            ascending.extend(chunk)
            raw_by_code.update((row["digits"], row) for row in chunk)
            time.sleep(0.05)
        if len({x["digits"] for x in ascending}) != len(ascending):
            raise ValueError(f"duplicate Tonghuashun reverse membership: {code}")
        raw = list(raw_by_code.values())
        if expected_pages > 5 and len(raw) <= (expected_pages - 1) * 20:
            raise ValueError(f"incomplete Tonghuashun membership union: {code}")

        members = []
        for row in raw:
            digits = row["digits"]
            if digits.startswith(("4", "8", "92")):
                exchange = "BJ"
            elif digits.startswith("6"):
                exchange = "SH"
            elif digits.startswith(("0", "3")):
                exchange = "SZ"
            else:
                continue
            members.append({"stock": f"{digits}.{exchange}", "name": row["name"]})
        return members, len(raw) - len(members)


class MixedPublicSectorProvider:
    """Tonghuashun industries plus Eastmoney concepts, with explicit provenance."""

    def __init__(self, *, direct: bool = False):
        self.ths = TonghuashunSectorProvider(direct=direct)
        self.eastmoney = EastmoneySectorProvider(direct=direct)
        self.catalog_metadata = {}
        self.catalog_cache_path = METADATA_ROOT / "sector_source_cache" / "eastmoney_directory.json"
        self.member_cache_dir = METADATA_ROOT / "sector_source_cache" / "eastmoney_members"

    def close(self):
        self.ths.close()
        self.eastmoney.close()

    def catalog(self) -> list[dict]:
        ths_industries = [b for b in self.ths.catalog() if b["kind"] == "industry"]
        try:
            eastmoney = self.eastmoney.catalog()
            self.catalog_cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.catalog_cache_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "boards": eastmoney,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.catalog_cache_path)
            self.catalog_metadata["eastmoney_directory"] = "live"
        except requests.RequestException:
            if not self.catalog_cache_path.exists():
                raise
            cached = json.loads(self.catalog_cache_path.read_text(encoding="utf-8"))
            observed = datetime.fromisoformat(cached["observed_at"])
            if datetime.now().astimezone() - observed > timedelta(days=7):
                raise ValueError("Eastmoney sector directory cache is older than 7 days")
            eastmoney = cached["boards"]
            self.catalog_metadata.update(
                eastmoney_directory="cached",
                eastmoney_directory_observed_at=cached["observed_at"],
            )
        industry_codes_by_name: dict[str, list[str]] = {}
        for board in eastmoney:
            if board["kind"] == "industry":
                industry_codes_by_name.setdefault(board["name"], []).append(board["code"])
        for board in ths_industries:
            matches = industry_codes_by_name.get(board["name"], [])
            if len(matches) == 1:
                board["fallback_code"] = matches[0]
        concepts = [
            {**board, "source_code": board["code"], "source": "eastmoney",
             "member_source": "eastmoney"}
            for board in eastmoney if board["kind"] == "concept"
        ]
        return ths_industries + concepts

    def _eastmoney_members(self, code: str) -> tuple[list[dict], int, str]:
        cache_path = self.member_cache_dir / f"{code}.json"
        try:
            members, excluded = self.eastmoney.members(code)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "members": members,
                "excluded_non_a": excluded,
            }, ensure_ascii=False), encoding="utf-8")
            temporary.replace(cache_path)
            return members, excluded, "eastmoney"
        except requests.RequestException:
            if not cache_path.exists():
                raise
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            observed = datetime.fromisoformat(cached["observed_at"])
            if datetime.now().astimezone() - observed > timedelta(days=7):
                raise ValueError(f"Eastmoney member cache is older than 7 days: {code}")
            return cached["members"], int(cached.get("excluded_non_a", 0)), "eastmoney_cached"

    @staticmethod
    def _margin_eligible_members() -> tuple[list[dict], int, str]:
        """Build the margin-trading concept from the public cross-market list."""
        url = "https://stock.pingan.com/fss/servlet/fsscoreapp/stockSource/mrgRatio"
        date = datetime.now().astimezone().strftime("%Y-%m-%d")
        rows = []
        excluded = 0
        for market in ("10", "00", "30"):
            response = requests.post(url, json={
                "currentPage": 1, "pageSize": 50000, "type": "bdzq", "setdate": date,
                "stockMes": "", "market": market, "appName": "AYLCH5", "tokenId": "",
                "appChannel": "LRSP", "requestId": "194055910e2075c03e25fabf6ffc5a7f", "channel": "pa18",
            }, timeout=30)
            response.raise_for_status()
            payload = response.json()
            source_rows = payload.get("data", {}).get("list")
            if not isinstance(source_rows, list):
                raise ValueError(f"invalid margin eligible list for market={market}")
            for item in source_rows:
                digits = str(item.get("secuCode", "")).zfill(6)
                if digits.startswith(("4", "8", "92")):
                    exchange = "BJ"
                elif digits.startswith("6"):
                    exchange = "SH"
                elif digits.startswith(("0", "3")):
                    exchange = "SZ"
                else:
                    excluded += 1
                    continue
                name = str(item.get("secuName", "")).strip()
                if not name:
                    raise ValueError(f"missing margin eligible name: {digits}")
                rows.append({"stock": f"{digits}.{exchange}", "name": name})
        members = {row["stock"]: row for row in rows}
        if len(members) != len(rows):
            raise ValueError("duplicate margin eligible stocks")
        if len(members) < 3000:
            raise ValueError(f"unexpectedly small margin eligible list: {len(members)}")
        return list(members.values()), excluded, "pingan_margin_eligible"

    def members_for(self, board: dict) -> tuple[list[dict], int, str]:
        if board["kind"] == "concept":
            try:
                return self._eastmoney_members(board["code"])
            except (requests.RequestException, ValueError):
                if board["code"] != "BK0596":
                    raise
                return self._margin_eligible_members()
        try:
            members, excluded = self.ths.members(board["code"])
            return members, excluded, "tonghuashun_public"
        except (TonghuashunAnonymousLimit, requests.RequestException, ValueError):
            fallback_code = board.get("fallback_code")
            if not fallback_code:
                raise
            members, excluded, source = self._eastmoney_members(fallback_code)
            suffix = "_cached" if source == "eastmoney_cached" else ""
            return members, excluded, f"eastmoney_exact_name_fallback{suffix}"


# Backward-compatible name used by existing scripts and tests.
ENDPOINT = EASTMONEY_ENDPOINT

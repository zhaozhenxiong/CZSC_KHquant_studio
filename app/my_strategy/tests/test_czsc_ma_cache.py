"""Shared runtime never relabels incompatible cached MA features as current."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_research as research
from my_strategy.services import czsc_research_features as extraction
from my_strategy.services import czsc_research_runtime as runtime
from my_strategy.services.czsc_research_profiles import bind_feature_profile, get_feature_profile


@pytest.fixture
def cache_case(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-10-06", periods=3)
    raw = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "open": 10.0,
                        "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100.0,
                        "amount": 1050.0, "source": "sina_unadjusted_v1", "has_trade_price": True})
    raw.attrs["data_version"] = "unchanged-input"
    profile = get_feature_profile("ma_trend_v1")
    fresh = raw.copy()
    for column in set(extraction.FEATURE_COLUMNS) | set(profile["columns"]):
        fresh[column] = .25
    fresh["input_eligible"] = True
    fresh["model_input_eligible"] = True
    fresh["reason_codes"] = [[] for _ in dates]
    fresh["model_reason_codes"] = [[] for _ in dates]
    for column in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell"):
        fresh[column] = False
    settings = runtime.strategy_config()
    quality_window = int(settings.get("research", {}).get("quality_window_bars", 120))
    fresh.attrs.update(feature_version=extraction.FEATURE_VERSION, schema_hash=extraction.FEATURE_SCHEMA_HASH,
                       feature_columns=list(extraction.FEATURE_COLUMNS), source_sha256=SOURCE_SHA256,
                       config_hash=stable_hash(settings), quality_window_bars=quality_window,
                       model_quality_window_bars=max(120, quality_window), model_minimum_active_bars=60)
    saved = bind_feature_profile(fresh, "ma_trend_v1")
    saved["ma5_bias"] = 777.0
    built = []
    monkeypatch.setattr(runtime, "load_bars", lambda *args, **kwargs: raw)
    def build(value):
        built.append(value)
        return fresh.copy()
    monkeypatch.setattr(extraction, "build_features", build)
    def persist(frame=saved, *, declared=True):
        path = tmp_path / "cache.parquet"
        frame.to_parquet(path, index=False)
        record = {"symbol": "000001.SZ", "path": str(path), "sha256": research._file_hash(path)}
        if declared:
            record.update(feature_profile=frame.attrs.get("feature_profile", "legacy"),
                          feature_schema_hash=frame.attrs["schema_hash"])
        return record
    def prepare(record, feature_profile="ma_trend_v1"):
        return runtime._prepare_stock("000001.SZ", "2026-10-08", dates.strftime("%Y-%m-%d").tolist(),
                                      record, feature_profile=feature_profile)
    return SimpleNamespace(raw=raw, saved=saved, fresh=fresh, profile=profile,
                           built=built, persist=persist, prepare=prepare)


def assert_rebuilt(case, result):
    assert not result["cache_hit"] and len(case.built) == 1
    assert result["features"]["ma5_bias"].eq(.25).all()
    assert result["features"].attrs["schema_hash"] == case.profile["schema_hash"]
    assert result["features"].attrs["feature_columns"] == case.profile["columns"]
    assert result["features"].attrs["model_quality_window_bars"] == case.fresh.attrs["model_quality_window_bars"]
    assert result["features"].attrs["model_minimum_active_bars"] == 60


@pytest.mark.parametrize("declared", [True, False])
def test_current_ma_cache_hits_with_optional_record_identity(cache_case, declared):
    result = cache_case.prepare(cache_case.persist(declared=declared))
    assert result["cache_hit"] and cache_case.built == []
    assert result["features"]["ma5_bias"].eq(777.0).all()


@pytest.mark.parametrize("attribute,change", [
    ("feature_profile", "legacy"), ("feature_version", "previous-ma-version"),
    ("schema_hash", "previous-ma-formula-hash"), ("feature_columns", "reversed"),
    ("feature_columns", None), ("feature_columns", "missing"),
])
def test_incompatible_ma_frame_identity_rebuilds_without_relabeling(cache_case, attribute, change):
    if change == "reversed":
        change = cache_case.profile["columns"][::-1]
    if change == "missing":
        cache_case.saved.attrs.pop(attribute)
    else:
        cache_case.saved.attrs[attribute] = change
    cache_case.saved.attrs["model_quality_window_bars"] = 60
    assert_rebuilt(cache_case, cache_case.prepare(cache_case.persist()))


@pytest.mark.parametrize("key,value", [("feature_profile", "legacy"), ("feature_schema_hash", "old-hash")])
def test_declared_record_identity_cannot_contradict_current_ma_frame(cache_case, key, value):
    record = cache_case.persist()
    record[key] = value
    assert_rebuilt(cache_case, cache_case.prepare(record))


@pytest.mark.parametrize("attribute,value", [
    ("model_quality_window_bars", 60), ("model_quality_window_bars", "missing"),
    ("model_minimum_active_bars", 30), ("model_minimum_active_bars", "missing"),
])
def test_ma_qualification_metadata_must_match_even_with_current_profile_identity(cache_case, attribute, value):
    if value == "missing":
        cache_case.saved.attrs.pop(attribute)
    else:
        cache_case.saved.attrs[attribute] = value
    assert_rebuilt(cache_case, cache_case.prepare(cache_case.persist()))


@pytest.mark.parametrize("cached_window", [120, 180])
def test_ma_window_tracks_larger_current_strategy_quality_window(cache_case, monkeypatch, cached_window):
    settings = deepcopy(runtime.strategy_config())
    settings.setdefault("research", {})["quality_window_bars"] = 180
    monkeypatch.setattr(runtime, "strategy_config", lambda: settings)
    for frame in (cache_case.saved, cache_case.fresh):
        frame.attrs.update(config_hash=stable_hash(settings), quality_window_bars=180)
    cache_case.saved.attrs["model_quality_window_bars"] = cached_window
    cache_case.fresh.attrs["model_quality_window_bars"] = 180
    result = cache_case.prepare(cache_case.persist())
    if cached_window == 180:
        assert result["cache_hit"] and cache_case.built == []
        assert result["features"]["ma5_bias"].eq(777.0).all()
    else:
        assert_rebuilt(cache_case, result)


def test_old_cache_without_model_input_eligibility_rebuilds(cache_case):
    saved = cache_case.saved.drop(columns="model_input_eligible")
    assert_rebuilt(cache_case, cache_case.prepare(cache_case.persist(saved)))


@pytest.mark.parametrize("attribute", ["base_feature_version", "base_schema_hash", "source_sha256", "config_hash"])
def test_incompatible_ma_base_semantics_rebuild_instead_of_failing_stock(cache_case, attribute):
    cache_case.saved.attrs[attribute] = "previous-extraction-identity"
    assert_rebuilt(cache_case, cache_case.prepare(cache_case.persist()))


def test_physical_file_hash_mismatch_still_rejects_before_reading(cache_case):
    record = cache_case.persist()
    with open(record["path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="文件哈希不匹配"):
        cache_case.prepare(record)
    assert cache_case.built == []


@pytest.mark.parametrize("feature_profile", [None, "legacy"])
def test_legacy_caller_without_profile_metadata_keeps_cache_hit(cache_case, feature_profile):
    saved = cache_case.fresh.drop(columns="model_input_eligible")
    saved["ma5_bias"] = 777.0
    result = cache_case.prepare(cache_case.persist(saved, declared=False), feature_profile)
    assert result["cache_hit"] and cache_case.built == []
    assert result["features"]["ma5_bias"].eq(777.0).all()
    assert result["features"].attrs["schema_hash"] == extraction.FEATURE_SCHEMA_HASH


def test_legacy_base_schema_rejection_is_preserved(cache_case):
    saved = cache_case.fresh.copy()
    saved.attrs["schema_hash"] = "previous-base-schema"
    with pytest.raises(ValueError, match="缓存特征语义"):
        cache_case.prepare(cache_case.persist(saved, declared=False), "legacy")
    assert cache_case.built == []


def test_matching_ma_identity_still_rebuilds_if_raw_bars_changed(cache_case):
    record = cache_case.persist()
    cache_case.raw.loc[0, "close"] = 10.6
    assert_rebuilt(cache_case, cache_case.prepare(record))

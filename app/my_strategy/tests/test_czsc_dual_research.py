from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_dual_models import (
    DualPredictorSession, DualResolver, expert_inputs, freeze_threshold, oof_splits,
    train_dual_bundle, verified_bundle,
)
from my_strategy.services.czsc_dual_research import prepare_frozen_dataset, round_trip_statistics
from my_strategy.services.czsc_research_ml import _file_hash, _hash
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS, get_feature_profile, prepare_structure_inputs


@pytest.fixture
def learning_frame():
    dates = pd.bdate_range("2023-01-02", "2025-06-30")
    x = np.sin(np.arange(len(dates)) / 7)
    frame = pd.DataFrame({name: x + i * .01 for i, name in enumerate([*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS])})
    frame["finished_bi_count"] = 6.
    frame["date"], frame["symbol"] = dates.strftime("%Y-%m-%d"), "000001.SZ"
    frame["available_at"] = [date.strftime("%Y-%m-%d") + "T15:00:00+08:00" for date in dates]
    frame["structure_confirmed_at"] = (dates - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    frame["zone_confirmed_at"] = frame["weekly_available_at"] = frame["monthly_available_at"] = None
    frame["label"], frame["net_return"] = (x > 0).astype(float), np.where(x > 0, .02, -.01)
    frame["label_end"] = (dates + pd.offsets.BDay(11)).strftime("%Y-%m-%d")
    frame["label_available"] = frame["input_eligible"] = frame["model_input_eligible"] = True
    frame["reason_codes"] = [[] for _ in dates]
    frame["model_reason_codes"] = [[] for _ in dates]
    frame.attrs = {"data_version": "fixture-frozen", "label_version": "czsc_fixed_horizon_broker_v1",
        "label_contract": {"label_version": "czsc_fixed_horizon_broker_v1", "horizon": 10, "initial_cash": 100000, "calendar_hash": "fixture"}}
    return frame


def test_pure_structure_profile_excludes_ma_and_ma_assisted_native_signals(learning_frame):
    profile = get_feature_profile("czsc_structure_v1")
    assert len(profile["columns"]) == len(set(profile["columns"])) == 22
    assert not set(profile["columns"]) & set(MA_TREND_COLUMNS)
    assert not any(name.startswith("native_") for name in profile["columns"])
    original = learning_frame.copy()
    result = prepare_structure_inputs(learning_frame)
    assert result["structure_input_eligible"].all()
    pd.testing.assert_frame_equal(learning_frame, original)
    assert result["input_eligible"].equals(original["input_eligible"])


def test_structure_requires_actual_confirmation_and_preserves_optional_missing(learning_frame):
    frame = learning_frame.iloc[:6].copy()
    frame.loc[0, "structure_confirmed_at"] = "2099-01-01"
    frame.loc[1, "finished_bi_count"] = 2
    frame.loc[2, "input_eligible"] = False
    frame.loc[2, "reason_codes"] = ["unverified_source_structural"]
    frame.loc[3, "zone_width_ratio"] = np.nan
    frame.loc[4, "weekly_available_at"] = "2099-01-01"
    frame.loc[5, "bi3_return"] = np.nan
    result = prepare_structure_inputs(frame)
    assert result["structure_input_eligible"].tolist() == [False, False, False, True, False, False]
    assert "structure_not_observable" in result.iloc[0]["structure_reason_codes"]
    assert "unverified_source_structural" in result.iloc[2]["structure_reason_codes"]
    assert "future_structure_metadata" in result.iloc[4]["structure_reason_codes"]
    assert np.isnan(expert_inputs(frame, "structure").iloc[3]["zone_width_ratio"])


def test_each_expert_projects_only_its_own_copy_qualification(learning_frame):
    frame = learning_frame.iloc[:2].copy()
    frame.loc[0, "model_input_eligible"] = False
    frame.loc[1, "input_eligible"] = False
    assert expert_inputs(frame, "ma")["input_eligible"].tolist() == [False, True]
    assert expert_inputs(frame, "structure")["input_eligible"].tolist() == [True, False]
    assert frame["input_eligible"].tolist() == [True, False]


def test_oof_quarters_have_disjoint_forward_availability():
    splits = oof_splits("2023-01-01", "2024-12-31")
    assert [s["name"] for s in splits] == ["2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4"]
    for split in splits:
        assert split["train_end"] < split["validation_start"] <= split["validation_end"] < split["test_start"] <= split["test_end"]


def test_threshold_is_costed_validation_proxy_and_keeps_unavailable_outcomes_out():
    frame = pd.DataFrame({"label_available": [True, True, False], "net_return": [.02, -.01, 100.]})
    result = freeze_threshold(frame, [.6, .51, .99], {"threshold_grid": [.5, .55], "threshold_min_rows": 1})
    assert result["threshold"] == .55
    assert result["scores"][1]["rows"] == 1
    assert result["scores"][1]["mean_net_return"] == .02
    assert "not_strategy_acceptance" in result["qualification"]


def test_complete_round_trips_use_actual_cash_fees_and_do_not_count_open_inventory():
    stats = round_trip_statistics([
        [{"action": "BUY", "shares": 100, "cash_flow": -1005}, {"action": "SELL", "shares": 50, "cash_flow": 550},
         {"action": "SELL", "shares": 50, "cash_flow": 550}],
        [{"action": "BUY", "shares": 100, "cash_flow": -1005}, {"action": "SELL", "shares": 100, "cash_flow": 905}],
        [{"action": "BUY", "shares": 100, "cash_flow": -1005}],
    ])
    assert stats["completed_round_trips"] == 2 and stats["win_rate"] == .5
    assert stats["unclosed_positions"] == 1 and stats["payoff_ratio"] == .95
    assert stats["trade_expectancy"] == pytest.approx(((1100 / 1005 - 1) + (905 / 1005 - 1)) / 2)


@pytest.fixture(scope="module")
def trained_pair(tmp_path_factory):
    # A real, small CPU training path exercises OOF, preprocessing and the stack.
    return tmp_path_factory.mktemp("dual-models")


def test_real_oof_stack_cutoffs_reuse_hashes_and_future_feature_guard(learning_frame, trained_pair):
    fold = {"train_end": "2024-12-31", "validation_start": "2025-01-01", "validation_end": "2025-03-31"}
    cfg = {"feature_start": "2023-01-01", "epochs": 1, "batch_size": 128, "min_train_rows": 20,
           "min_validation_rows": 20, "threshold_min_rows": 10, "seed": 42}
    first_path, second_path = trained_pair / "first", trained_pair / "second"
    first = train_dual_bundle(learning_frame, first_path, fold, device="cpu", config=cfg)
    assert first["status"] == "shadow" and first["publication_allowed"] is False
    oof = pd.read_parquet(first["oof"]["path"])
    assert (oof["expert_available_at"] < oof["date"]).all()
    assert (oof["target_label_end"] <= fold["train_end"]).all()
    assert first["oof"]["rows"] == len(oof)
    fusion = json.loads((first_path / "fusion/manifest.json").read_text())
    assert fusion["sample_counts"]["train"] == len(oof)
    future_changed = learning_frame.copy()
    future_changed.loc[future_changed["date"] > "2025-03-31", list(MA_TREND_COLUMNS)] = 1000000.
    second = train_dual_bundle(future_changed, second_path, fold, device="cpu", config=cfg)
    assert second["compute_info"]["trained_models"] == 3
    assert second["compute_info"]["reused_oof_blocks"] == 5
    for expert in ("ma", "structure"):
        a = json.loads((first_path / "experts" / expert / "manifest.json").read_text())
        b = json.loads((second_path / "experts" / expert / "manifest.json").read_text())
        assert a["preprocess"] == b["preprocess"]
        assert a["dataset_sha256"] == b["dataset_sha256"]
    session = DualPredictorSession(device="cpu")
    test = learning_frame.loc[learning_frame["date"] > "2025-03-31"].copy()
    values = session.predict(test, first_path, as_of="2025-06-30")
    assert np.isfinite(values).all() and set(session.last_expert_values) == {"ma", "structure", "fusion"}
    with pytest.raises(ValueError, match="future feature"):
        session.predict(test, first_path, as_of="2025-04-01")
    with pytest.raises(ValueError, match="unavailable"):
        session.predict(learning_frame.iloc[:1], first_path, as_of="2025-06-30")
    assert verified_bundle(first_path, check_oof=True)["bundle_sha256"] == first["bundle_sha256"]
    cached_rows = Path(second["oof"]["folds"][0]["rows_path"])
    original = cached_rows.read_bytes(); cached_rows.write_bytes(original + b"tamper")
    try:
        with pytest.raises(ValueError, match="cache integrity"):
            train_dual_bundle(learning_frame, trained_pair / "third", fold, device="cpu", config=cfg)
    finally:
        cached_rows.write_bytes(original)


def _source_fixture(frame):
    root = artifact_run_dir("dual-source-fixture")
    calendar = {"verified": True, "source": "fixture-calendar", "dates": frame["date"].tolist()}
    cp = root / "reports/calendar.json"; cp.parent.mkdir(exist_ok=True); cp.write_text(json.dumps(calendar))
    records = []
    for symbol in ("000001.SZ", "688001.SH"):
        values = frame.copy(); values["symbol"] = symbol
        for key, number in {"open": 10., "high": 10.2, "low": 9.8, "close": 10., "volume": 1000000.,
                            "amount": 10000000., "source": "sina_unadjusted_v1", "has_trade_price": 1}.items(): values[key] = number
        profile = get_feature_profile("ma_trend_v1")
        values.attrs.update(price_basis="unadjusted", config_hash=stable_hash(strategy_config()), data_version=symbol,
            feature_version=profile["version"], schema_hash=profile["schema_hash"], feature_columns=profile["columns"],
            feature_profile=profile["name"], calendar_hash=stable_hash(calendar["dates"]))
        path = root / (symbol.replace(".", "_") + ".parquet"); values.to_parquet(path, index=False)
        records.append({"symbol": symbol, "path": str(path), "sha256": _file_hash(path), "data_version": symbol,
                        "data_end": frame["date"].iloc[-1], "bars": len(frame)})
    contract = {"label_version": "czsc_fixed_horizon_broker_v1", "horizon": 10, "initial_cash": 100000,
                "calendar_hash": stable_hash(calendar["dates"])}
    source = {"data_end": frame["date"].iloc[-1], "dataset_records": records,
              "coverage": {"requested": 2, "success": 2, "failed": 0}, "model_manifest": {"label_contract": contract},
              "calendar": {"path": str(cp), "hash": stable_hash(calendar), "run_id": "dual-source-fixture", "source": "fixture-calendar"}}
    (root / "reports/research.json").write_text(json.dumps(source))
    return root, records


def test_frozen_source_subset_is_not_misreported_as_all_universe_and_unselected_hashes_checked(learning_frame):
    _, records = _source_fixture(learning_frame)
    context = create_run_context(task="dual-test", config={})
    cfg = {"horizon": 10, "initial_cash": 100000, "feature_start": "2023-01-01"}
    data, selected, snapshot, dates, calendar = prepare_frozen_dataset("dual-source-fixture", context, cfg, symbols=["000001.SZ"])
    assert len(selected) == 1 and snapshot["selected_count"] == 1
    assert snapshot["source_universe_count"] == 2 and not snapshot["selected_full_source_universe"]
    assert data.attrs["label_contract"]["calendar_hash"] == stable_hash(dates)
    assert calendar["path"] != calendar["source_path"]
    path = Path(records[1]["path"]); original = path.read_bytes(); path.write_bytes(original + b"tamper")
    try:
        with pytest.raises(ValueError, match="path/hash"):
            prepare_frozen_dataset("dual-source-fixture", create_run_context(task="dual-test"), cfg, symbols=["000001.SZ"])
    finally:
        path.write_bytes(original)


def test_dual_resolver_never_falls_into_legacy_active_alias_and_requires_pin():
    with pytest.raises(ValueError, match="explicit pin"):
        DualResolver(usage_mode="retrospective")
    route = DualResolver().resolve("2026-10-08")
    assert route["date"] == "2026-10-08" and route["applied_to_entry"] is False
    assert route["release_id"] is None


def _catalog_fixture(frame, monkeypatch):
    import my_strategy.services.czsc_dual_models as models
    _, records = _source_fixture(frame)
    context = create_run_context(task="dual-routing-test", stocks=["000001.SZ"])
    cfg = {"horizon": 10, "initial_cash": 100000, "feature_start": "2023-01-01"}
    data, selected, snapshot, dates, calendar = prepare_frozen_dataset("dual-source-fixture", context, cfg, symbols=["000001.SZ"])
    report = {"status": "complete", "publication_allowed": False, "source_snapshot": snapshot,
        "calendar": calendar, "market_dates": dates, "dataset_records": selected,
        "data_version": data.attrs["data_version"], "model_bundles": [{"name": "production", "path": str(context.subdir("models", "production"))}]}
    context.write_metadata({"status": "complete"})
    report_path = context.subdir("reports") / "dual-research.json"
    report_path.write_text(json.dumps(report))
    bundle = {"target": data.attrs["label_contract"], "data_version": data.attrs["data_version"],
        "target_hash": _hash(data.attrs["label_contract"]),
        "available_at": "2025-03-31", "trained_at": "2025-04-01T00:00:00+08:00",
        "probability_threshold": .55, "bundle_sha256": "fixture-binding"}
    monkeypatch.setattr(models, "verified_bundle", lambda *args, **kwargs: bundle)
    return context, report, report_path, records


def test_future_calendar_tampering_keeps_entire_run_cache_uncommitted(learning_frame, monkeypatch):
    context, report, path, _ = _catalog_fixture(learning_frame, monkeypatch)
    good = DualResolver(usage_mode="historical", model_run_id=context.run_id)
    assert good.catalog and good.cache and good.market_dates, good.catalog_errors
    report["market_dates"] = [*report["market_dates"], "2099-01-01"]
    path.write_text(json.dumps(report))
    bad = DualResolver(usage_mode="historical", model_run_id=context.run_id)
    assert not bad.catalog and not bad.cache and not bad.calendar and not bad.market_dates
    assert "calendar" in bad.catalog_errors[0]["error"]
    assert bad.resolve("2025-04-02")["applied_to_entry"] is False


def test_calendar_byte_hash_and_cached_feature_attrs_fail_closed(learning_frame, monkeypatch):
    context, report, _, records = _catalog_fixture(learning_frame, monkeypatch)
    calendar_path = Path(report["calendar"]["path"])
    original = calendar_path.read_bytes(); calendar_path.write_bytes(original + b"\n")
    bad = DualResolver(usage_mode="historical", model_run_id=context.run_id)
    assert not bad.cache and "file hash" in bad.catalog_errors[0]["error"]
    calendar_path.write_bytes(original)
    source_path = Path(records[0]["path"])
    source_original = source_path.read_bytes()
    frame = pd.read_parquet(source_path); frame.attrs["calendar_hash"] = "future-calendar"
    frame.to_parquet(source_path, index=False)
    try:
        bad = DualResolver(usage_mode="historical", model_run_id=context.run_id)
        assert not bad.catalog and not bad.cache
        assert "cached feature attrs" in bad.catalog_errors[0]["error"]
    finally:
        source_path.write_bytes(source_original)


def test_later_broken_bundle_does_not_commit_earlier_checkpoint_or_cache(learning_frame, monkeypatch):
    import my_strategy.services.czsc_dual_models as models
    context, report, path, _ = _catalog_fixture(learning_frame, monkeypatch)
    actual = models.verified_bundle
    report["model_bundles"].append({"name": "broken", "path": "broken"})
    path.write_text(json.dumps(report))
    def verifier(directory, **kwargs):
        if directory == "broken": raise ValueError("broken checkpoint")
        return actual(directory, **kwargs)
    monkeypatch.setattr(models, "verified_bundle", verifier)
    bad = DualResolver(usage_mode="historical", model_run_id=context.run_id)
    assert not bad.catalog and not bad.cache and bad.catalog_errors

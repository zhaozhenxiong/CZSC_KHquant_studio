"""Research arbitration cannot promote missing evidence or future models."""
import json
import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_research import adjudicate, run_path, _round_trips, _json


def evidence(**overrides):
    return {"symbol": "000001.SZ", "input_eligible": True, "point_confirmed": True, "point_type": "确认二买辅助",
            "rule_buy": True, "rule_sell": False, "reason_codes": np.array([], dtype=object), **overrides}


def test_risk_veto_beats_confirmed_point_and_high_probability():
    result = adjudicate(evidence(input_eligible=False, reason_codes=np.array(["unverified_source", "missing_amount"])), probability=.99, model_gate=True,
                        held=False, data_end="2026-09-30", requested_end="2026-09-30")
    assert result[0] == "excluded"
    assert result[3][-1]["judgment"] == "veto"
    assert set(result[2]) == {"unverified_source", "missing_amount"}


@pytest.mark.parametrize("overrides,gate", [({"point_confirmed": False}, True), ({}, False), ({"rule_buy": False}, True)])
def test_uncertain_or_shadow_model_cannot_confirm_buy(overrides, gate):
    category, _, _, _ = adjudicate(evidence(**overrides), probability=.99, model_gate=gate, held=False, data_end="2026-09-30", requested_end="2026-09-30")
    assert category == "watch"


def test_stale_date_is_explicit_exclusion_and_holdings_exit_is_separate():
    args = dict(probability=.99, model_gate=True, held=False, requested_end="2026-09-30")
    assert adjudicate(evidence(), data_end="2026-09-28", **args)[0] == "excluded"
    args["held"] = True
    assert adjudicate(evidence(rule_sell=True), data_end="2026-09-30", **args)[0] == "exit"


def test_completed_round_trip_requires_all_inventory_sold():
    ledger = [{"action": "BUY", "shares": 300, "cash_flow": -3005}, {"action": "SELL", "shares": 100, "cash_flow": 1095}]
    assert _round_trips(ledger) == (0, [])
    ledger.append({"action": "SELL", "shares": 200, "cash_flow": 2195})
    count, returns = _round_trips(ledger)
    assert count == 1 and returns[0] == pytest.approx(3290 / 3005 - 1)


def test_unverified_close_cannot_trigger_position_stop_loss():
    from my_strategy.services.czsc_research import rule_decisions
    dates = pd.bdate_range("2026-01-01", periods=70)
    raw = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "dt": dates + pd.Timedelta(hours=15), "id": np.arange(70), "close": 10.})
    raw.loc[62, "close"] = 8.
    features = pd.DataFrame({"input_eligible": True, "rule_buy": False, "rule_sell": False}, index=raw.index)
    features.loc[59, "rule_buy"] = True
    features.loc[62, "input_eligible"] = False
    decisions = rule_decisions(features, raw, "rules")
    assert decisions[59]["target_weight"] > 0
    assert decisions[62]["target_weight"] == decisions[61]["target_weight"]


@pytest.mark.parametrize("run_id", ["../x", "/x", "x/y", ".", ".."])
def test_research_reference_cannot_escape_runs(run_id):
    with pytest.raises(ValueError):
        run_path(run_id)


def test_saved_evidence_has_no_nonfinite_or_numpy_values():
    payload = _json({"probability": np.nan, "reasons": np.array(["unverified_source"]), "date": pd.Timestamp("2026-09-30")})
    assert payload["probability"] is None
    assert payload["reasons"] == ["unverified_source"]
    json.dumps(payload, allow_nan=False)


@pytest.fixture
def checkpoint_training(tmp_path, monkeypatch):
    from my_strategy.services import czsc_research as service
    from my_strategy.services import czsc_research_runtime as runtime, czsc_research_models as models
    root = tmp_path / "training"
    (root / "reports").mkdir(parents=True)
    folds = [{"name": "2025Q4", "test_start": "2025-10-01", "test_end": "2025-12-31"},
             {"name": "2026Q2", "test_start": "2026-04-01", "test_end": "2026-06-30"}]
    training = {"calendar": {"run_id": "verified"}, "model_gate": {"passed": False},
                "model_dir": str(tmp_path / "outside-production"),
                "model_manifest": {"available_at": "2025-01-01"},
                "config": {"probability_threshold": .55}, "evaluation": [{"fold": fold} for fold in folds]}
    (root / "reports/research.json").write_text(json.dumps(training), encoding="utf-8")
    for name, available in (("production", "2026-09-30"), ("2025Q4", "2025-09-30"), ("2026Q2", "2026-03-31")):
        directory = root / "models" / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"available_at": available, "validation_end": available,
                                                           "train_label_end": "2025-06-30"}), encoding="utf-8")
    monkeypatch.setattr(service, "run_path", lambda value: root)
    monkeypatch.setattr(service, "_calendar", lambda value: (["2025-09-29", "2025-09-30", "2026-01-05", "2026-09-30"], {"run_id": "verified"}))
    monkeypatch.setattr(runtime, "latest_market_date", lambda value: "2026-09-30")
    resolver_type = models.ModelResolver
    class FixtureResolver(resolver_type):
        def __init__(self, *, usage_mode, model_policy, model_run_id, checkpoint, **kwargs):
            if model_policy == "pinned" and (not model_run_id or not checkpoint):
                raise ValueError("pinned model requires run and checkpoint identities")
            self.usage_mode, self.model_policy = usage_mode, model_policy
            self.model_run_id, self.checkpoint = model_run_id, checkpoint
            self._releases, self._release_checks = [], {}
            self.catalog = [{"model_run_id": "training", "checkpoint": name, "model_dir": str((root / "models" / name).resolve()),
                             "available_at": available, "probability_threshold": .55, "manifest": {}}
                            for name, available in (("2025Q4", "2025-09-30"), ("2026Q2", "2026-03-31"), ("production", "2026-09-30"))]
    monkeypatch.setattr(models, "ModelResolver", FixtureResolver)
    return root, training


@pytest.mark.parametrize("fold,reason", [(None, "pinned model requires"), ("production", "固定检查点"),
                                        ("2026Q2", "固定检查点"), ("unlisted", "固定检查点"), ("../2025Q4", "固定检查点")])
def test_backtest_checkpoint_rejects_future_or_unknown_selection_before_run(checkpoint_training, monkeypatch, fold, reason):
    from my_strategy.services import czsc_research as service
    from my_strategy.services import czsc_research_runtime as runtime
    monkeypatch.setattr(runtime, "create_run_context", lambda **kwargs: pytest.fail("Rejected checkpoint created a run"))
    with pytest.raises(ValueError, match=reason):
        service.backtest_research(symbols=["600027.SH"], start="2026-01-01", end="2026-09-30", initial_cash=100000,
                                  model_run_id="training", model_fold=fold, model_policy="pinned", device="cpu")


def test_backtest_explicit_past_checkpoint_preserves_shadow_gate_and_prediction_dates(checkpoint_training, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from my_strategy.services import czsc_research as service, czsc_research_runtime as runtime, czsc_research_ml as ml
    root, training = checkpoint_training
    # A future global run gate cannot grant an earlier reconstructed model a release.
    training["model_gate"]["passed"] = True
    (root / "reports/research.json").write_text(json.dumps(training), encoding="utf-8")
    dates = pd.to_datetime(["2025-09-29", "2025-09-30", "2026-01-05"])
    raw = pd.DataFrame({"symbol": "600027.SH", "date": dates, "dt": dates + pd.Timedelta(hours=15),
                        "id": range(3), "close": 10.})
    raw.attrs["data_version"] = "frozen"
    features = pd.DataFrame({"symbol": "600027.SH", "date": dates.strftime("%Y-%m-%d"),
                            "input_eligible": True, "rule_buy": [False, True, False], "rule_sell": False,
                            "reason_codes": [[], [], []]})
    features.attrs.update(feature_version="fixture", schema_hash="fixture")
    metadata, calls = [], {}
    def subdir(*parts):
        path = tmp_path.joinpath("backtest", *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    context = SimpleNamespace(run_id="past-checkpoint", subdir=subdir, to_dict=lambda: {}, write_metadata=metadata.append)
    monkeypatch.setattr(runtime, "create_run_context", lambda **kwargs: context)
    monkeypatch.setattr(runtime, "_prepare_stock", lambda *args: {"symbol": "600027.SH", "raw": raw, "features": features,
                       "cache_hit": False, "data_end": "2026-01-05", "data_version": "frozen"})
    class Prediction:
        def __init__(self, **kwargs):
            pass
        def predict(self, frame, directory, **kwargs):
            calls.update(prediction_dates=frame.date.tolist(), directory=directory)
            return np.full(len(frame), .99)
        @property
        def diagnostics(self):
            return {"device": "cpu", "rows": 2, "batches": 1, "actual_cuda_inference": False}
    monkeypatch.setattr(ml, "PredictorSession", Prediction)
    def execute(symbol, raw, decisions, *args, **kwargs):
        calls["decisions"] = decisions
        return {
        "daily": [{"date": "2026-01-05", "equity": 100000}], "ledger": [], "trades": [], "rejections": []}
    monkeypatch.setattr(runtime, "execute_decisions", execute)
    result = service.backtest_research(symbols=["600027.SH"], start="2026-01-01", end="2026-09-30", initial_cash=100000,
                                       model_run_id="training", model_fold="2025Q4", model_policy="pinned", device="cpu", cpu_workers=1)
    assert calls["directory"] == str((root / "models/2025Q4").resolve())
    assert calls["prediction_dates"] == ["2025-09-30", "2026-01-05"]
    decisions = calls["decisions"]
    assert decisions[0]["probability"] is None and all(row["probability"] == .99 for row in decisions[1:])
    assert not any(row["ml_filter_applied"] for row in decisions) and result["model_gate"]["passed"] is False
    assert decisions[1]["position_intent"] == "BUY"  # combined rules remain executable in shadow
    assert result["model_fold"] == metadata[-1]["request"]["model_fold"] == "2025Q4"
    assert result["model_routes"][-1]["identity"]["available_at"] == "2025-09-30"


def test_research_status_lists_only_saved_checkpoint_dates(checkpoint_training, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from my_strategy.services import czsc_research as service
    from my_strategy.storage import czsc_results
    monkeypatch.setattr(service, "ARTIFACT_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(czsc_results, "ResultStore", lambda: SimpleNamespace(list=lambda **kwargs: [
        {"run_id": "training", "created_at": "2026-10-01", "summary": {"model_gate": {"passed": False}}}]))
    checkpoints = service.research_status()["models"][0]["checkpoints"]
    assert [(item["name"], item["available_at"]) for item in checkpoints] == [
        ("production", "2026-09-30"), ("2025Q4", "2025-09-30"), ("2026Q2", "2026-03-31")]
    assert checkpoints[0]["kind"] == "production" and checkpoints[0]["test_start"] is None
    assert checkpoints[1]["kind"] == "evaluation" and checkpoints[1]["test_start"] == "2025-10-01"


def test_dataset_labels_train_and_executable_comparison_integrate(tmp_path, monkeypatch):
    """Exercise the real feature/label/network/Position/broker interfaces together."""
    from types import SimpleNamespace
    from my_strategy.services import czsc_research as service
    from my_strategy.services.czsc_research_features import FEATURE_COLUMNS
    from my_strategy.services.czsc_research_ml import train_model
    dates = pd.bdate_range("2023-01-02", periods=700)
    close = 20 + np.arange(len(dates)) * .002 + np.sin(np.arange(len(dates)) / 13)
    raw = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "dt": dates + pd.Timedelta(hours=15), "id": np.arange(len(dates)),
                        "open": close, "high": close + .2, "low": close - .2, "close": close + .02,
                        "volume": 1_000_000., "amount": close * 1_000_000, "source": "tushare", "has_trade_price": 1})
    raw.attrs.update(data_version="fixture-snapshot", price_basis="unadjusted")
    def load(symbol, end=None, **kwargs):
        result = raw[raw.date <= pd.Timestamp(end)].copy() if end else raw.copy()
        result.attrs = raw.attrs.copy()
        return result
    monkeypatch.setattr(service, "load_bars", load)
    from my_strategy.services import czsc_research_evaluation
    monkeypatch.setattr(czsc_research_evaluation, "load_bars", load)
    def subdir(*parts):
        path = tmp_path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    context = SimpleNamespace(run_id="fixture-research", subdir=subdir)
    config = {"horizon": 10, "initial_cash": 100000, "feature_start": "2023-01-01", "probability_threshold": .55}
    data, records, failures = service.build_dataset(["000001.SZ"], dates[-1].date().isoformat(), context, config,
                                                    dates.strftime("%Y-%m-%d").tolist(), cpu_workers=1)
    assert not failures and len(data) > 100 and data.label_available.all()
    directory = tmp_path / "model"
    train_end = dates[420].date().isoformat()
    validation_start, validation_end = dates[421].date().isoformat(), dates[520].date().isoformat()
    train_model(data, FEATURE_COLUMNS, directory, train_end, validation_start, validation_end, device="cpu", epochs=1)
    fold = {"name": "fixture", "test_start": dates[521].date().isoformat(), "test_end": dates[-1].date().isoformat()}
    result = service.evaluate_fold(records, directory, fold, context, config, device="cpu", requested_symbols=["000001.SZ", "000002.SZ"], market_dates=dates.strftime("%Y-%m-%d").tolist())
    assert result["frozen_mainboard_accounts"] == 2
    assert result["unavailable_accounts_cash_retained"] == [{"symbol": "000002.SZ", "reason": "dataset_preparation_failed_cash_retained"}]
    assert set(result["variants"]) == {"czsc", "price_volume", "czsc_price_volume", "ml"}
    assert all(item["accounts"] == 1 for item in result["variants"].values())
    for mode in result["variants"]:
        daily = pd.read_csv(tmp_path / "evaluation" / "fixture" / mode / "000001_SZ" / "daily.csv")
        assert np.allclose(daily.cash + daily.shares * daily.close, daily.equity)
        assert result["variants"][mode]["metrics"]["final_equity"] == pytest.approx(daily.equity.iloc[-1] + 100000)


def test_research_execution_rejects_calendar_gap_and_ineligible_persistent_target():
    from my_strategy.services.czsc_backtest import execute_decisions
    from my_strategy.services.czsc_analysis import strategy_config
    dates = pd.bdate_range("2026-01-01", periods=65)
    raw = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "dt": dates + pd.Timedelta(hours=15), "id": range(65),
                        "open": 10., "high": 10.1, "low": 9.9, "close": 10., "volume": 1_000_000., "amount": 10_000_000., "source": "tushare", "has_trade_price": 1})
    raw = raw.drop(index=61).reset_index(drop=True)
    decisions = [{"date": d.date().isoformat(), "target_weight": .1 if d >= dates[60] else 0., "eligible": d != dates[62]} for d in raw.date]
    result = execute_decisions("000001.SZ", raw, decisions, dates[60].date().isoformat(), 100000, strategy_config(), "fixture",
                               market_dates=dates.strftime("%Y-%m-%d").tolist(), verified_sources={"tushare"})
    rejected = {(r["date"], r["reason"]) for r in result["rejections"]}
    assert (dates[62].date().isoformat(), "missing_next_market_bar") in rejected
    assert (dates[63].date().isoformat(), "research_signal_input_ineligible") in rejected
    assert all(r["date"] >= dates[64].date().isoformat() for r in result["ledger"])


def test_calendar_gap_veto_preserves_rows_and_prefix():
    from my_strategy.services.czsc_research import _calendar_quality
    dates = pd.bdate_range("2026-01-01", periods=8)
    raw = pd.DataFrame({"date": dates.delete(3)})
    features = pd.DataFrame({"input_eligible": True, "reason_codes": [[] for _ in raw.index],
                             **{k: True for k in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell")}})
    features.attrs["quality_window_bars"] = 2
    result = _calendar_quality(features, raw, dates.strftime("%Y-%m-%d").tolist())
    assert len(result) == len(raw)
    assert result.input_eligible.tolist() == [True, True, True, False, False, True, True]
    assert result.iloc[3].reason_codes == ["calendar_missing_market_bar_window"]


@pytest.mark.parametrize("actual_date,eligible,reason", [
    ("2026-09-29", True, "missing_target_date"),
    ("2026-09-30", False, "unverified_source"),
])
def test_scan_hard_veto_precedes_model_inference(tmp_path, monkeypatch, actual_date, eligible, reason):
    from types import SimpleNamespace
    from my_strategy.services import czsc_research as service, czsc_research_runtime as runtime, czsc_research_ml as ml, czsc_research_models as models

    training_root = tmp_path / "training"
    (training_root / "reports").mkdir(parents=True)
    training = {"data_end": "2026-09-30", "calendar": {"run_id": "verified"}, "dataset_records": [],
                "model_dir": str(training_root / "models/production"), "model_gate": {"passed": True},
                "model_manifest": {"train_label_end": "2026-06-30", "available_at": "2026-09-30"},
                "config": {"probability_threshold": .55}}
    (training_root / "reports/research.json").write_text(json.dumps(training), encoding="utf-8")
    raw = pd.DataFrame({"symbol": ["000001.SZ"], "date": pd.to_datetime([actual_date]),
                        "dt": pd.to_datetime([actual_date]) + pd.Timedelta(hours=15), "id": [0], "close": [10.]})
    raw.attrs["data_version"] = "frozen"
    values = {**evidence(input_eligible=eligible, reason_codes=[] if eligible else [reason]), "date": actual_date,
              "available_at": actual_date + "T15:00:00+08:00"}
    features = pd.DataFrame([values])
    def subdir(*parts):
        path = tmp_path.joinpath("screen", *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path
    context = SimpleNamespace(run_id="scan-hard-veto", subdir=subdir, to_dict=lambda: {}, write_metadata=lambda value: None)
    monkeypatch.setattr(runtime, "create_run_context", lambda **kwargs: context)
    monkeypatch.setattr(service, "run_path", lambda value: training_root)
    monkeypatch.setattr(service, "_calendar", lambda value: (["2026-09-29", "2026-09-30", "2026-10-08"], {"run_id": "verified"}))
    monkeypatch.setattr(runtime, "latest_market_date", lambda value: "2026-09-30")
    monkeypatch.setattr(runtime, "_prepare_stock", lambda *args: {"symbol": "000001.SZ", "raw": raw, "features": features,
                       "cache_hit": False, "data_end": actual_date, "data_version": "frozen"})
    class Resolver:
        def __init__(self, **kwargs):
            pass
        def resolve_dates(self, dates):
            return [{"date": str(day), "model_dir": "fixture-model" if str(day) >= "2026-09-30" else None,
                     "probability_threshold": .55, "applied_to_entry": False} for day in dates]
    class Prediction:
        def __init__(self, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            pytest.fail("Hard-vetoed stock reached model inference")
        @property
        def diagnostics(self):
            return {"rows": 0, "batches": 0, "actual_cuda_inference": False}
    monkeypatch.setattr(models, "ModelResolver", Resolver)
    monkeypatch.setattr(ml, "PredictorSession", Prediction)
    result = service.scan_research(end="2026-09-30", symbols=["000001.SZ"], model_run_id="training", cpu_workers=1)
    assert result["coverage"] == {"requested": 1, "success": 1, "failed": 0}
    assert result["rows"][0]["category"] == "excluded" and reason in result["rows"][0]["reason_codes"]
    assert result["rows"][0]["model_probability"] is None
    assert result["compute_info"]["model_inference_rows"] == 0 and not result["compute_info"]["cuda_work"]

"""Real MA predictions remain separate from continuous rule Position state."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from my_strategy.services.czsc_research import _calendar_quality
from my_strategy.services.czsc_research_ml import PredictorSession, train_model
from my_strategy.services.czsc_research_profiles import get_feature_profile
from my_strategy.services.czsc_research_runtime import ResearchRuntime, _replay_stock


def test_ma_calendar_window_does_not_inherit_shorter_rule_window():
    calendar = pd.bdate_range("2024-01-01", periods=220).strftime("%Y-%m-%d").tolist()
    dates = calendar[:30] + calendar[31:]
    raw = pd.DataFrame({"date": dates})
    features = pd.DataFrame({"input_eligible": True, "model_input_eligible": True,
                             "reason_codes": [[] for _ in dates], "model_reason_codes": [[] for _ in dates]})
    for key in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell"):
        features[key] = True
    features.attrs.update(quality_window_bars=60, model_quality_window_bars=120)
    checked = _calendar_quality(features, raw, calendar)
    assert checked.iloc[100]["input_eligible"] and not checked.iloc[100]["model_input_eligible"]
    assert checked.iloc[151]["model_input_eligible"]
    assert len(checked) == len(dates) and features["model_input_eligible"].all()


def test_real_ma_predictions_switch_checkpoints_without_resetting_position(tmp_path):
    profile = get_feature_profile("ma_trend_v1")
    dates = pd.bdate_range("2020-01-02", periods=240)
    x = np.sin(np.arange(len(dates)) / 5)
    learning = pd.DataFrame({column: x + index / 100 for index, column in enumerate(profile["columns"])})
    learning = learning.assign(symbol="000001.SZ", date=dates, label=(x > 0).astype(float),
                               label_end=dates + pd.offsets.BDay(11), label_available=True, input_eligible=True)
    learning.attrs.update(feature_version=profile["version"], schema_hash=profile["schema_hash"])
    for name, train_end, validation_start, validation_end in (
        ("A", "2020-06-30", "2020-07-01", "2020-08-31"),
        ("B", "2020-07-31", "2020-08-01", "2020-09-30"),
    ):
        train_model(learning, profile["columns"], tmp_path / name, train_end, validation_start, validation_end,
                    device="cpu", epochs=2, hidden_sizes=(8, 4))
    trace_dates = pd.bdate_range("2020-10-01", periods=5)
    features = pd.DataFrame({column: np.linspace(-.1, .1, 5) for column in profile["columns"]})
    features = features.assign(symbol="000001.SZ", date=trace_dates, source="tushare",
                               input_eligible=[True, True, False, True, True],
                               model_input_eligible=[True, True, True, False, True],
                               rule_buy=[False, True, False, False, False], rule_sell=[False, False, False, False, True],
                               rule_price_volume_buy=False, rule_price_volume_sell=False)
    features["reason_codes"] = [[] for _ in trace_dates]
    features["model_reason_codes"] = [[] for _ in trace_dates]
    features.attrs.update(feature_version=profile["version"], schema_hash=profile["schema_hash"])
    raw = pd.DataFrame({"symbol": "000001.SZ", "date": trace_dates, "dt": trace_dates + pd.Timedelta(hours=15),
                        "id": range(5), "close": 10.0})
    def resolve_dates(days):
        return [{"date": str(day)[:10], "model_dir": None if i == 0 else str(tmp_path / ("A" if i < 3 else "B")),
                 "checkpoint": None if i == 0 else "A" if i < 3 else "B", "applied_to_entry": False,
                 "status": "rules_no_model" if i == 0 else "historical_shadow", "probability_threshold": .55}
                for i, day in enumerate(days)]
    runtime = ResearchRuntime.__new__(ResearchRuntime)
    runtime.profile, runtime.resolver = profile, SimpleNamespace(resolve_dates=resolve_dates)
    runtime.predictor = PredictorSession(device="cpu")
    runtime.entry_policy, runtime.entry_parameters, runtime.position_start = "legacy", None, None
    runtime.mode, runtime.end, runtime._replay_pool = "historical", "2020-10-07", None
    runtime.market_dates, runtime.calendar = trace_dates.strftime("%Y-%m-%d").tolist(), {}
    runtime.inference_reason_counts = Counter()
    runtime.timings = {"inference_wall_seconds": 0., "decision_wall_seconds": 0.}
    result = runtime.replay_batch([{"symbol": "000001.SZ", "raw": raw, "features": features,
                                    "data_end": "2020-10-07"}])[0]
    assert [decision["action"] for decision in result["replay"]["decisions"]] == ["WAIT", "BUY", "HOLD", "HOLD", "SELL"]
    assert np.isnan(result["probabilities"][[0, 3]]).all()
    assert np.isfinite(result["probabilities"][[1, 2, 4]]).all()
    assert not any(decision["ml_filter_applied"] for decision in result["replay"]["decisions"])
    assert runtime.predictor.diagnostics["rows"] == 3 and runtime.predictor.diagnostics["model_loads"] == 2
    assert runtime.inference_reason_counts == {"model_unavailable": 1, "model_input_ineligible": 1}
    # A released model still cannot claim to filter a trade-ineligible row.
    for route in result["routes"]:
        route["applied_to_entry"] = True
    result["probabilities"] = np.full(5, .9)
    released = _replay_stock(result)
    vetoed = released["decisions"][2]
    assert vetoed["probability"] == .9 and not vetoed["ml_filter_applied"]
    assert vetoed["agent_evidence"][2]["judgment"] == "shadow" and not vetoed["agent_evidence"][2]["applied"]

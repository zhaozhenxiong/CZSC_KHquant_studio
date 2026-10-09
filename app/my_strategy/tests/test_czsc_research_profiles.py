"""Model profiles keep numerical semantics separate from trade eligibility."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from my_strategy.services.czsc_research_features import FEATURE_COLUMNS, FEATURE_VERSION, FEATURE_SCHEMA_HASH, _trailing_features
from my_strategy.services.czsc_research_profiles import bind_feature_profile, get_feature_profile, recognized_feature_profile


def test_default_profile_preserves_legacy_identity_and_ma_is_independent():
    legacy, ma = get_feature_profile(), get_feature_profile("ma_trend_v1")
    assert legacy == {"name": "legacy", "version": FEATURE_VERSION, "schema_hash": FEATURE_SCHEMA_HASH,
                      "columns": list(FEATURE_COLUMNS), "strategy_version": "czsc_price_volume_mlp_v1"}
    assert ma["strategy_version"] == "czsc_ma_trend_mlp_v1"
    assert len(ma["columns"]) == len(set(ma["columns"])) == 20
    assert ma["schema_hash"] != legacy["schema_hash"]
    ma["columns"].reverse()
    assert get_feature_profile("ma_trend_v1")["columns"] != ma["columns"]
    with pytest.raises(ValueError, match="unknown"):
        get_feature_profile("unknown")


def test_ma_bind_only_changes_metadata_and_preserves_extraction_identity():
    profile = get_feature_profile("ma_trend_v1")
    frame = pd.DataFrame({column: [1.0, 2.0] for column in (*FEATURE_COLUMNS, *profile["columns"])})
    frame["input_eligible"], frame["model_input_eligible"] = [False, True], [True, True]
    frame["rule_buy"] = [False, True]
    frame.attrs = {"feature_version": FEATURE_VERSION, "schema_hash": FEATURE_SCHEMA_HASH, "data_version": "frozen"}
    original = frame.copy()
    bound = bind_feature_profile(frame, "ma_trend_v1")
    assert_frame_equal(bound, original)
    assert frame.attrs == original.attrs
    assert bound.attrs["base_feature_version"] == FEATURE_VERSION
    assert bound.attrs["base_schema_hash"] == FEATURE_SCHEMA_HASH
    assert bound.attrs["feature_profile"] == "ma_trend_v1"
    rebound = bind_feature_profile(bound, "legacy")
    assert rebound.attrs["base_feature_version"] == FEATURE_VERSION
    assert rebound.attrs["base_schema_hash"] == FEATURE_SCHEMA_HASH
    with pytest.raises(ValueError, match="missing"):
        bind_feature_profile(frame.drop(columns=profile["columns"][0]), "ma_trend_v1")


def test_recognized_profile_rejects_order_and_cross_profile_identity():
    profile = get_feature_profile("ma_trend_v1")
    assert recognized_feature_profile(profile["version"], profile["schema_hash"], profile["columns"], profile["strategy_version"])["name"] == profile["name"]
    for columns, version, schema_hash, strategy in (
        (profile["columns"][::-1], profile["version"], profile["schema_hash"], profile["strategy_version"]),
        (profile["columns"], FEATURE_VERSION, profile["schema_hash"], profile["strategy_version"]),
        (profile["columns"], profile["version"], FEATURE_SCHEMA_HASH, profile["strategy_version"]),
        (profile["columns"], profile["version"], profile["schema_hash"], "czsc_price_volume_mlp_v1"),
    ):
        with pytest.raises(ValueError, match="profile"):
            recognized_feature_profile(version, schema_hash, columns, strategy)


def test_ma_gaps_are_trailing_scale_invariant_and_future_changes_do_not_rewrite_prefix():
    close = pd.Series(20 + np.arange(180) * .04 + np.sin(np.arange(180) / 9))
    bars = pd.DataFrame({"close": close, "high": close + .3, "low": close - .3,
                         "volume": 10000.0, "amount": close * 10000})
    full = _trailing_features(bars)
    assert full.loc[130, "ma5_ma10_gap"] == pytest.approx(close.iloc[126:131].mean() / close.iloc[121:131].mean() - 1)
    profile = get_feature_profile("ma_trend_v1")
    scaled = bars.copy()
    scaled[["close", "high", "low", "amount"]] *= 7
    np.testing.assert_allclose(full[profile["columns"]], _trailing_features(scaled)[profile["columns"]], rtol=1e-10, atol=1e-12, equal_nan=True)
    changed = bars.copy()
    changed.loc[140:, ["close", "high", "low", "volume", "amount"]] *= 100
    assert_frame_equal(full.iloc[:140], _trailing_features(changed).iloc[:140])
    assert_frame_equal(full.iloc[:140], _trailing_features(bars.iloc[:140]))

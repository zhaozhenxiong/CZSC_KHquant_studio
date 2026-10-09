"""Independent accounting checks for append-only descriptive metrics."""
import pandas as pd
import pytest

from my_strategy.scripts.summarize_czsc_wyckoff_metrics import account_metrics, tail


def test_partial_reduction_is_one_fee_inclusive_round_trip_and_open_episode_is_separate():
    days=["2025-01-02","2025-01-03","2025-01-06","2025-01-07","2025-01-08"]
    ledger=pd.DataFrame([
        dict(date=days[0],action="BUY",shares=100,position_before=0,position_after=100,
             price=10.,fee=1.,cash_flow=-1001.,cash_after=8999.,holding_cost_after=10.01),
        dict(date=days[1],action="REDUCE",shares=40,position_before=100,position_after=60,
             price=9.,fee=.36,cash_flow=359.64,cash_after=9358.64,holding_cost_after=10.01),
        dict(date=days[2],action="SELL",shares=60,position_before=60,position_after=0,
             price=12.,fee=.72,cash_flow=719.28,cash_after=10077.92,holding_cost_after=0.),
        dict(date=days[3],action="BUY",shares=100,position_before=0,position_after=100,
             price=8.,fee=.8,cash_flow=-800.8,cash_after=9277.12,holding_cost_after=8.008),
    ])
    cash=[8999.,9358.64,10077.92,9277.12,9277.12]
    shares=[100,60,0,100,100];prices=[10.,9.,12.,8.,9.]
    daily=pd.DataFrame([dict(date=day,cash=c,shares=s,close=p,market_value=s*p,equity=c+s*p)
                        for day,c,s,p in zip(days,cash,shares,prices)])
    result=account_metrics(ledger,daily,days,10000.)
    assert len(result["closed"])==1
    assert result["closed"][0]["net_pnl"]==pytest.approx(77.92)
    assert result["closed"][0]["net_return"]==pytest.approx(77.92/1001.)
    assert result["closed"][0]["holding_sessions"]==2
    assert result["open"]["marked_net_pnl"]==pytest.approx(99.2)
    assert result["open"]["future_liquidation_cost_included"] is False
    assert result["open"]["holding_sessions_to_window_end"]==2
    assert result["cost_basis"].tolist()==pytest.approx([1001.,600.6,0.,800.8,800.8])
    assert result["fees"]==pytest.approx(2.88)


def test_empirical_cvar_uses_exact_fractional_worst_five_percent_mass():
    value=tail(list(range(1,26)))
    assert value["n"]==25
    assert value["q05"]==pytest.approx(2.2)
    assert value["cvar05"]==pytest.approx(1.2)
    assert tail([-.2])["cvar05"]==pytest.approx(-.2)
    assert tail([])=={"q05":None,"cvar05":None,"n":0}


def test_daily_cash_is_verified_against_actual_ledger():
    daily=pd.DataFrame([dict(date="2025-01-02",cash=10001.,shares=0,close=10.,market_value=0.,equity=10001.)])
    with pytest.raises(ValueError,match="daily cash"):
        account_metrics(pd.DataFrame(),daily,["2025-01-02"],10000.)

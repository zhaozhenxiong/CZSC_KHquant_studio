"""Progress failures remain visible without interrupting local update results."""

from unittest.mock import Mock

import pandas as pd
import pytest

from my_strategy.data_manager import updater as updater_module
from my_strategy.data_manager.downloader import DownloadResult
from my_strategy.data_manager.updater import DailyUpdater


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("use_batch", [False, True])
def test_callback_failure_logs_context_and_preserves_results(monkeypatch, caplog, workers, use_batch):
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    updater = DailyUpdater.__new__(DailyUpdater)
    updater.config = {"update": {"use_batch": use_batch, "sleep_seconds": 0}}
    updater.workers = workers
    updater.use_progress_bar = False
    updater.cache = Mock()
    updater.cache.read_stock_daily.return_value = pd.DataFrame(
        {"date": ["2026-09-16"], "trade_close": [10.0]}
    )
    updater.update_one = Mock(side_effect=lambda code, **kwargs: {"code": code, "status": "success"})
    updater.progress_callback = Mock(side_effect=RuntimeError("progress sink unavailable"))
    monkeypatch.setattr(updater_module, "StockPoolManager", Mock(return_value=Mock(
        build_pool=Mock(return_value=pd.DataFrame({"stock": codes}))
    )))

    report = updater.update_pool(end_date="2026-09-16")

    assert report["code"].tolist() == codes
    assert report["status"].tolist() == ["success"] * 3
    assert updater.progress_callback.call_count == 3
    warnings = [record for record in caplog.records if record.name == updater_module.__name__]
    assert len(warnings) == 3
    for record, call in zip(warnings, updater.progress_callback.call_args_list):
        payload = call.args[0]
        assert payload["row"]["code"] in record.getMessage()
        assert f"({payload['current']}/3)" in record.getMessage()
        assert record.exc_info[0] is RuntimeError
    if use_batch:
        updater.cache.update_status_many.assert_called_once()
        updater.cache.append_log_many.assert_called_once()


def test_threaded_progress_keeps_independent_input_order_snapshots(monkeypatch):
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    updater = DailyUpdater.__new__(DailyUpdater)
    updater.config = {"update": {"use_batch": False, "sleep_seconds": 0}}
    updater.workers = 2
    updater.use_progress_bar = False
    updater.update_one = Mock(side_effect=lambda code, **kwargs: {"code": code, "status": "success"})
    events = []
    updater.progress_callback = events.append
    monkeypatch.setattr(updater_module, "StockPoolManager", Mock(return_value=Mock(
        build_pool=Mock(return_value=pd.DataFrame({"stock": codes}))
    )))
    # Make completion order deterministic and different from input order.
    monkeypatch.setattr(updater_module, "as_completed", lambda futures: reversed(list(futures)))

    report = updater.update_pool(end_date="2026-09-16")

    assert report["code"].tolist() == codes
    assert [event["current"] for event in events] == [1, 2, 3]
    assert [[row["code"] for row in event["rows"]] for event in events] == [
        codes[2:], codes[1:], codes,
    ]
    assert len({id(event["rows"]) for event in events}) == 3


@pytest.mark.parametrize("workers", [1, 2])
def test_adaptive_batches_preserve_partition_order_and_short_window_limits(workers):
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]
    recent = {codes[0], codes[2], codes[4]}
    updater = DailyUpdater.__new__(DailyUpdater)
    updater.config = {
        "update": {"start_date": "2015-01-01"},
        "tushare": {"bulk_batch_size": 2, "batch_size": 1},
    }
    updater.workers = workers
    updater.cache = Mock()
    updater.cache.read_stock_daily.side_effect = lambda code: (
        pd.DataFrame({"date": ["2026-09-15"], "trade_close": [10.0]})
        if code in recent else pd.DataFrame()
    )
    updater.downloader = Mock()
    updater.downloader.download_stocks_daily.return_value = DownloadResult(False, None, "", "offline")

    report = updater.update_batch(codes, end_date="2026-09-16")

    batches = [call.args for call in updater.downloader.download_stocks_daily.call_args_list]
    expected = [
        ([codes[0], codes[2]], "2026-09-16", "2026-09-16"),
        ([codes[4]], "2026-09-16", "2026-09-16"),
        ([codes[1]], "2015-01-01", "2026-09-16"),
        ([codes[3]], "2015-01-01", "2026-09-16"),
    ]
    assert sorted(batches) == sorted(expected)
    if workers == 1:
        assert batches == expected
    assert sorted(report["code"]) == sorted(codes)
    updater.cache.save_stocks_daily.assert_not_called()

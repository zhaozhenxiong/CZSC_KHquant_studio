"""Exercise an installed distribution using isolated, explicitly synthetic inputs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def model_probe(device: str) -> dict:
    import numpy as np
    import pandas as pd
    from my_strategy.services.czsc_research_ml import PredictorSession, train_model

    dates = pd.bdate_range("2024-01-02", periods=180)
    values = np.arange(len(dates))
    dataset = pd.DataFrame({"date": dates, "x": np.sin(values / 5), "z": np.cos(values / 7),
                            "label": values % 2, "label_end": dates + pd.offsets.BDay(1), "label_available": True})
    directory = Path(os.environ["KHQUANT_ARTIFACT_ROOT"]) / "installation-model"
    manifest = train_model(dataset, ["x", "z"], directory, str(dates[89].date()),
                           str(dates[90].date()), str(dates[139].date()), device=device,
                           epochs=2, batch_size=32, hidden_sizes=(8, 4), min_train_rows=30, min_validation_rows=20)
    predictor = PredictorSession(device=device, batch_size=32)
    probabilities = predictor.predict(dataset.iloc[140:].copy(), directory)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Installed model inference produced non-finite values")
    return {"training_device": manifest["device"], "inference": predictor.diagnostics,
            "prediction_rows": len(probabilities), "fixture": "synthetic_installation_only"}


def request(base: str, path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"}), timeout=30) as response:
        payload = response.read()
        return json.loads(payload) if response.headers.get_content_type() == "application/json" else payload.decode()


def fixture_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    date, rows = dt.date(2024, 1, 2), []
    while len(rows) < 260:
        if date.weekday() < 5:
            price = 12 + len(rows) * 0.003 + math.sin(len(rows) / 11)
            rows.append(("000001.SZ", date.isoformat(), price, price + 0.2, price - 0.2,
                         price + 0.03, 1000000.0, price * 1000000, 1, "synthetic_installation_fixture"))
        date += dt.timedelta(days=1)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
        connection.execute("CREATE TABLE securities(stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)")
        connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        connection.execute("INSERT INTO securities VALUES (?,?,?,?,?)", ("000001.SZ", "安装测试数据", rows[0][1], rows[-1][1], len(rows)))


def verify(device: str, output: Path | None) -> dict:
    state = Path(tempfile.mkdtemp(prefix="khquant-install-smoke-")).resolve()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("KHQUANT_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(KHQUANT_HOME=str(state), KHQUANT_DATA_ROOT=str(state / "data"),
                       KHQUANT_ARTIFACT_ROOT=str(state / "artifacts"), KHQUANT_METADATA_ROOT=str(state / "data/metadata"),
                       KHQUANT_RAW_DB=str(state / "data/raw/khquant_raw.db"), KHQUANT_LOG_ROOT=str(state / "data/logs"),
                       KHQUANT_COMPUTE_DEVICE=device, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    fixture_database(Path(environment["KHQUANT_RAW_DB"]))

    def command(*args: str, json_output: bool = True):
        completed = subprocess.run([sys.executable, "-I", "-X", "utf8", *args], cwd=state, env=environment,
                                   text=True, encoding="utf-8", capture_output=True, check=False, timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        return json.loads(completed.stdout) if json_output else completed.stdout

    doctor = command("-m", "my_strategy.cli", "doctor", "--check-write", "--strict", "--json")
    model = command(str(Path(__file__).resolve()), "--internal-probe", "--device", device)
    command("-m", "my_strategy.cli", "update-data", "--help", json_output=False)
    analysis = command("-m", "my_strategy.cli", "analyze", "--symbol", "000001.SZ", "--start", "2024-01-01", "--structure-only", "--device", device, "--json")
    scan = command("-m", "my_strategy.cli", "scan", "--stocks", "000001.SZ", "--start", "2024-01-01", "--structure-only", "--device", device, "--cpu-workers", "1", "--json")
    backtest = command("-m", "my_strategy.cli", "backtest", "--symbol", "000001.SZ", "--start", "2024-01-01", "--structure-only", "--device", device, "--cpu-workers", "1", "--json")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = None
    pages = ("analysis", "scan", "backtest", "watchlist", "holdings", "data")
    with (state / "server.log").open("w", encoding="utf-8") as log:
        for restart in range(2):
            process = subprocess.Popen([sys.executable, "-I", "-X", "utf8", "-m", "my_strategy.web_dashboard.scripts.serve_dashboard", "--port", str(port)],
                                       cwd=state, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        request(base, "/api/health")
                        break
                    except (URLError, TimeoutError, ConnectionResetError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError(f"Dashboard did not become healthy: {state / 'server.log'}")
                        time.sleep(0.1)
                for page in pages:
                    if "<html" not in request(base, "/" + page).lower():
                        raise RuntimeError(f"Missing installed dashboard page: {page}")
                if restart == 0:
                    request(base, "/api/watchlist", "POST", {"symbols": ["000001.SZ"]})
                    request(base, "/api/holdings/000001.SZ", "PUT", {"shares": 100, "average_cost": 12})
                else:
                    if "000001.SZ" not in json.dumps(request(base, "/api/watchlist")) or "000001.SZ" not in json.dumps(request(base, "/api/holdings")):
                        raise RuntimeError("Personal data did not survive dashboard restart")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    report = {"status": "passed", "python": sys.executable, "state": str(state), "device": device,
              "doctor": doctor, "model": model, "dashboard_pages": list(pages), "personal_restart": True,
              "analysis_run": analysis.get("run_id"), "scan_run": scan.get("run_id"), "backtest_run": backtest.get("run_id"),
              "update_entrypoint": True,
              "input": "synthetic_installation_fixture; does not certify market data or returns"}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from my_strategy.runtime_env import configure_utf8_stdio

    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--internal-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(model_probe(args.device) if args.internal_probe else verify(args.device, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

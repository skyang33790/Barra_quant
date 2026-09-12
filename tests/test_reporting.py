import json

import pandas as pd
import pytest

import barra_quant.reporting as reporting
from barra_quant.execution import BacktestResult
from barra_quant.reporting import compute_run_id, publish_run
from barra_quant.research import ResearchResult


def _minimal_results() -> tuple[ResearchResult, BacktestResult]:
    dates = pd.bdate_range("2026-03-02", periods=2)
    research = ResearchResult(
        panel=pd.DataFrame({
            "signal_date": [dates[0]],
            "symbol": ["sh600000"],
            "composite_score": [0.5],
        }),
        diagnostics={
            "ic": pd.DataFrame(),
            "decay": pd.DataFrame(),
            "quantiles": pd.DataFrame(),
            "style": pd.DataFrame(),
        },
        benchmark=pd.DataFrame({
            "trade_date": dates,
            "benchmark_return": [0.0, 0.01],
        }),
    )
    backtest = BacktestResult(
        orders=pd.DataFrame({
            "client_order_id": ["order-1"],
            "status": ["FILLED"],
        }),
        fills=pd.DataFrame({
            "fill_id": ["fill-1"],
            "client_order_id": ["order-1"],
            "trade_date": [dates[0]],
            "side": ["BUY"],
            "notional": [1000.0],
            "commission": [5.0],
            "stamp_duty": [0.0],
            "transfer_fee": [0.01],
        }),
        positions=pd.DataFrame({
            "trade_date": [dates[0]],
            "symbol": ["sh600000"],
            "market_value": [1000.0],
        }),
        nav=pd.DataFrame({
            "trade_date": dates,
            "cash": [8995.0, 8995.0],
            "positions_value": [1000.0, 1010.0],
            "gross_nav": [10000.0, 10010.0],
            "nav": [9995.0, 10005.0],
        }),
    )
    return research, backtest


def test_run_id_is_content_addressed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = {"mode": "SMOKE_TEST", "initial_cash": 10_000_000.0}
    manifest = {"source_commit": "abc123", "files": []}
    first = compute_run_id(config, manifest, [source])
    second = compute_run_id(config, manifest, [source])
    assert first == second
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_run_id(config, manifest, [source]) != first


def test_publish_writes_success_last(tmp_path):
    research_result, backtest_result = _minimal_results()
    destination = publish_run(
        tmp_path,
        "run-1",
        {"mode": "SMOKE_TEST"},
        {"source_commit": "abc123"},
        research_result,
        backtest_result,
        {"mode": "SMOKE_TEST"},
    )
    assert (destination / "_SUCCESS").is_file()
    assert (destination / "research_panel.parquet").is_file()
    metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "SMOKE_TEST"
    assert (destination / "report.md").read_text(encoding="utf-8").startswith(
        "> SMOKE_TEST — not production evidence."
    )
    assert all(
        (destination / "figures" / name).is_file()
        for name in reporting.FIGURE_NAMES
    )


def test_failed_publish_never_creates_success(tmp_path, monkeypatch):
    research_result, backtest_result = _minimal_results()

    def fail_write(frame, path):
        raise RuntimeError("write failed")

    monkeypatch.setattr(reporting, "_write_parquet", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        publish_run(
            tmp_path,
            "run-2",
            {},
            {},
            research_result,
            backtest_result,
            {},
        )
    assert not (tmp_path / "run-2" / "_SUCCESS").exists()

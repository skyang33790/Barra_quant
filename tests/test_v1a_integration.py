import json

import pandas as pd

from barra_quant.data_universe import (
    UniverseConfig,
    build_base_universe,
    make_signal_dates,
    validate_prices,
)
from barra_quant.research import (
    FACTOR_COLUMNS,
    ResearchConfig,
    compute_factor_label_panel,
    score_and_target,
)
from barra_quant.run import run_v1a


def test_synthetic_end_to_end_is_deterministic(synthetic_v1a_config):
    first = run_v1a(synthetic_v1a_config)
    second = run_v1a(synthetic_v1a_config)
    assert first == second
    assert (first / "_SUCCESS").is_file()
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "SMOKE_TEST"
    assert metrics["risk_free_rate"] == 0.0


def test_current_data_smoke_run_has_no_performance_assertion(project_smoke_config):
    output = run_v1a(project_smoke_config)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "SMOKE_TEST"
    assert metrics["source_commit"] == "08c4692f2b56d3ae47feae93de5576d4c626be8a"
    report = (output / "report.md").read_text(encoding="utf-8")
    assert report.startswith("> SMOKE_TEST")


def test_appending_future_rows_does_not_change_historical_signal(synthetic_prices):
    prices = validate_prices(synthetic_prices)
    calendar = pd.DatetimeIndex(prices["trade_date"].drop_duplicates())
    historical_date = make_signal_dates(calendar)[20]
    universe_config = UniverseConfig()
    research_config = ResearchConfig()

    def historical_signal(frame: pd.DataFrame) -> pd.DataFrame:
        frame_calendar = pd.DatetimeIndex(frame["trade_date"].drop_duplicates())
        signal_dates = make_signal_dates(frame_calendar)
        universe = build_base_universe(frame, signal_dates, universe_config)
        factors = compute_factor_label_panel(
            frame, frame_calendar, signal_dates, research_config
        )
        scored = score_and_target(factors, universe, universe_config, "fixed-seed")
        columns = [
            "symbol",
            *FACTOR_COLUMNS,
            *[f"forward_return_{horizon}" for horizon in research_config.horizons],
            "eligible",
            "composite_score",
            "target_weight",
        ]
        return scored.loc[scored["signal_date"].eq(historical_date), columns].reset_index(
            drop=True
        )

    last_date = calendar.max()
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=10)
    template = prices.loc[prices["trade_date"].eq(last_date)].copy()
    appended = []
    for offset, future_date in enumerate(future_dates, start=1):
        day = template.copy()
        day["trade_date"] = future_date
        day["available_at"] = future_date.tz_localize("Asia/Shanghai") + pd.Timedelta(
            hours=15, minutes=30
        )
        day[["open", "close", "high", "low"]] *= 1.0 + offset * 0.001
        day["amount"] = day["volume"] * day["close"]
        day["source_file"] = f"stock_price_{future_date:%Y_%m_%d}.csv"
        appended.append(day)
    extended = pd.concat([prices, *appended], ignore_index=True).sort_values(
        ["trade_date", "symbol"], kind="mergesort"
    )

    pd.testing.assert_frame_equal(
        historical_signal(prices),
        historical_signal(extended),
        check_dtype=False,
    )

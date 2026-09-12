import pandas as pd

from barra_quant.data_universe import (
    UniverseConfig,
    build_base_universe,
    make_signal_dates,
)
from barra_quant.research import (
    ResearchConfig,
    build_benchmark,
    compute_barra_lite,
    compute_factor_label_panel,
    compute_rank_ic,
    hac_ic_summary,
    score_and_target,
)


def test_factor_formulas_use_exact_market_offsets(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    signal_date = calendar[80]
    panel = compute_factor_label_panel(
        synthetic_prices,
        calendar,
        pd.DatetimeIndex([signal_date]),
        ResearchConfig(),
    )
    symbol = synthetic_prices["symbol"].iloc[0]
    row = panel.loc[panel["symbol"] == symbol].iloc[0]
    prices = synthetic_prices.set_index(["trade_date", "symbol"])
    expected_momentum = (
        prices.loc[(calendar[75], symbol), "close"]
        / prices.loc[(calendar[60], symbol), "close"] - 1.0
    )
    expected_reversal = -(
        prices.loc[(calendar[80], symbol), "close"]
        / prices.loc[(calendar[75], symbol), "close"] - 1.0
    )
    assert row["momentum_20_5"] == expected_momentum
    assert row["reversal_5"] == expected_reversal


def test_label_does_not_skip_missing_symbol_day(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    symbol = synthetic_prices["symbol"].iloc[0]
    signal_date = calendar[80]
    missing_start = calendar[81]
    prices = synthetic_prices.loc[
        ~(
            (synthetic_prices["symbol"] == symbol)
            & (synthetic_prices["trade_date"] == missing_start)
        )
    ]
    panel = compute_factor_label_panel(
        prices,
        calendar,
        pd.DatetimeIndex([signal_date]),
        ResearchConfig(),
    )
    assert pd.isna(
        panel.loc[panel["symbol"] == symbol, "forward_return_5"].iloc[0]
    )


def test_target_requires_one_hundred_complete_candidates(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    signal_dates = pd.DatetimeIndex([calendar[80]])
    base = build_base_universe(synthetic_prices, signal_dates, UniverseConfig())
    factors = compute_factor_label_panel(
        synthetic_prices,
        calendar,
        signal_dates,
        ResearchConfig(),
    )
    result = score_and_target(factors, base, UniverseConfig(), run_seed="fixed")
    selected = result.loc[result["target_weight"] > 0]
    assert len(selected) == 50
    assert selected["target_weight"].eq(0.02).all()
    assert selected["signal_id"].is_unique


def test_rank_ic_is_cross_sectional_by_signal_date():
    panel = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-01-02"] * 4 + ["2026-01-09"] * 4),
        "symbol": ["a", "b", "c", "d"] * 2,
        "factor": [1, 2, 3, 4, 4, 3, 2, 1],
        "forward_return_5": [0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01],
    })
    result = compute_rank_ic(panel, "factor", 5)
    assert result["rank_ic"].tolist() == [1.0, 1.0]


def test_hac_lag_uses_weekly_overlap():
    ic = pd.Series([0.01, 0.03, -0.01, 0.02, 0.04])
    summary_5 = hac_ic_summary(ic, horizon=5)
    summary_20 = hac_ic_summary(ic, horizon=20)
    assert summary_5["max_lags"] == 0
    assert summary_20["max_lags"] == 3


def test_barra_lite_does_not_shorten_beta_window(synthetic_prices):
    short = synthetic_prices[synthetic_prices["trade_date"].isin(
        sorted(synthetic_prices["trade_date"].unique())[:90]
    )]
    result = compute_barra_lite(
        short,
        ResearchConfig(beta_window=120, beta_min_observations=100),
    )
    assert result["beta_120"].isna().all()
    assert result["residual_volatility_120"].isna().all()


def test_benchmark_is_named_gross_research_benchmark():
    panel = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "eligible": [True, True],
        "forward_return_5": [0.01, 0.03],
    })
    result = build_benchmark(panel, primary_horizon=5)
    assert result.iloc[0]["benchmark_name"] == "pit_universe_equal_weight_gross"
    assert result.iloc[0]["benchmark_return"] == 0.02

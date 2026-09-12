import pandas as pd

from barra_quant.data_universe import (
    UniverseConfig,
    build_base_universe,
    make_signal_dates,
)
from barra_quant.research import (
    ResearchConfig,
    compute_factor_label_panel,
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

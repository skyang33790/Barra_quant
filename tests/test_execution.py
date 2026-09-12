from datetime import date

import pandas as pd
import pytest

from barra_quant.execution import (
    ExecutionConfig,
    FeeRule,
    replay_fills,
    simulate,
    fee_for_fill,
    fill_price,
    legal_quantity,
)


FEE_RULE = FeeRule(
    effective_from=date(2023, 8, 28),
    effective_to=None,
    commission_bps=3.0,
    minimum_commission=5.0,
    stamp_duty_sell_bps=5.0,
    transfer_bps=0.1,
)


def test_fee_rule_applies_minimum_commission_and_sell_tax():
    buy = fee_for_fill(10_000.0, "BUY", date(2026, 3, 2), (FEE_RULE,))
    sell = fee_for_fill(10_000.0, "SELL", date(2026, 3, 2), (FEE_RULE,))
    assert buy == {
        "commission": 5.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.1,
    }
    assert sell == {
        "commission": 5.0,
        "stamp_duty": 5.0,
        "transfer_fee": 0.1,
    }


def test_board_quantity_rules():
    assert legal_quantity(375, min_buy_qty=100, qty_step=100, side="BUY") == 300
    assert legal_quantity(375, min_buy_qty=200, qty_step=1, side="BUY") == 375
    assert legal_quantity(175, min_buy_qty=200, qty_step=1, side="BUY") == 0
    assert legal_quantity(
        75,
        min_buy_qty=100,
        qty_step=100,
        side="SELL",
        position_quantity=75,
    ) == 75


def test_slippage_does_not_cross_price_limit():
    assert fill_price(
        10.0, "BUY", slippage_bps=20.0, limit_up=10.01, limit_down=9.0
    ) == 10.01
    assert fill_price(
        10.0, "SELL", slippage_bps=20.0, limit_up=11.0, limit_down=9.99
    ) == 9.99


def _simulation_inputs():
    dates = pd.bdate_range("2026-03-02", periods=4)
    prices = pd.DataFrame([
        {
            "trade_date": day,
            "symbol": "sh600000",
            "open": 10.0,
            "close": 10.0,
            "high": 10.2,
            "low": 9.8,
            "volume": 1_000_000,
            "amount": 10_000_000.0,
        }
        for day in dates
    ])
    targets = pd.DataFrame([
        {
            "signal_date": dates[0],
            "execution_date": dates[1],
            "symbol": "sh600000",
            "target_weight": 1.0,
            "signal_id": "signal-buy",
            "adv20_volume": 1_000_000.0,
        },
        {
            "signal_date": dates[1],
            "execution_date": dates[2],
            "symbol": "sh600000",
            "target_weight": 0.0,
            "signal_id": "signal-sell",
            "adv20_volume": 1_000_000.0,
        },
    ])
    constraints = pd.DataFrame([
        {
            "trade_date": day,
            "symbol": "sh600000",
            "is_tradable": True,
            "limit_up": 11.0,
            "limit_down": 9.0,
            "min_buy_qty": 100,
            "qty_step": 100,
            "price_tick": 0.01,
            "constraints_source": "inferred",
            "source_version": "synthetic",
        }
        for day in dates
    ])
    return prices, pd.DatetimeIndex(dates), targets, constraints


def test_simulator_enforces_t1_and_nav_identity():
    prices, calendar, targets, constraints = _simulation_inputs()
    result = simulate(
        prices,
        calendar,
        targets,
        constraints,
        (FEE_RULE,),
        ExecutionConfig(
            initial_cash=1_000_000.0,
            slippage_bps=0.0,
            volume_cap=0.05,
        ),
        "run",
    )
    assert result.fills["client_order_id"].is_unique
    difference = result.nav["cash"] + result.nav["positions_value"] - result.nav["nav"]
    assert difference.abs().max() < 1e-8
    sell = result.fills[result.fills["side"] == "SELL"].iloc[0]
    assert sell["trade_date"] == calendar[2]


def test_open_fill_does_not_change_when_future_intraday_columns_change():
    prices, calendar, targets, constraints = _simulation_inputs()
    before = simulate(
        prices, calendar, targets, constraints, (FEE_RULE,), ExecutionConfig(), "run"
    )
    changed = prices.copy()
    changed["high"] *= 10
    changed["low"] *= 0.1
    changed["volume"] *= 100
    changed["amount"] *= 100
    after = simulate(
        changed, calendar, targets, constraints, (FEE_RULE,), ExecutionConfig(), "run"
    )
    pd.testing.assert_frame_equal(before.fills, after.fills)


def test_replay_is_idempotent():
    prices, calendar, targets, constraints = _simulation_inputs()
    result = simulate(
        prices, calendar, targets, constraints, (FEE_RULE,), ExecutionConfig(), "run"
    )
    duplicated = pd.concat([result.fills, result.fills], ignore_index=True)
    cash, holdings = replay_fills(1_000_000.0, duplicated)
    unique_cash, unique_holdings = replay_fills(1_000_000.0, result.fills)
    assert cash == unique_cash
    pd.testing.assert_frame_equal(holdings, unique_holdings)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("limit_up", "LIMIT_UP"),
        ("limit_down", "LIMIT_DOWN"),
        ("no_bar", "NO_BAR"),
        ("lot_too_small", "LOT_TOO_SMALL"),
        ("volume_cap", "VOLUME_CAP"),
        ("insufficient_cash", "INSUFFICIENT_CASH"),
    ],
)
def test_rejection_statuses(case, expected):
    prices, calendar, targets, constraints = _simulation_inputs()
    config = ExecutionConfig(
        initial_cash=1_000_000.0,
        slippage_bps=0.0,
        volume_cap=0.05,
    )
    if case == "limit_up":
        prices.loc[prices["trade_date"] == calendar[1], "open"] = 11.0
    elif case == "limit_down":
        prices.loc[prices["trade_date"] == calendar[2], "open"] = 9.0
    elif case == "no_bar":
        prices = prices.loc[prices["trade_date"] != calendar[1]].copy()
    elif case == "lot_too_small":
        targets.loc[targets["signal_id"] == "signal-buy", "target_weight"] = 0.0005
    elif case == "volume_cap":
        targets.loc[targets["signal_id"] == "signal-buy", "adv20_volume"] = 1_000.0
    elif case == "insufficient_cash":
        config = ExecutionConfig(
            initial_cash=999.0,
            slippage_bps=0.0,
            volume_cap=0.05,
        )
        targets.loc[targets["signal_id"] == "signal-buy", "target_weight"] = 2.0

    result = simulate(
        prices, calendar, targets, constraints, (FEE_RULE,), config, "run"
    )
    rejected = result.orders.loc[result.orders["status"] == expected]
    assert len(rejected) == 1
    client_order_id = rejected.iloc[0]["client_order_id"]
    assert not result.fills["client_order_id"].eq(client_order_id).any()

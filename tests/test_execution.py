from datetime import date

from barra_quant.execution import (
    ExecutionConfig,
    FeeRule,
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

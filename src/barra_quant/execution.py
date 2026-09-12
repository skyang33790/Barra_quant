from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class FeeRule:
    effective_from: date
    effective_to: date | None
    commission_bps: float
    minimum_commission: float
    stamp_duty_sell_bps: float
    transfer_bps: float


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000_000.0
    slippage_bps: float = 5.0
    volume_cap: float = 0.05


@dataclass
class AcquisitionLot:
    symbol: str
    quantity: int
    acquired_on: pd.Timestamp
    unlock_on: pd.Timestamp
    cost: float


@dataclass(frozen=True)
class BacktestResult:
    orders: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    nav: pd.DataFrame


def _active_rule(trade_date: date, rules: tuple[FeeRule, ...]) -> FeeRule:
    matches = [
        rule for rule in rules
        if rule.effective_from <= trade_date
        and (rule.effective_to is None or trade_date <= rule.effective_to)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one fee rule for {trade_date}, found {len(matches)}"
        )
    return matches[0]


def fee_for_fill(
    notional: float,
    side: Side,
    trade_date: date,
    rules: tuple[FeeRule, ...],
) -> dict[str, float]:
    rule = _active_rule(trade_date, rules)
    return {
        "commission": max(
            rule.minimum_commission,
            notional * rule.commission_bps / 10_000.0,
        ),
        "stamp_duty": (
            notional * rule.stamp_duty_sell_bps / 10_000.0
            if side == "SELL"
            else 0.0
        ),
        "transfer_fee": notional * rule.transfer_bps / 10_000.0,
    }


def legal_quantity(
    requested: int,
    min_buy_qty: int,
    qty_step: int,
    side: Side,
    position_quantity: int = 0,
) -> int:
    if side == "SELL" and requested == position_quantity and position_quantity < min_buy_qty:
        return position_quantity
    if requested < min_buy_qty:
        return 0
    return min_buy_qty + ((requested - min_buy_qty) // qty_step) * qty_step


def fill_price(
    open_price: float,
    side: Side,
    slippage_bps: float,
    limit_up: float,
    limit_down: float,
) -> float:
    multiplier = (
        1.0 + slippage_bps / 10_000.0
        if side == "BUY"
        else 1.0 - slippage_bps / 10_000.0
    )
    slipped = open_price * multiplier
    if side == "BUY":
        return round(min(slipped, limit_up), 2)
    return round(max(slipped, limit_down), 2)


def deterministic_id(*parts: object) -> str:
    value = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]

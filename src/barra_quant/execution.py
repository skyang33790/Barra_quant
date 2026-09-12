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


ORDER_COLUMNS = [
    "trade_date", "as_of", "signal_id", "symbol", "side", "order_id",
    "client_order_id", "requested_quantity", "filled_quantity",
    "cancelled_quantity", "status", "reason",
]
FILL_COLUMNS = [
    "trade_date", "as_of", "signal_id", "symbol", "side", "order_id",
    "client_order_id", "fill_id", "quantity", "price", "notional",
    "commission", "stamp_duty", "transfer_fee",
]
POSITION_COLUMNS = [
    "trade_date", "as_of", "symbol", "total_quantity", "sellable_quantity",
    "locked_quantity", "mark_price", "market_value", "stale_days",
]
NAV_COLUMNS = [
    "trade_date", "as_of", "cash", "positions_value", "gross_nav", "nav",
]


def _total_quantity(lots: list[AcquisitionLot], symbol: str) -> int:
    return sum(lot.quantity for lot in lots if lot.symbol == symbol)


def _sellable_quantity(
    lots: list[AcquisitionLot],
    symbol: str,
    trade_date: pd.Timestamp,
) -> int:
    return sum(
        lot.quantity
        for lot in lots
        if lot.symbol == symbol and lot.unlock_on <= trade_date
    )


def _consume_lots(
    lots: list[AcquisitionLot],
    symbol: str,
    quantity: int,
    trade_date: pd.Timestamp,
) -> None:
    remaining = quantity
    candidates = sorted(
        (
            lot for lot in lots
            if lot.symbol == symbol and lot.unlock_on <= trade_date
        ),
        key=lambda lot: (lot.acquired_on, lot.symbol),
    )
    for lot in candidates:
        consumed = min(lot.quantity, remaining)
        lot.quantity -= consumed
        remaining -= consumed
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"insufficient unlocked lots for {symbol}")
    lots[:] = [lot for lot in lots if lot.quantity > 0]


def replay_fills(
    initial_cash: float,
    fills: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    cash = float(initial_cash)
    holdings: dict[str, int] = {}
    if fills.empty:
        return cash, pd.DataFrame(columns=["symbol", "quantity"])
    ordered = (
        fills.sort_values(["trade_date", "fill_id"], kind="mergesort")
        .drop_duplicates("client_order_id", keep="first")
    )
    for row in ordered.itertuples(index=False):
        quantity = int(row.quantity)
        fees = float(row.commission + row.stamp_duty + row.transfer_fee)
        if row.side == "BUY":
            cash -= float(row.notional) + fees
            holdings[row.symbol] = holdings.get(row.symbol, 0) + quantity
        else:
            cash += float(row.notional) - fees
            holdings[row.symbol] = holdings.get(row.symbol, 0) - quantity
            if holdings[row.symbol] < 0:
                raise ValueError(f"negative holding for {row.symbol}")
    result = pd.DataFrame(
        [
            {"symbol": symbol, "quantity": quantity}
            for symbol, quantity in sorted(holdings.items())
            if quantity != 0
        ],
        columns=["symbol", "quantity"],
    )
    return cash, result


def simulate(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    targets: pd.DataFrame,
    constraints: pd.DataFrame,
    fee_rules: tuple[FeeRule, ...],
    config: ExecutionConfig,
    run_id: str,
) -> BacktestResult:
    price_frame = prices.copy()
    price_frame["trade_date"] = pd.to_datetime(price_frame["trade_date"]).dt.normalize()
    constraint_frame = constraints.copy()
    constraint_frame["trade_date"] = pd.to_datetime(
        constraint_frame["trade_date"]
    ).dt.normalize()
    target_frame = targets.copy()
    if not target_frame.empty:
        target_frame["execution_date"] = pd.to_datetime(
            target_frame["execution_date"]
        ).dt.normalize()
    dates = list(pd.DatetimeIndex(calendar).sort_values().unique())
    next_date = {
        dates[index]: dates[index + 1]
        for index in range(len(dates) - 1)
    }
    price_lookup = {
        (row.trade_date, row.symbol): row
        for row in price_frame.itertuples(index=False)
    }
    constraint_lookup = {
        (row.trade_date, row.symbol): row
        for row in constraint_frame.itertuples(index=False)
    }
    lots: list[AcquisitionLot] = []
    cash = float(config.initial_cash)
    cumulative_cost = 0.0
    last_close: dict[str, float] = {}
    stale_days: dict[str, int] = {}
    seen_clients: set[str] = set()
    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []

    for trade_date in dates:
        held_symbols = sorted({lot.symbol for lot in lots})
        opening_positions_value = 0.0
        for symbol in held_symbols:
            bar = price_lookup.get((trade_date, symbol))
            mark = float(bar.open) if bar is not None else last_close.get(symbol, 0.0)
            opening_positions_value += _total_quantity(lots, symbol) * mark
        opening_nav = cash + opening_positions_value
        day_targets = target_frame.loc[
            target_frame["execution_date"] == trade_date
        ] if not target_frame.empty else target_frame
        intents: list[dict[str, object]] = []
        for _, target in day_targets.iterrows():
            symbol = str(target["symbol"])
            current_quantity = _total_quantity(lots, symbol)
            bar = price_lookup.get((trade_date, symbol))
            if bar is None:
                target_quantity = 0 if float(target["target_weight"]) == 0 else current_quantity + 1
            else:
                target_quantity = int(
                    opening_nav * float(target["target_weight"]) / float(bar.open)
                )
            if target_quantity == current_quantity:
                continue
            side: Side = "BUY" if target_quantity > current_quantity else "SELL"
            client_order_id = deterministic_id(
                run_id, target["signal_id"], trade_date, symbol
            )
            if client_order_id in seen_clients:
                continue
            seen_clients.add(client_order_id)
            intents.append({
                "target": target,
                "symbol": symbol,
                "side": side,
                "current_quantity": current_quantity,
                "target_quantity": target_quantity,
                "client_order_id": client_order_id,
            })
        intents.sort(key=lambda item: (0 if item["side"] == "SELL" else 1, item["symbol"]))

        for intent in intents:
            target = intent["target"]
            symbol = str(intent["symbol"])
            side: Side = intent["side"]
            current_quantity = int(intent["current_quantity"])
            target_quantity = int(intent["target_quantity"])
            requested_quantity = abs(target_quantity - current_quantity)
            client_order_id = str(intent["client_order_id"])
            order_id = deterministic_id("order", client_order_id)
            bar = price_lookup.get((trade_date, symbol))
            constraint = constraint_lookup.get((trade_date, symbol))
            status: str | None = None
            reason = ""
            filled_quantity = 0

            if bar is None or constraint is None:
                status = reason = "NO_BAR"
            elif not bool(constraint.is_tradable):
                status = reason = "SUSPENDED"
            elif side == "BUY" and float(bar.open) >= float(constraint.limit_up):
                status = reason = "LIMIT_UP"
            elif side == "SELL" and float(bar.open) <= float(constraint.limit_down):
                status = reason = "LIMIT_DOWN"

            if status is None:
                adv_cap = int(float(target["adv20_volume"]) * config.volume_cap)
                quantity = min(requested_quantity, adv_cap)
                sellable_quantity = _sellable_quantity(lots, symbol, trade_date)
                if side == "SELL":
                    quantity = min(quantity, sellable_quantity)
                quantity = legal_quantity(
                    requested=quantity,
                    min_buy_qty=int(constraint.min_buy_qty),
                    qty_step=int(constraint.qty_step),
                    side=side,
                    position_quantity=current_quantity,
                )
                if quantity == 0:
                    if side == "SELL" and sellable_quantity == 0:
                        status = reason = "T1_LOCKED"
                    elif requested_quantity < int(constraint.min_buy_qty):
                        status = reason = "LOT_TOO_SMALL"
                    elif adv_cap < int(constraint.min_buy_qty):
                        status = reason = "VOLUME_CAP"
                    else:
                        status = reason = "LOT_TOO_SMALL"
                else:
                    execution_price = fill_price(
                        float(bar.open),
                        side,
                        config.slippage_bps,
                        float(constraint.limit_up),
                        float(constraint.limit_down),
                    )
                    fees = fee_for_fill(
                        execution_price * quantity,
                        side,
                        trade_date.date(),
                        fee_rules,
                    )
                    if side == "BUY":
                        while quantity > 0 and (
                            execution_price * quantity + sum(fees.values()) > cash
                        ):
                            quantity = legal_quantity(
                                requested=quantity - int(constraint.qty_step),
                                min_buy_qty=int(constraint.min_buy_qty),
                                qty_step=int(constraint.qty_step),
                                side=side,
                                position_quantity=current_quantity,
                            )
                            if quantity > 0:
                                fees = fee_for_fill(
                                    execution_price * quantity,
                                    side,
                                    trade_date.date(),
                                    fee_rules,
                                )
                        if quantity == 0:
                            status = reason = "INSUFFICIENT_CASH"

            if status is None:
                notional = execution_price * quantity
                total_fees = sum(fees.values())
                if side == "BUY":
                    cash -= notional + total_fees
                    lots.append(AcquisitionLot(
                        symbol=symbol,
                        quantity=quantity,
                        acquired_on=trade_date,
                        unlock_on=next_date.get(trade_date, pd.Timestamp.max.normalize()),
                        cost=notional + total_fees,
                    ))
                else:
                    cash += notional - total_fees
                    _consume_lots(lots, symbol, quantity, trade_date)
                cumulative_cost += total_fees
                filled_quantity = quantity
                status = "FILLED" if quantity == requested_quantity else "PARTIAL"
                if status == "PARTIAL":
                    reason = "VOLUME_OR_LOT_OR_CASH_CAP"
                fill_rows.append({
                    "trade_date": trade_date,
                    "as_of": trade_date,
                    "signal_id": target["signal_id"],
                    "symbol": symbol,
                    "side": side,
                    "order_id": order_id,
                    "client_order_id": client_order_id,
                    "fill_id": deterministic_id("fill", order_id, trade_date),
                    "quantity": quantity,
                    "price": execution_price,
                    "notional": notional,
                    **fees,
                })

            order_rows.append({
                "trade_date": trade_date,
                "as_of": trade_date,
                "signal_id": target["signal_id"],
                "symbol": symbol,
                "side": side,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "requested_quantity": requested_quantity,
                "filled_quantity": filled_quantity,
                "cancelled_quantity": requested_quantity - filled_quantity,
                "status": status,
                "reason": reason,
            })

        day_bars = price_frame.loc[price_frame["trade_date"] == trade_date]
        for bar in day_bars.itertuples(index=False):
            last_close[bar.symbol] = float(bar.close)
            stale_days[bar.symbol] = 0

        positions_value = 0.0
        for symbol in sorted({lot.symbol for lot in lots}):
            total_quantity = _total_quantity(lots, symbol)
            sellable_quantity = _sellable_quantity(lots, symbol, trade_date)
            bar = price_lookup.get((trade_date, symbol))
            if bar is None:
                stale_days[symbol] = stale_days.get(symbol, 0) + 1
            mark_price = last_close.get(symbol, 0.0)
            market_value = total_quantity * mark_price
            positions_value += market_value
            position_rows.append({
                "trade_date": trade_date,
                "as_of": trade_date,
                "symbol": symbol,
                "total_quantity": total_quantity,
                "sellable_quantity": sellable_quantity,
                "locked_quantity": total_quantity - sellable_quantity,
                "mark_price": mark_price,
                "market_value": market_value,
                "stale_days": stale_days.get(symbol, 0),
            })
        nav = cash + positions_value
        if cash < -1e-8:
            raise AssertionError(f"negative cash: {cash}")
        if abs(cash + positions_value - nav) >= 1e-8:
            raise AssertionError("NAV identity failed")
        nav_rows.append({
            "trade_date": trade_date,
            "as_of": trade_date,
            "cash": cash,
            "positions_value": positions_value,
            "gross_nav": nav + cumulative_cost,
            "nav": nav,
        })

    orders = pd.DataFrame(order_rows, columns=ORDER_COLUMNS)
    fills = pd.DataFrame(fill_rows, columns=FILL_COLUMNS)
    positions = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
    nav = pd.DataFrame(nav_rows, columns=NAV_COLUMNS)
    if not orders.empty:
        orders = orders.sort_values(
            ["trade_date", "side", "symbol", "order_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    if not fills.empty:
        fills = fills.sort_values(
            ["trade_date", "symbol", "fill_id"], kind="mergesort"
        ).reset_index(drop=True)
    if not positions.empty:
        positions = positions.sort_values(
            ["trade_date", "symbol"], kind="mergesort"
        ).reset_index(drop=True)
    nav = nav.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    return BacktestResult(
        orders=orders,
        fills=fills,
        positions=positions,
        nav=nav,
    )

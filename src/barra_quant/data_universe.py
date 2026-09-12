from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "symbol",
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "source_file",
    "source_commit",
}


@dataclass(frozen=True)
class UniverseConfig:
    initial_cash: float = 10_000_000.0
    top_n: int = 50
    min_cross_section: int = 100
    min_history_days: int = 60
    liquidity_window: int = 20
    max_position_to_adv: float = 0.01
    min_avg_amount: float = 20_000_000.0


@dataclass(frozen=True)
class DataBundle:
    prices: pd.DataFrame
    calendar: pd.DatetimeIndex
    source_manifest: dict[str, object]


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"missing price columns: {sorted(missing)}")
    if prices.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("duplicate symbol trade_date keys")
    result = prices.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result = result.sort_values(
        ["trade_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    localized = result["trade_date"].dt.tz_localize("Asia/Shanghai")
    result["available_at"] = localized + pd.Timedelta(hours=15, minutes=30)
    result["availability_source"] = "inferred_for_smoke_test"
    return result


def make_signal_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    frame = pd.DataFrame(
        {"trade_date": pd.DatetimeIndex(calendar).sort_values().unique()}
    )
    frame["week"] = frame["trade_date"].dt.to_period("W-FRI")
    return pd.DatetimeIndex(
        frame.groupby("week", sort=True)["trade_date"].max()
    )


def load_data(database: Path, source_manifest_path: Path) -> DataBundle:
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    with duckdb.connect(str(database), read_only=True) as connection:
        prices = connection.execute(
            "SELECT * FROM prices ORDER BY trade_date, symbol"
        ).df()
    prices = validate_prices(prices)
    commits = set(prices["source_commit"].unique())
    if commits != {manifest["source_commit"]}:
        raise ValueError(
            f"source commit mismatch: prices={commits}, "
            f"manifest={manifest['source_commit']}"
        )
    calendar = pd.DatetimeIndex(
        prices["trade_date"].drop_duplicates().sort_values()
    )
    manifest = dict(manifest)
    manifest["calendar_source"] = "inferred_from_prices"
    manifest["price_basis"] = "raw_unadjusted_smoke_test"
    manifest["corporate_actions_complete"] = False
    return DataBundle(
        prices=prices,
        calendar=calendar,
        source_manifest=manifest,
    )


def classify_board(symbol: str) -> str:
    value = symbol.lower()
    if value.startswith("sh900") or value.startswith("sz200"):
        return "EXCLUDED_B_SHARE"
    if value.startswith(("sh688", "sh689")):
        return "STAR"
    if value.startswith(("sz300", "sz301")):
        return "CHINEXT"
    if value.startswith("bj"):
        return "BSE"
    if value.startswith(("sh", "sz")) and value[2:].isdigit() and len(value) == 8:
        return "MAIN"
    return "INVALID"


def build_base_universe(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: UniverseConfig,
) -> pd.DataFrame:
    ordered = prices.sort_values(["symbol", "trade_date"], kind="mergesort").copy()
    grouped = ordered.groupby("symbol", sort=False)
    ordered["history_days"] = grouped.cumcount() + 1
    ordered["adv20_amount"] = grouped["amount"].transform(
        lambda values: values.rolling(
            config.liquidity_window, min_periods=config.liquidity_window
        ).mean()
    )
    ordered["adv20_volume"] = grouped["volume"].transform(
        lambda values: values.rolling(
            config.liquidity_window, min_periods=config.liquidity_window
        ).mean()
    )
    panel = ordered[ordered["trade_date"].isin(signal_dates)].copy()
    panel = panel.rename(columns={"trade_date": "signal_date"})
    panel["board"] = panel["symbol"].map(classify_board)
    target_value = config.initial_cash / config.top_n
    panel["base_eligible"] = (
        panel["board"].isin(["MAIN", "STAR", "CHINEXT", "BSE"])
        & panel["history_days"].ge(config.min_history_days)
        & panel["adv20_amount"].ge(config.min_avg_amount)
        & (target_value <= panel["adv20_amount"] * config.max_position_to_adv)
    )
    panel["universe_reason"] = np.select(
        [
            panel["board"].isin(["EXCLUDED_B_SHARE", "INVALID"]),
            panel["history_days"].lt(config.min_history_days),
            panel["adv20_amount"].lt(config.min_avg_amount),
            target_value > panel["adv20_amount"] * config.max_position_to_adv,
        ],
        ["MARKET_EXCLUDED", "INSUFFICIENT_HISTORY", "LOW_ABSOLUTE_LIQUIDITY", "LOW_CAPACITY"],
        default="ELIGIBLE",
    )
    columns = [
        "signal_date", "symbol", "board", "history_days", "adv20_amount",
        "adv20_volume", "base_eligible", "universe_reason",
    ]
    return panel[columns].sort_values(
        ["signal_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def infer_constraints(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    del calendar
    result = prices[["trade_date", "symbol", "open", "close"]].copy()
    result["board"] = result["symbol"].map(classify_board)
    previous = (
        result.sort_values(["symbol", "trade_date"])
        .groupby("symbol")["close"]
        .shift(1)
    )
    rate = result["board"].map(
        {"MAIN": 0.10, "CHINEXT": 0.20, "STAR": 0.20, "BSE": 0.30}
    )
    result["is_tradable"] = True
    result["limit_up"] = (previous * (1.0 + rate)).round(2)
    result["limit_down"] = (previous * (1.0 - rate)).round(2)
    result["min_buy_qty"] = result["board"].map(
        {"MAIN": 100, "CHINEXT": 100, "STAR": 200, "BSE": 100}
    )
    result["qty_step"] = result["board"].map(
        {"MAIN": 100, "CHINEXT": 100, "STAR": 1, "BSE": 1}
    )
    result["price_tick"] = 0.01
    result["constraints_source"] = "inferred"
    result["source_version"] = "v1a-inferred"
    columns = [
        "trade_date", "symbol", "is_tradable", "limit_up", "limit_down",
        "min_buy_qty", "qty_step", "price_tick", "constraints_source",
        "source_version",
    ]
    return result[columns]

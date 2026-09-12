from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
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

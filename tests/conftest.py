from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest


def make_price_panel(symbol_count: int = 120, periods: int = 140) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    symbols = [f"sh{600000 + index:06d}" for index in range(symbol_count)]
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, trade_date in enumerate(dates):
            base = 10.0 + symbol_index * 0.01 + date_index * 0.002
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": base,
                    "close": base * (1.0 + ((symbol_index % 7) - 3) * 0.0002),
                    "high": base * 1.01,
                    "low": base * 0.99,
                    "volume": 5_000_000 + symbol_index * 1_000,
                    "amount": (5_000_000 + symbol_index * 1_000) * base,
                    "source_file": f"stock_price_{trade_date:%Y_%m_%d}.csv",
                    "source_commit": "synthetic-commit",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    return make_price_panel()

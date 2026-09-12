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


def _smoke_config(database: str, manifest: str) -> dict[str, object]:
    return {
        "database": database,
        "source_manifest": manifest,
        "output_root": "artifacts/runs",
        "mode": "SMOKE_TEST",
        "initial_cash": 10_000_000.0,
        "top_n": 50,
        "min_cross_section": 100,
        "min_history_days": 60,
        "liquidity_window": 20,
        "max_position_to_adv": 0.01,
        "min_avg_amount": 20_000_000.0,
        "horizons": [1, 5, 10, 20],
        "primary_horizon": 5,
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
        "beta_window": 120,
        "beta_min_observations": 100,
        "volume_cap": 0.05,
        "slippage_bps": 5.0,
        "slippage_stress_bps": [10.0, 20.0],
        "annualization_days": 252,
        "risk_free_rate": 0.0,
    }


@pytest.fixture
def synthetic_v1a_config(
    tmp_path: Path,
    synthetic_prices: pd.DataFrame,
) -> Path:
    database = tmp_path / "data" / "warehouse" / "synthetic.duckdb"
    manifest_path = tmp_path / "data" / "normalized" / "synthetic" / "manifest.json"
    config_path = tmp_path / "configs" / "v1a_smoke.json"
    database.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    with duckdb.connect(str(database)) as connection:
        connection.register("synthetic_prices", synthetic_prices)
        connection.execute("CREATE TABLE prices AS SELECT * FROM synthetic_prices")
    manifest = {
        "source_commit": "synthetic-commit",
        "source_committed_at": "2026-09-11T00:00:00+08:00",
        "files": [],
        "calendar_source": "inferred_from_prices",
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    config = _smoke_config(
        database.relative_to(tmp_path).as_posix(),
        manifest_path.relative_to(tmp_path).as_posix(),
    )
    config_path.write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    return config_path


@pytest.fixture
def project_smoke_config() -> Path:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "v1a_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [
        root / config[key]
        for key in ("database", "source_manifest")
        if not (root / config[key]).exists()
    ]
    if missing:
        pytest.skip(f"materialize current commit-keyed data first: {missing}")
    return config_path

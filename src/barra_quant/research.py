from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_universe import UniverseConfig


@dataclass(frozen=True)
class ResearchConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    primary_horizon: int = 5
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    beta_window: int = 120
    beta_min_observations: int = 100


@dataclass(frozen=True)
class ResearchResult:
    panel: pd.DataFrame
    diagnostics: dict[str, object]
    benchmark: pd.DataFrame


FACTOR_COLUMNS = ("momentum_20_5", "reversal_5", "low_volatility_20")


def _long_at_signal_dates(
    wide: pd.DataFrame,
    name: str,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    selected = wide.reindex(signal_dates)
    return (
        selected.rename_axis(index="signal_date", columns="symbol")
        .stack(future_stack=True)
        .rename(name)
        .reset_index()
    )


def compute_factor_label_panel(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    signal_dates: pd.DatetimeIndex,
    config: ResearchConfig,
) -> pd.DataFrame:
    close = prices.pivot(
        index="trade_date", columns="symbol", values="close"
    ).reindex(calendar)
    open_price = prices.pivot(
        index="trade_date", columns="symbol", values="open"
    ).reindex(calendar)
    log_return = np.log(close).diff()
    series = {
        "momentum_20_5": close.shift(5) / close.shift(20) - 1.0,
        "reversal_5": -(close / close.shift(5) - 1.0),
        "low_volatility_20": -log_return.rolling(
            20, min_periods=20
        ).std(ddof=1),
    }
    for horizon in config.horizons:
        series[f"forward_return_{horizon}"] = (
            open_price.shift(-(horizon + 1)) / open_price.shift(-1) - 1.0
        )
    frames = [
        _long_at_signal_dates(value, name, signal_dates)
        for name, value in series.items()
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(
            frame,
            on=["signal_date", "symbol"],
            how="outer",
            validate="one_to_one",
        )
    return result.sort_values(
        ["signal_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def _rank_factor(
    group: pd.DataFrame,
    factor: str,
    lower: float,
    upper: float,
) -> pd.Series:
    values = group[factor]
    clipped = values.clip(values.quantile(lower), values.quantile(upper))
    rank = clipped.rank(method="average")
    count = rank.notna().sum()
    if count < 2:
        return pd.Series(np.nan, index=group.index)
    return 2.0 * (rank - 1.0) / (count - 1.0) - 1.0


def score_and_target(
    factors: pd.DataFrame,
    base_universe: pd.DataFrame,
    config: UniverseConfig,
    run_seed: str,
) -> pd.DataFrame:
    panel = factors.merge(
        base_universe,
        on=["signal_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    for factor in FACTOR_COLUMNS:
        score_column = f"{factor}_score"
        panel[score_column] = np.nan
        for group_index in panel.groupby("signal_date", sort=True).groups.values():
            eligible_index = panel.index[group_index][
                panel.loc[group_index, "base_eligible"].fillna(False)
            ]
            panel.loc[eligible_index, score_column] = _rank_factor(
                panel.loc[eligible_index], factor, 0.01, 0.99
            ).to_numpy()
    scores = [f"{factor}_score" for factor in FACTOR_COLUMNS]
    panel["factor_complete"] = panel[scores].notna().all(axis=1)
    panel["eligible"] = (
        panel["base_eligible"].fillna(False) & panel["factor_complete"]
    )
    panel["composite_score"] = panel[scores].mean(axis=1).where(panel["eligible"])
    panel["composite_rank"] = np.nan
    eligible = panel.loc[panel["eligible"]].sort_values(
        ["signal_date", "composite_score", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    panel.loc[eligible.index, "composite_rank"] = (
        eligible.groupby("signal_date", sort=True).cumcount() + 1
    )
    counts = panel.groupby("signal_date")["eligible"].transform("sum")
    panel["target_weight"] = np.where(
        panel["eligible"]
        & counts.ge(config.min_cross_section)
        & panel["composite_rank"].le(config.top_n),
        1.0 / config.top_n,
        0.0,
    )
    panel["signal_id"] = panel.apply(
        lambda row: hashlib.sha256(
            f"{run_seed}|{row['signal_date']:%Y-%m-%d}|{row['symbol']}".encode(
                "utf-8"
            )
        ).hexdigest()[:24],
        axis=1,
    )
    panel["signal_status"] = np.where(
        counts.ge(config.min_cross_section),
        "READY",
        "INSUFFICIENT_UNIVERSE",
    )
    return panel.sort_values(
        ["signal_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)

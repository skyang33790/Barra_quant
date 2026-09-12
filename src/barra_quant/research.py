from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

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


def compute_rank_ic(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
) -> pd.DataFrame:
    outcome = f"forward_return_{horizon}"
    rows: list[dict[str, object]] = []
    complete = panel.dropna(subset=[factor, outcome])
    for signal_date, group in complete.groupby("signal_date", sort=True):
        value = spearmanr(group[factor], group[outcome]).statistic
        rows.append({
            "signal_date": signal_date,
            "factor": factor,
            "horizon": horizon,
            "rank_ic": value,
        })
    return pd.DataFrame(
        rows,
        columns=["signal_date", "factor", "horizon", "rank_ic"],
    )


def hac_ic_summary(ic: pd.Series, horizon: int) -> dict[str, float | int]:
    clean = ic.dropna().astype(float)
    max_lags = max(0, math.ceil(horizon / 5) - 1)
    standard_deviation = float(clean.std(ddof=1))
    summary: dict[str, float | int] = {
        "observations": len(clean),
        "mean_ic": float(clean.mean()),
        "standard_deviation": standard_deviation,
        "icir": (
            float(clean.mean() / standard_deviation)
            if standard_deviation > 0
            else float("nan")
        ),
        "hac_t": float("nan"),
        "max_lags": max_lags,
    }
    if len(clean) < max(3, max_lags + 2):
        return summary
    model = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )
    summary["hac_t"] = float(model.tvalues[0])
    return summary


def build_benchmark(panel: pd.DataFrame, primary_horizon: int) -> pd.DataFrame:
    column = f"forward_return_{primary_horizon}"
    result = (
        panel.loc[panel["eligible"]]
        .groupby("signal_date", as_index=False)[column]
        .mean()
    )
    result = result.rename(columns={column: "benchmark_return"})
    result["benchmark_name"] = "pit_universe_equal_weight_gross"
    return result


def compute_quantile_returns(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
) -> pd.DataFrame:
    outcome = f"forward_return_{horizon}"
    work = panel.dropna(subset=[factor, outcome]).copy()
    if work.empty:
        return pd.DataFrame(
            columns=["signal_date", "quantile", outcome]
        )
    work["quantile"] = work.groupby("signal_date")[factor].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 5, labels=False
        ) + 1
    )
    return work.groupby(
        ["signal_date", "quantile"], as_index=False
    )[outcome].mean()


def compute_barra_lite(
    prices: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(
        pd.to_datetime(prices["trade_date"]).sort_values().unique()
    )
    close = prices.pivot(
        index="trade_date", columns="symbol", values="close"
    ).reindex(calendar)
    amount = prices.pivot(
        index="trade_date", columns="symbol", values="amount"
    ).reindex(calendar)
    stock_returns = close.pct_change(fill_method=None)
    market_returns = stock_returns.mean(axis=1, skipna=True)
    momentum = close.shift(5) / close.shift(20) - 1.0
    liquidity = np.log(amount.rolling(20, min_periods=20).mean())
    beta = pd.DataFrame(np.nan, index=calendar, columns=close.columns)
    residual_volatility = beta.copy()

    for symbol in close.columns:
        paired = pd.DataFrame({
            "stock": stock_returns[symbol],
            "market": market_returns,
        })
        for end in range(config.beta_window - 1, len(calendar)):
            window = paired.iloc[end - config.beta_window + 1:end + 1].dropna()
            if len(window) < config.beta_min_observations:
                continue
            if window["market"].nunique() < 2:
                continue
            model = sm.OLS(
                window["stock"].to_numpy(),
                sm.add_constant(window["market"].to_numpy()),
            ).fit()
            beta.iat[end, beta.columns.get_loc(symbol)] = float(model.params[1])
            residual_volatility.iat[
                end, residual_volatility.columns.get_loc(symbol)
            ] = float(pd.Series(model.resid).std(ddof=1))

    frames = [
        _long_at_signal_dates(beta, "beta_120", calendar),
        _long_at_signal_dates(
            residual_volatility,
            "residual_volatility_120",
            calendar,
        ),
        _long_at_signal_dates(momentum, "momentum_style_20_5", calendar),
        _long_at_signal_dates(liquidity, "liquidity_style_20", calendar),
    ]
    result = frames[0].rename(columns={"signal_date": "trade_date"})
    for frame in frames[1:]:
        frame = frame.rename(columns={"signal_date": "trade_date"})
        result = result.merge(
            frame,
            on=["trade_date", "symbol"],
            how="outer",
            validate="one_to_one",
        )
    return result.sort_values(
        ["trade_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def residualize_factor(panel: pd.DataFrame, factor: str) -> pd.DataFrame:
    exposure_columns = [
        "beta_120",
        "residual_volatility_120",
        "momentum_style_20_5",
        "liquidity_style_20",
    ]
    rows: list[pd.DataFrame] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        output = group[["signal_date", "symbol"]].copy()
        output["factor"] = factor
        output["residual"] = np.nan
        output["status"] = "UNAVAILABLE"
        available = [
            column for column in exposure_columns
            if column in group and group[column].notna().any()
        ]
        if not available:
            rows.append(output)
            continue
        complete = group[[factor, *available]].dropna()
        varying = [
            column for column in available
            if complete[column].std(ddof=1) > 0
        ]
        complete = group[[factor, *varying]].dropna() if varying else complete.iloc[0:0]
        if len(complete) < 30 or not varying:
            rows.append(output)
            continue
        standardized = complete[varying].apply(
            lambda values: (values - values.mean()) / values.std(ddof=1)
        )
        model = sm.OLS(
            complete[factor].to_numpy(),
            sm.add_constant(standardized.to_numpy()),
        ).fit()
        output.loc[complete.index, "residual"] = model.resid
        output.loc[complete.index, "status"] = "AVAILABLE"
        rows.append(output)
    if not rows:
        return pd.DataFrame(
            columns=["signal_date", "symbol", "factor", "residual", "status"]
        )
    return pd.concat(rows, ignore_index=True)


def _coverage_and_extremes(
    panel: pd.DataFrame,
    factors: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict[str, object]] = []
    extreme_rows: list[dict[str, object]] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        base_mask = group["base_eligible"].fillna(False)
        eligible_count = int(base_mask.sum())
        for factor in factors:
            values = group.loc[base_mask, factor].dropna()
            non_null = len(values)
            coverage_rows.append({
                "signal_date": signal_date,
                "factor": factor,
                "eligible_count": eligible_count,
                "non_null_count": non_null,
                "missing_count": eligible_count - non_null,
                "coverage": non_null / eligible_count if eligible_count else np.nan,
            })
            extreme_rows.append({
                "signal_date": signal_date,
                "factor": factor,
                "minimum": float(values.min()) if non_null else np.nan,
                "p01": float(values.quantile(0.01)) if non_null else np.nan,
                "p99": float(values.quantile(0.99)) if non_null else np.nan,
                "maximum": float(values.max()) if non_null else np.nan,
            })
    return pd.DataFrame(coverage_rows), pd.DataFrame(extreme_rows)


def _quantile_diagnostics(
    panel: pd.DataFrame,
    factors: tuple[str, ...],
    horizon: int,
) -> pd.DataFrame:
    outcome = f"forward_return_{horizon}"
    frames: list[pd.DataFrame] = []
    for factor in factors:
        quantiles = compute_quantile_returns(panel, factor, horizon)
        if quantiles.empty:
            continue
        quantiles = quantiles.rename(columns={outcome: "quantile_return"})
        quantiles["factor"] = factor
        summaries: list[dict[str, object]] = []
        for signal_date, group in quantiles.groupby("signal_date", sort=True):
            indexed = group.set_index("quantile")["quantile_return"]
            spread = (
                float(indexed.get(5, np.nan) - indexed.get(1, np.nan))
                if 1 in indexed.index and 5 in indexed.index
                else np.nan
            )
            monotonicity = (
                float(spearmanr(group["quantile"], group["quantile_return"]).statistic)
                if len(group) >= 2 and group["quantile_return"].nunique() >= 2
                else np.nan
            )
            summaries.append({
                "signal_date": signal_date,
                "top_minus_bottom": spread,
                "monotonicity": monotonicity,
            })
        quantiles = quantiles.merge(
            pd.DataFrame(summaries), on="signal_date", validate="many_to_one"
        )
        frames.append(quantiles)
    columns = [
        "signal_date", "quantile", "quantile_return", "factor",
        "top_minus_bottom", "monotonicity",
    ]
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)


def _turnover(panel: pd.DataFrame) -> pd.DataFrame:
    weights = panel.pivot(
        index="signal_date", columns="symbol", values="target_weight"
    ).fillna(0.0).sort_index()
    previous = weights.shift(1).fillna(0.0)
    turnover = 0.5 * (weights - previous).abs().sum(axis=1)
    return turnover.rename("turnover").reset_index()


def _factor_correlations(
    panel: pd.DataFrame,
    factors: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        correlation = group[list(factors)].corr(method="spearman")
        for left_index, left in enumerate(factors):
            for right in factors[left_index + 1:]:
                rows.append({
                    "signal_date": signal_date,
                    "factor_left": left,
                    "factor_right": right,
                    "rank_correlation": correlation.loc[left, right],
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary = detail.groupby(
        ["factor_left", "factor_right"], as_index=False
    )["rank_correlation"].agg(["mean", "count"]).reset_index()
    return summary.rename(
        columns={"mean": "mean_rank_correlation", "count": "observations"}
    )


def _rank_and_stability(
    panel: pd.DataFrame,
    factors: tuple[str, ...],
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    dates = pd.DatetimeIndex(panel["signal_date"].drop_duplicates().sort_values())
    rank_frames: list[pd.DataFrame] = []
    hac_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    eligible = panel.loc[panel["eligible"]]
    for factor in factors:
        for horizon in horizons:
            observed = compute_rank_ic(eligible, factor, horizon)
            expected = pd.DataFrame({"signal_date": dates})
            expected["factor"] = factor
            expected["horizon"] = horizon
            complete = expected.merge(
                observed,
                on=["signal_date", "factor", "horizon"],
                how="left",
                validate="one_to_one",
            )
            rank_frames.append(complete)
            summary = hac_ic_summary(complete["rank_ic"], horizon)
            hac_rows.append({"factor": factor, "horizon": horizon, **summary})

            midpoint = len(dates) // 2
            periods = {
                "FIRST_HALF": dates[:midpoint],
                "SECOND_HALF": dates[midpoint:],
            }
            periods.update({
                f"YEAR_{year}": dates[dates.year == year]
                for year in sorted(set(dates.year))
            })
            for period, period_dates in periods.items():
                period_ic = complete.loc[
                    complete["signal_date"].isin(period_dates), "rank_ic"
                ]
                period_summary = hac_ic_summary(period_ic, horizon)
                stability_rows.append({
                    "factor": factor,
                    "horizon": horizon,
                    "period": period,
                    **period_summary,
                })
    rank_columns = ["signal_date", "factor", "horizon", "rank_ic"]
    rank_ic = (
        pd.concat(rank_frames, ignore_index=True)[rank_columns]
        if rank_frames
        else pd.DataFrame(columns=rank_columns)
    )
    return rank_ic, hac_rows, pd.DataFrame(stability_rows)


def build_research(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    base_universe: pd.DataFrame,
    universe_config: UniverseConfig,
    config: ResearchConfig,
    run_seed: str,
) -> ResearchResult:
    signal_dates = pd.DatetimeIndex(
        base_universe["signal_date"].drop_duplicates().sort_values()
    )
    factors = compute_factor_label_panel(
        prices, calendar, signal_dates, config
    )
    panel = score_and_target(
        factors, base_universe, universe_config, run_seed
    )
    style = compute_barra_lite(prices, config).rename(
        columns={"trade_date": "signal_date"}
    )
    panel = panel.merge(
        style,
        on=["signal_date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    diagnostic_factors = (*FACTOR_COLUMNS, "composite_score")
    rank_ic, hac_rows, stability = _rank_and_stability(
        panel,
        diagnostic_factors,
        config.horizons,
    )
    coverage, extremes = _coverage_and_extremes(panel, diagnostic_factors)
    quantiles = _quantile_diagnostics(
        panel.loc[panel["eligible"]],
        diagnostic_factors,
        config.primary_horizon,
    )
    style_frames = [
        residualize_factor(panel, f"{factor}_score")
        for factor in FACTOR_COLUMNS
    ]
    style_status = pd.concat(style_frames, ignore_index=True)
    return ResearchResult(
        panel=panel,
        diagnostics={
            "mode": "SMOKE_TEST",
            "rank_ic": rank_ic,
            "hac": hac_rows,
            "quantiles": quantiles,
            "coverage": coverage,
            "extremes": extremes,
            "turnover": _turnover(panel),
            "factor_correlation": _factor_correlations(
                panel, FACTOR_COLUMNS
            ),
            "stability": stability,
            "regime_status": "UNAVAILABLE_IN_V1A",
            "style_status": style_status,
            "unavailable_styles": [
                "size",
                "nonlinear_size",
                "value",
                "profitability",
                "growth",
                "leverage",
                "industry",
            ],
        },
        benchmark=build_benchmark(panel, config.primary_horizon),
    )

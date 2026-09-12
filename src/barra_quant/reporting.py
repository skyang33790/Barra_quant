from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .execution import BacktestResult
from .research import ResearchResult


FIGURE_NAMES = (
    "ic_series.png",
    "decay.png",
    "quantiles.png",
    "nav_drawdown.png",
    "style_exposure.png",
)


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def compute_run_id(
    config: dict[str, object],
    source_manifest: dict[str, object],
    source_files: Sequence[Path],
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(config))
    digest.update(_canonical_json(source_manifest))
    for path in sorted(source_files, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:24]


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    with duckdb.connect() as connection:
        connection.register("frame_to_write", frame)
        escaped = path.as_posix().replace("'", "''")
        connection.execute(
            f"COPY frame_to_write TO '{escaped}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )


def compute_metrics(
    backtest: BacktestResult,
    benchmark: pd.DataFrame,
    trading_days: int = 252,
) -> dict[str, object]:
    nav = backtest.nav.sort_values("trade_date").copy()
    returns = nav["nav"].pct_change().dropna()
    annual_return = (
        (nav["nav"].iloc[-1] / nav["nav"].iloc[0])
        ** (trading_days / max(1, len(returns)))
        - 1.0
    )
    annual_volatility = returns.std(ddof=1) * np.sqrt(trading_days)
    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0
        else float("nan")
    )
    drawdown = nav["nav"] / nav["nav"].cummax() - 1.0
    status_counts = (
        backtest.orders["status"].value_counts().sort_index().to_dict()
        if "status" in backtest.orders
        else {}
    )
    total_cost = float(
        backtest.fills[["commission", "stamp_duty", "transfer_fee"]]
        .sum()
        .sum()
    )
    return {
        "mode": "SMOKE_TEST",
        "annualization_days": trading_days,
        "risk_free_rate": 0.0,
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "gross_total_return": float(
            nav["gross_nav"].iloc[-1] / nav["gross_nav"].iloc[0] - 1.0
        ),
        "net_total_return": float(
            nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0
        ),
        "total_cost": total_cost,
        "benchmark_name": "pit_universe_equal_weight_gross",
        "benchmark_observations": int(len(benchmark)),
        "order_status_counts": status_counts,
    }


def _unavailable_figure(path: Path, title: str, reason: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        f"UNAVAILABLE\n{reason}",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_figures(
    staging: Path,
    research: ResearchResult,
    backtest: BacktestResult,
) -> None:
    figure_dir = staging / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rank_ic = research.diagnostics.get(
        "rank_ic", research.diagnostics.get("ic", pd.DataFrame())
    )
    if isinstance(rank_ic, pd.DataFrame) and not rank_ic.empty:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for keys, group in rank_ic.dropna(subset=["rank_ic"]).groupby(
            ["factor", "horizon"], sort=True
        ):
            axis.plot(group["signal_date"], group["rank_ic"], label=str(keys))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title("Cross-sectional RankIC")
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(figure_dir / "ic_series.png", dpi=150)
        plt.close(figure)

        decay = rank_ic.groupby(["factor", "horizon"], as_index=False)["rank_ic"].mean()
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for factor, group in decay.groupby("factor", sort=True):
            axis.plot(group["horizon"], group["rank_ic"], marker="o", label=factor)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title("Mean RankIC decay")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(figure_dir / "decay.png", dpi=150)
        plt.close(figure)
    else:
        _unavailable_figure(
            figure_dir / "ic_series.png", "Cross-sectional RankIC", "no RankIC rows"
        )
        _unavailable_figure(
            figure_dir / "decay.png", "Mean RankIC decay", "no RankIC rows"
        )

    quantiles = research.diagnostics.get("quantiles", pd.DataFrame())
    if isinstance(quantiles, pd.DataFrame) and not quantiles.empty:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        work = quantiles.sort_values("signal_date").copy()
        work["cumulative_gross_return"] = work.groupby(
            ["factor", "quantile"]
        )["quantile_return"].transform(lambda values: (1.0 + values).cumprod() - 1.0)
        for keys, group in work.groupby(["factor", "quantile"], sort=True):
            axis.plot(
                group["signal_date"],
                group["cumulative_gross_return"],
                label=str(keys),
            )
        axis.set_title("Quantile cumulative gross returns")
        axis.legend(fontsize=5, ncol=2)
        figure.tight_layout()
        figure.savefig(figure_dir / "quantiles.png", dpi=150)
        plt.close(figure)
    else:
        _unavailable_figure(
            figure_dir / "quantiles.png", "Quantile returns", "no quantile rows"
        )

    nav = backtest.nav.sort_values("trade_date")
    if not nav.empty:
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        top.plot(nav["trade_date"], nav["nav"] / nav["nav"].iloc[0], label="net strategy")
        top.plot(
            nav["trade_date"],
            nav["gross_nav"] / nav["gross_nav"].iloc[0],
            label="gross strategy",
        )
        benchmark = research.benchmark.copy()
        if not benchmark.empty and "benchmark_return" in benchmark:
            date_column = "signal_date" if "signal_date" in benchmark else "trade_date"
            benchmark = benchmark.sort_values(date_column)
            top.plot(
                benchmark[date_column],
                (1.0 + benchmark["benchmark_return"].fillna(0.0)).cumprod(),
                label="gross research benchmark",
            )
        top.legend(fontsize=7)
        top.set_title("NAV and drawdown")
        drawdown = nav["nav"] / nav["nav"].cummax() - 1.0
        bottom.fill_between(nav["trade_date"], drawdown, 0.0)
        figure.tight_layout()
        figure.savefig(figure_dir / "nav_drawdown.png", dpi=150)
        plt.close(figure)
    else:
        _unavailable_figure(
            figure_dir / "nav_drawdown.png", "NAV and drawdown", "no NAV rows"
        )

    style_columns = [
        "beta_120", "residual_volatility_120",
        "momentum_style_20_5", "liquidity_style_20",
    ]
    available = [column for column in style_columns if column in research.panel]
    style = research.panel[["signal_date", *available]].groupby(
        "signal_date", as_index=False
    ).mean() if available else pd.DataFrame()
    if not style.empty and style[available].notna().any().any():
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for column in available:
            axis.plot(style["signal_date"], style[column], label=column)
        axis.set_title("Barra-lite mean exposure")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(figure_dir / "style_exposure.png", dpi=150)
        plt.close(figure)
    else:
        _unavailable_figure(
            figure_dir / "style_exposure.png",
            "Barra-lite exposure",
            "required exposure rows unavailable",
        )


def _render_report(
    metrics: dict[str, object],
    source_manifest: dict[str, object],
    research: ResearchResult,
) -> str:
    unavailable = research.diagnostics.get("unavailable_styles", [
        "size", "nonlinear_size", "value", "profitability",
        "growth", "leverage", "industry",
    ])
    lines = [
        "> SMOKE_TEST — not production evidence.",
        "",
        "# V1a cross-sectional factor audit",
        "",
        "## Preregistered factors",
        "",
        "- Momentum 20-5: medium-horizon continuation; direction positive; "
        "formula `close[T-5] / close[T-20] - 1`.",
        "- Reversal 5: short-horizon reversal; direction negative raw return; "
        "formula `-(close[T] / close[T-5] - 1)`.",
        "- Low volatility 20: lower realized volatility preference; direction "
        "negative volatility; formula `-std(log_return, 20)`.",
        "- Parameters are frozen by the checked-in configuration. Primary horizon "
        "is 5 market days from T+1 open.",
        "- Rejection criteria require V1b significance, stability, decay, uniqueness, "
        "capacity, and implementability checks; none are evaluated as passed here.",
        "",
        "## Data and time contract",
        "",
        f"- Source commit: `{source_manifest.get('source_commit', 'UNKNOWN')}`.",
        f"- Source committed at: `{source_manifest.get('source_committed_at', 'UNKNOWN')}`.",
        f"- Calendar: `{metrics.get('calendar_source', 'inferred_from_prices')}`.",
        f"- Availability: `{metrics.get('availability_source', 'inferred_for_smoke_test')}`.",
        f"- Trading constraints: `{metrics.get('constraints_source', 'inferred')}`.",
        "- WARNING: prices are raw and unadjusted; corporate actions are incomplete.",
        "- Historical security status and universe membership are not complete "
        "point-in-time data.",
        "- The source data license must be reviewed before redistribution or production use.",
        "",
        "## Engineering smoke results",
        "",
        f"- Gross total return: `{metrics.get('gross_total_return')}`.",
        f"- Net total return: `{metrics.get('net_total_return')}`.",
        f"- Total explicit cost: `{metrics.get('total_cost')}`.",
        f"- Order terminal statuses: `{json.dumps(metrics.get('order_status_counts', {}), sort_keys=True)}`.",
        "- Strategy gross and net NAV are execution-account results. The benchmark "
        "is a separate gross, equal-weight PIT-approximate research benchmark.",
        f"- Benchmark: `{metrics.get('benchmark_name', 'pit_universe_equal_weight_gross')}`.",
        "",
        "## Barra coverage",
        "",
        "- Available only where sufficient observations exist: beta, residual "
        "volatility, price momentum, and liquidity.",
        f"- Full-Barra styles UNAVAILABLE: `{', '.join(map(str, unavailable))}`.",
        "",
        "## V1b gate",
        "",
        "- Chronological split: 60% train / 20% validation / 20% final test.",
        "- RESEARCH_CANDIDATE: `NOT_EVALUATED_IN_V1A`.",
        "- Positive IC, Sharpe, or return in this short sample is not evidence of "
        "factor validity, deployability, or investment readiness.",
        "",
    ]
    report = "\n".join(lines)
    assert report.startswith("> SMOKE_TEST — not production evidence.")
    return report


def _assert_unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{name} missing primary key columns: {columns}")
    if frame.duplicated(columns).any():
        raise ValueError(f"{name} has duplicate primary keys: {columns}")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def publish_run(
    output_root: Path,
    run_id: str,
    config: dict[str, object],
    source_manifest: dict[str, object],
    research: ResearchResult,
    backtest: BacktestResult,
    metrics: dict[str, object],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    config_hash = hashlib.sha256(_canonical_json(config)).hexdigest()
    manifest_hash = hashlib.sha256(_canonical_json(source_manifest)).hexdigest()
    identity = {
        "run_id": run_id,
        "config_sha256": config_hash,
        "source_manifest_sha256": manifest_hash,
    }
    if destination.exists():
        if not (destination / "_SUCCESS").is_file():
            raise ValueError(f"incomplete immutable run exists: {destination}")
        stored = json.loads(
            (destination / "identity.json").read_text(encoding="utf-8")
        )
        if stored != identity:
            raise ValueError(f"run identity mismatch: {destination}")
        return destination

    staging = output_root / f".{run_id}.staging"
    failed = output_root / f"{run_id}.failed"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_json(staging / "config.json", config)
        _write_json(staging / "source_manifest.json", source_manifest)
        _write_json(staging / "identity.json", identity)
        _write_json(staging / "metrics.json", metrics)

        _assert_unique(research.panel, ["signal_date", "symbol"], "research panel")
        _assert_unique(backtest.orders, ["client_order_id"], "orders")
        _assert_unique(backtest.fills, ["fill_id"], "fills")
        position_date = "as_of" if "as_of" in backtest.positions else "trade_date"
        nav_date = "as_of" if "as_of" in backtest.nav else "trade_date"
        _assert_unique(backtest.positions, [position_date, "symbol"], "positions")
        _assert_unique(backtest.nav, [nav_date], "nav")

        tables = {
            "research_panel.parquet": research.panel,
            "benchmark.parquet": research.benchmark,
            "orders.parquet": backtest.orders,
            "fills.parquet": backtest.fills,
            "positions.parquet": backtest.positions,
            "nav.parquet": backtest.nav,
        }
        for name, frame in tables.items():
            _write_parquet(frame, staging / name)
        diagnostic_summary: dict[str, object] = {}
        for name, value in research.diagnostics.items():
            if isinstance(value, pd.DataFrame):
                diagnostic_summary[name] = {"rows": len(value)}
                if len(value.columns):
                    _write_parquet(value, staging / f"diagnostic_{name}.parquet")
            elif isinstance(value, list) and value and all(
                isinstance(item, dict) for item in value
            ):
                frame = pd.DataFrame(value)
                diagnostic_summary[name] = {"rows": len(frame)}
                _write_parquet(frame, staging / f"diagnostic_{name}.parquet")
            else:
                diagnostic_summary[name] = value
        _write_json(staging / "diagnostics.json", diagnostic_summary)
        _write_figures(staging, research, backtest)
        report = _render_report(metrics, source_manifest, research)
        (staging / "report.md").write_text(report, encoding="utf-8")

        expected = [
            "config.json", "source_manifest.json", "identity.json",
            "metrics.json", "diagnostics.json", "report.md", *tables,
        ]
        expected.extend(f"figures/{name}" for name in FIGURE_NAMES)
        missing = [name for name in expected if not (staging / name).is_file()]
        if missing:
            raise RuntimeError(f"missing run artifacts: {missing}")
        (staging / "_SUCCESS").write_text(
            json.dumps(identity, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return destination
    except Exception as error:
        success = staging / "_SUCCESS"
        if success.exists():
            success.unlink()
        if failed.exists():
            shutil.rmtree(failed)
        if staging.exists():
            os.replace(staging, failed)
        else:
            failed.mkdir(parents=True, exist_ok=True)
        _write_json(
            failed / "error.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise

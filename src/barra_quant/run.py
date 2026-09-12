from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data_universe import (
    UniverseConfig,
    build_base_universe,
    infer_constraints,
    load_data,
    make_signal_dates,
)
from .execution import ExecutionConfig, FeeRule, simulate
from .reporting import compute_metrics, compute_run_id, publish_run
from .research import ResearchConfig, build_research


CORE_SOURCE_FILES = (
    Path(__file__).with_name("data_universe.py"),
    Path(__file__).with_name("research.py"),
    Path(__file__).with_name("execution.py"),
    Path(__file__).with_name("reporting.py"),
    Path(__file__),
)


def run_v1a(config_path: Path) -> Path:
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if raw_config["mode"] != "SMOKE_TEST":
        raise ValueError("V1a mode must be SMOKE_TEST")
    root = config_path.resolve().parents[1]
    bundle = load_data(
        root / raw_config["database"],
        root / raw_config["source_manifest"],
    )
    universe_config = UniverseConfig(
        initial_cash=raw_config["initial_cash"],
        top_n=raw_config["top_n"],
        min_cross_section=raw_config["min_cross_section"],
        min_history_days=raw_config["min_history_days"],
        liquidity_window=raw_config["liquidity_window"],
        max_position_to_adv=raw_config["max_position_to_adv"],
        min_avg_amount=raw_config["min_avg_amount"],
    )
    research_config = ResearchConfig(
        horizons=tuple(raw_config["horizons"]),
        primary_horizon=raw_config["primary_horizon"],
        winsor_lower=raw_config["winsor_lower"],
        winsor_upper=raw_config["winsor_upper"],
        beta_window=raw_config["beta_window"],
        beta_min_observations=raw_config["beta_min_observations"],
    )
    run_id = compute_run_id(
        raw_config, bundle.source_manifest, CORE_SOURCE_FILES
    )
    signal_dates = make_signal_dates(bundle.calendar)
    base_universe = build_base_universe(
        bundle.prices, signal_dates, universe_config
    )
    research = build_research(
        bundle.prices,
        bundle.calendar,
        base_universe,
        universe_config,
        research_config,
        run_id,
    )
    constraints = infer_constraints(bundle.prices, bundle.calendar)
    targets = research.panel.loc[
        research.panel["target_weight"] > 0,
        [
            "signal_date",
            "symbol",
            "target_weight",
            "signal_id",
            "adv20_volume",
        ],
    ].copy()
    next_dates = {
        bundle.calendar[index]: bundle.calendar[index + 1]
        for index in range(len(bundle.calendar) - 1)
    }
    targets = targets.loc[targets["signal_date"].isin(next_dates)].copy()
    targets["execution_date"] = targets["signal_date"].map(next_dates)
    fee_rules = (
        FeeRule(
            effective_from=pd.Timestamp("2023-08-28").date(),
            effective_to=None,
            commission_bps=3.0,
            minimum_commission=5.0,
            stamp_duty_sell_bps=5.0,
            transfer_bps=0.1,
        ),
    )
    backtest = simulate(
        bundle.prices,
        bundle.calendar,
        targets,
        constraints,
        fee_rules,
        ExecutionConfig(
            initial_cash=raw_config["initial_cash"],
            slippage_bps=raw_config["slippage_bps"],
            volume_cap=raw_config["volume_cap"],
        ),
        run_id,
    )
    metrics = compute_metrics(
        backtest, research.benchmark, raw_config["annualization_days"]
    )
    metrics.update({
        "source_commit": bundle.source_manifest["source_commit"],
        "calendar_source": bundle.source_manifest["calendar_source"],
        "availability_source": "inferred_for_smoke_test",
        "constraints_source": "inferred",
        "universe_pit_complete": False,
        "corporate_actions_complete": False,
    })
    return publish_run(
        root / raw_config["output_root"],
        run_id,
        raw_config,
        bundle.source_manifest,
        research,
        backtest,
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the V1a A-share smoke pipeline"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_v1a(args.config))


if __name__ == "__main__":
    main()

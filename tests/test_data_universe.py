from barra_quant import __version__

import pandas as pd
import pytest

from barra_quant.data_universe import (
    UniverseConfig,
    build_base_universe,
    classify_board,
    infer_constraints,
    make_signal_dates,
    validate_prices,
)


def test_package_version_marks_v1a():
    assert __version__ == "0.1.0"


def test_validate_prices_adds_inferred_availability(synthetic_prices):
    result = validate_prices(synthetic_prices)
    first = result.iloc[0]
    assert str(first["available_at"].tz) == "Asia/Shanghai"
    assert first["available_at"].hour == 15
    assert first["available_at"].minute == 30
    assert first["availability_source"] == "inferred_for_smoke_test"


def test_validate_prices_rejects_duplicate_symbol_date(synthetic_prices):
    duplicate = pd.concat([synthetic_prices, synthetic_prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*symbol.*trade_date"):
        validate_prices(duplicate)


def test_signal_date_is_last_market_day_in_each_week():
    calendar = pd.DatetimeIndex(pd.to_datetime([
        "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06",
        "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12"
    ]))
    assert make_signal_dates(calendar).tolist() == [
        pd.Timestamp("2026-03-06"), pd.Timestamp("2026-03-12")
    ]


def test_board_mapping_and_b_share_exclusion():
    assert classify_board("sh600000") == "MAIN"
    assert classify_board("sh688001") == "STAR"
    assert classify_board("sz300001") == "CHINEXT"
    assert classify_board("bj920001") == "BSE"
    assert classify_board("sh900901") == "EXCLUDED_B_SHARE"
    assert classify_board("sz200001") == "EXCLUDED_B_SHARE"


def test_universe_uses_only_information_through_signal_date(synthetic_prices):
    signal_dates = make_signal_dates(pd.DatetimeIndex(synthetic_prices["trade_date"].unique()))
    cutoff = signal_dates[-2]
    before = build_base_universe(
        synthetic_prices[synthetic_prices["trade_date"] <= cutoff],
        signal_dates[signal_dates <= cutoff],
        UniverseConfig(),
    )
    future = synthetic_prices.copy()
    future.loc[future["trade_date"] > cutoff, "amount"] *= 1000
    after = build_base_universe(future, signal_dates, UniverseConfig())
    columns = ["signal_date", "symbol", "adv20_amount", "base_eligible"]
    pd.testing.assert_frame_equal(
        before[columns].reset_index(drop=True),
        after.loc[after["signal_date"] <= cutoff, columns].reset_index(drop=True),
    )


def test_inferred_constraints_do_not_contain_company_snapshot_fields(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    result = infer_constraints(synthetic_prices, calendar)
    assert result["constraints_source"].eq("inferred").all()
    assert "stock_type" not in result.columns
    assert "mktcap" not in result.columns

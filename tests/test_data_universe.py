from barra_quant import __version__

import pandas as pd
import pytest

from barra_quant.data_universe import make_signal_dates, validate_prices


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

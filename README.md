# China stock data pipeline

This project contains a minimal, reproducible pipeline for the public
[`guidebee/china-stock-data`](https://github.com/guidebee/china-stock-data) dataset:

```text
upstream Git snapshot
  -> data/raw/china-stock-data
  -> data/normalized/china-stock-data/*.parquet
  -> data/warehouse/china_stock.duckdb
```

## Data contract

- `prices`: daily OHLCV keyed by `(symbol, trade_date)`. Symbols retain the upstream
  exchange prefix (`sh`, `sz`, `bj`). Prices and turnover are CNY; volume is shares.
- `company_snapshot_current`: the upstream weekly company snapshot. It contains
  current price, market cap, float market cap and turnover ratio. It is explicitly
  marked `pit_safe = false`.
- `pipeline_metadata`: source URL, exact Git commit, source commit time, build time,
  row counts and date coverage.

The company snapshot is **not** a historical security master. Do not use it to build
a historical universe: doing so would introduce survivorship and point-in-time bias.
The price files contain unadjusted-looking OHLCV and no corporate-action metadata;
the pipeline preserves the source values and does not claim they are adjusted.

## Run

Use the project's required `tsffids` environment:

```powershell
conda activate tsffids
python scripts/china_stock_pipeline.py all
```

Individual stages are also available:

```powershell
python scripts/china_stock_pipeline.py download
python scripts/china_stock_pipeline.py normalize
python scripts/china_stock_pipeline.py load
```

`download` creates a shallow sparse checkout and later uses `git pull --ff-only`.
`normalize` rebuilds partitioned Zstandard Parquet (`trade_year` / `trade_month`) from
the complete raw snapshot. `load` creates a DuckDB database with unique indexes and
rechecks row counts and data quality.

## Query

```python
import duckdb

con = duckdb.connect("data/warehouse/china_stock.duckdb", read_only=True)
bars = con.execute("""
    SELECT trade_date, symbol, close, volume
    FROM prices
    WHERE symbol = 'sh600000'
    ORDER BY trade_date
""").df()
```

## Validation

The build fails on duplicate `(symbol, trade_date)` keys, null required values,
invalid symbols, negative volume/amount, inconsistent OHLC ranges, or a mismatch
between the date stored in a row and its source filename.

Run the regression tests with:

```powershell
python -m unittest discover -s tests -v
```

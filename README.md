# China stock data pipeline

This project contains a minimal, reproducible pipeline for the public
[`guidebee/china-stock-data`](https://github.com/guidebee/china-stock-data) dataset:

```text
upstream Git snapshot
  -> data/raw/_checkouts/china-stock-data
  -> data/raw/china-stock-data/<source_commit>
  -> data/normalized/china-stock-data/<source_commit>/*.parquet
  -> data/warehouse/china_stock_<source_commit>.duckdb
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
python -m pip install -e . --no-deps
python scripts/china_stock_pipeline.py all
```

Individual stages are also available:

```powershell
python scripts/china_stock_pipeline.py download
python scripts/china_stock_pipeline.py normalize --source-commit <source_commit>
python scripts/china_stock_pipeline.py load --source-commit <source_commit>
```

`download` creates a shallow sparse checkout and later uses `git pull --ff-only`.
`normalize` rebuilds partitioned Zstandard Parquet (`trade_year` / `trade_month`) from
the complete raw snapshot. `load` creates a DuckDB database with unique indexes and
rechecks row counts and data quality.

## Query

```python
import duckdb

con = duckdb.connect(
    "data/warehouse/china_stock_<source_commit>.duckdb",
    read_only=True,
)
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
python -m pytest tests -v
```

## V1a strategy smoke run

V1a validates engineering behavior only. It does not establish that any factor or
portfolio is effective. The bundled source lacks complete point-in-time security
status and corporate actions.

```powershell
conda activate tsffids
python -m pip install -e . --no-deps
python scripts/china_stock_pipeline.py all
python -m barra_quant.run --config configs/v1a_smoke.json
python -m pytest tests -v
```

Consume a run only when `artifacts/runs/<run_id>/_SUCCESS` exists. Inspect
`report.md`, `metrics.json`, `orders.parquet`, `fills.parquet`, `positions.parquet`,
and `nav.parquet` together; a positive return in this short sample is not research
evidence.

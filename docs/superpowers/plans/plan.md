# A-Share Cross-Sectional Factor Strategy V1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic V1a smoke-test pipeline that turns the existing A-share daily bars into point-in-time-guarded factor research, a Top-50 long-only target portfolio, a conservative stateful A-share execution simulation, and auditable run artifacts without claiming factor validity.

**Architecture:** Keep the runtime surface to four focused modules plus a thin runner: `data_universe.py`, `research.py`, `execution.py`, `reporting.py`, and `run.py`. Research calculations remain vectorized in pandas/NumPy; order execution and account state advance one market day at a time. The existing data ingestion script is first changed to preserve immutable source snapshots, and every successful strategy run is content-addressed and atomically published.

**Tech Stack:** Python 3.11 in `tsffids`, DuckDB 1.5.x, pandas 2.3.x, NumPy 2.x, SciPy 1.17.x, statsmodels 0.14.x, Matplotlib 3.10.x, pytest 9.x.

**Spec:** `docs/superpowers/specs/2026-09-11-cross-sectional-factor-strategy-framework-design.md`

## Global Constraints

- Execute Python with `E:\0KD\envs\tsffids\python.exe` or from an activated `tsffids` environment.
- V1a uses the current 2026-02-10 through 2026-06-05 dataset only as a development fixture and always reports `SMOKE_TEST`.
- Never assert that real-data return, IC, ICIR, Sharpe, or alpha is positive.
- The decision time is T-day close; signals are generated after T close and orders are evaluated at T+1 open.
- V1a infers `available_at = trade_date 15:30 Asia/Shanghai` and records `availability_source = inferred_for_smoke_test`.
- Do not read `company_snapshot_current` when constructing historical universes, factors, targets, or constraints.
- Do not read T+1 high, low, close, amount, or full-day volume when deciding a T+1 open fill.
- V1a may infer board rules and calendar only when it records `constraints_source = inferred` and `calendar_source = inferred_from_prices`.
- Add statsmodels as the only new analytical dependency; do not add Polars, PyArrow, a backtesting framework, an ML framework, or a Broker abstraction.
- V1a supports one strategy and one RMB cash account; no ML, complete Barra model, PaperBroker, real-time service, multi-account support, or dashboard.
- The report records the fixed factor hypothesis, economic rationale, direction, formulas, parameters, primary horizon, source version, and rejection criteria before showing results; failed smoke runs remain auditable.
- The V1b 60/20/20 chronological split and `RESEARCH_CANDIDATE` gate are reported as `NOT_EVALUATED_IN_V1A`, because this short dataset cannot meet the five-year/200-week qualification gate.
- Raw source snapshots are append-only and keyed by the full source commit. The same commit must never be overwritten.
- Use TDD: add one focused failing test, observe the expected failure, implement the smallest behavior, rerun the focused test, then run the affected test file.
- The workspace currently has no `.git`. Do not initialize Git without user authorization. Each commit step is mandatory when Git exists; otherwise record the checkpoint as skipped because no repository exists.

---

## File and Interface Map

### Existing files to modify

- `scripts/china_stock_pipeline.py`: replace mutable raw checkout behavior with immutable snapshot materialization; make normalized and warehouse outputs source-versioned.
- `tests/test_china_stock_pipeline.py`: preserve the current normalization checks and add snapshot immutability/provenance tests.
- `requirements.txt`: record the exact dependency families required by V1a.
- `README.md`: document snapshot, strategy run, smoke-test, and audit commands.

### Files to create

- `pytest.ini`: expose `src` to pytest and define test discovery.
- `configs/v1a_smoke.json`: checked-in, frozen V1a assumptions.
- `src/barra_quant/__init__.py`: package version only.
- `src/barra_quant/data_universe.py`: price loading, time contract, inferred calendar, board mapping, base universe, and inferred constraints.
- `src/barra_quant/research.py`: fixed factors, labels, cross-sectional scores, targets, IC/quantile diagnostics, and Barra-lite.
- `src/barra_quant/execution.py`: fee rules, lots, order sizing, fills, account replay, and day-by-day simulation.
- `src/barra_quant/reporting.py`: metrics, figures, Markdown, content hashes, and atomic artifact publishing.
- `src/barra_quant/run.py`: thin orchestration entry point.
- `tests/conftest.py`: deterministic synthetic multi-symbol daily bars and shared fixtures.
- `tests/test_data_universe.py`: time and universe invariants.
- `tests/test_research.py`: factor, label, score, diagnostic, and Barra-lite behavior.
- `tests/test_execution.py`: A-share execution and accounting behavior.
- `tests/test_reporting.py`: metrics, identity, and atomic artifacts.
- `tests/test_v1a_integration.py`: synthetic end-to-end and current-data smoke tests.

### Stable interfaces

The tasks below must preserve these exact public signatures and return contracts:

| Module | Interface | Return contract |
|---|---|---|
| `data_universe.py` | `load_data(database: Path, source_manifest_path: Path)` | `DataBundle(prices, calendar, source_manifest)` |
| `data_universe.py` | `classify_board(symbol: str)` | one of `MAIN`, `STAR`, `CHINEXT`, `BSE`, `EXCLUDED` |
| `data_universe.py` | `make_signal_dates(calendar: pd.DatetimeIndex)` | last observed market date of each calendar week |
| `data_universe.py` | `build_base_universe(prices, signal_dates, config)` | unique `(signal_date, symbol)` eligibility rows |
| `data_universe.py` | `infer_constraints(prices, calendar)` | unique `(trade_date, symbol)` inferred tradability rows |
| `research.py` | `build_research(prices, calendar, base_universe, universe_config, config, run_seed)` | `ResearchResult(panel, diagnostics, benchmark)` |
| `execution.py` | `simulate(prices, calendar, targets, constraints, fee_rules, config, run_id)` | `BacktestResult(orders, fills, positions, nav)` |
| `execution.py` | `replay_fills(initial_cash: float, fills: pd.DataFrame)` | final cash plus symbol-sorted holdings |
| `reporting.py` | `compute_metrics(backtest, benchmark, trading_days=252)` | JSON-serializable metrics mapping |
| `reporting.py` | `compute_run_id(config, source_manifest, source_files)` | 24-character content-derived identifier |
| `reporting.py` | `publish_run(output_root, run_id, config, source_manifest, research, backtest, metrics)` | completed immutable run directory |
| `run.py` | `run_v1a(config_path: Path)` | completed immutable run directory |

`UniverseConfig`, `ResearchConfig`, `FeeRule`, `ExecutionConfig`, `DataBundle`, `ResearchResult`, and `BacktestResult` are defined exactly in their implementation tasks. Callers must not introduce alternate names or argument orders.

---

### Task 1: Establish the Testable Python Package and Frozen V1a Configuration

**Files:**
- Create: `pytest.ini`
- Create: `configs/v1a_smoke.json`
- Create: `src/barra_quant/__init__.py`
- Create: `tests/conftest.py`
- Modify: `requirements.txt:1`
- Test: `tests/test_data_universe.py`

**Interfaces:**
- Consumes: existing `data/warehouse/china_stock.duckdb` and Python 3.11 `tsffids` environment.
- Produces: importable `barra_quant` package, deterministic synthetic bars, and the exact checked-in configuration consumed by all later tasks.

- [ ] **Step 1: Write the package import test**

Create `tests/test_data_universe.py` with:

```python
from barra_quant import __version__


def test_package_version_marks_v1a():
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run the import test and observe the missing package**

Run:

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_data_universe.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'barra_quant'`.

- [ ] **Step 3: Add the package, pytest path, dependencies, fixture, and frozen config**

Create `src/barra_quant/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `pytest.ini`:

```ini
[pytest]
pythonpath = src
testpaths = tests
```

Replace `requirements.txt` with:

```text
duckdb>=1.5,<2
matplotlib>=3.10,<4
numpy>=2.0,<3
pandas>=2.3,<3
pytest>=9,<10
scipy>=1.17,<2
statsmodels>=0.14,<1
```

Create `configs/v1a_smoke.json`:

```json
{
  "database": "data/warehouse/china_stock.duckdb",
  "source_manifest": "data/normalized/china-stock-data/manifest.json",
  "output_root": "artifacts/runs",
  "mode": "SMOKE_TEST",
  "initial_cash": 10000000.0,
  "top_n": 50,
  "min_cross_section": 100,
  "min_history_days": 60,
  "liquidity_window": 20,
  "max_position_to_adv": 0.01,
  "min_avg_amount": 20000000.0,
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
  "risk_free_rate": 0.0
}
```

Create `tests/conftest.py` with deterministic, non-random synthetic prices:

```python
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
```

- [ ] **Step 4: Install only the declared dependency delta and run the test**

Run after approval for environment writes:

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pip install "statsmodels>=0.14,<1"
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_data_universe.py -v
```

Expected: package test passes and no additional package is deliberately introduced by the project.

- [ ] **Step 5: Create the task checkpoint**

Run:

```powershell
git rev-parse --is-inside-work-tree
git add requirements.txt pytest.ini configs/v1a_smoke.json src/barra_quant/__init__.py tests/conftest.py tests/test_data_universe.py
git commit -m "chore: establish V1a test configuration"
```

Expected: commit succeeds when Git exists. In the current non-Git workspace, do not initialize a repository; record `Task 1 checkpoint: commit skipped because .git is absent`.

---

### Task 2: Preserve Immutable Source Snapshots and File Hashes

**Files:**
- Modify: `scripts/china_stock_pipeline.py:29-379`
- Modify: `tests/test_china_stock_pipeline.py:1-80`
- Modify: `configs/v1a_smoke.json`

**Interfaces:**
- Consumes: `SourceVersion(commit: str, committed_at: str)` and a downloaded source checkout.
- Produces: `materialize_snapshot(source_data, snapshot_root, version) -> Path`, `sha256_file(path) -> str`, versioned normalized output, and a manifest containing file hashes.

- [ ] **Step 1: Add failing immutability and provenance tests**

Append these tests to `tests/test_china_stock_pipeline.py`:

```python
from scripts.china_stock_pipeline import materialize_snapshot, sha256_file


def test_materialize_snapshot_is_immutable(tmp_path):
    source = tmp_path / "checkout" / "data"
    source.mkdir(parents=True)
    raw_file = source / "sample.csv"
    raw_file.write_text("a,b\n1,2\n", encoding="utf-8")
    version = SourceVersion("abc123", "2026-06-05T15:15:04+08:00")

    snapshot = materialize_snapshot(source, tmp_path / "raw", version)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))

    assert snapshot.name == "abc123"
    assert manifest["files"][0]["sha256"] == sha256_file(snapshot / "data" / "sample.csv")
    raw_file.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable snapshot"):
        materialize_snapshot(source, tmp_path / "raw", version)


def test_normalized_path_is_keyed_by_source_commit(tmp_path):
    raw, normalized, database = project_paths(tmp_path, source_commit="abc123")
    assert raw == tmp_path / "data" / "raw" / "china-stock-data" / "abc123"
    assert normalized == tmp_path / "data" / "normalized" / "china-stock-data" / "abc123"
    assert database == tmp_path / "data" / "warehouse" / "china_stock_abc123.duckdb"
```

Also add `import pytest` and import `project_paths` at the top of the file.

- [ ] **Step 2: Run the snapshot tests and observe missing interfaces**

Run:

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_china_stock_pipeline.py -k "immutable or keyed" -v
```

Expected: collection fails because `materialize_snapshot` and `sha256_file` do not exist and `project_paths` lacks `source_commit`.

- [ ] **Step 3: Implement content hashing and immutable materialization**

Add these concrete functions to `scripts/china_stock_pipeline.py` and change `project_paths` to the tested signature:

```python
import hashlib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_snapshot(
    source_data: Path,
    snapshot_root: Path,
    version: SourceVersion,
) -> Path:
    destination = snapshot_root / version.commit
    files = sorted(path for path in source_data.rglob("*") if path.is_file())
    expected = [
        {
            "path": (Path("data") / path.relative_to(source_data)).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest["files"] != expected:
            raise ValueError(f"immutable snapshot mismatch: {destination}")
        return destination

    staging = snapshot_root / f".{version.commit}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(source_data, staging / "data")
    manifest = {
        "source_repo": REPO_URL,
        "source_commit": version.commit,
        "source_committed_at": version.committed_at,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "files": expected,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    snapshot_root.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return destination


def project_paths(
    project_root: Path,
    source_commit: str,
) -> tuple[Path, Path, Path]:
    return (
        project_root / "data" / "raw" / "china-stock-data" / source_commit,
        project_root / "data" / "normalized" / "china-stock-data" / source_commit,
        project_root / "data" / "warehouse" / f"china_stock_{source_commit}.duckdb",
    )
```

Change the CLI workflow so the mutable Git checkout lives only at
`data/raw/_checkouts/china-stock-data`. After clone or fast-forward, call
`materialize_snapshot(checkout / "data", data/raw/china-stock-data, source_version(checkout))`.
When the current legacy path contains `.git`, move it once to
`data/raw/_checkouts/china-stock-data` only after resolving both paths under the project root and verifying the destination does not exist. Do not delete the legacy checkout.

Make `normalize` consume `<snapshot>/data`, copy the raw snapshot manifest fields into the normalized manifest, and write under the matching commit directory. Make `load` create the commit-keyed DuckDB path. Add `--source-commit` to `normalize` and `load`; `all` uses the commit returned by the snapshot stage.

- [ ] **Step 4: Run pipeline tests and inspect the real snapshot migration plan**

Run:

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_china_stock_pipeline.py -v
& 'E:\0KD\envs\tsffids\python.exe' scripts\china_stock_pipeline.py --help
```

Expected: all pipeline tests pass; help lists `--source-commit`; no network call or real data move occurs during tests.

Update `configs/v1a_smoke.json` so `database` and `source_manifest` point to the full current commit
`08c4692f2b56d3ae47feae93de5576d4c626be8a` after the real snapshot command is run during Task 10.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add scripts/china_stock_pipeline.py tests/test_china_stock_pipeline.py configs/v1a_smoke.json
git commit -m "feat: preserve immutable market data snapshots"
```

Expected: commit succeeds in a Git workspace; otherwise record the skipped checkpoint without initializing Git.

---

### Task 3: Enforce the Price Time Contract and Inferred Market Calendar

**Files:**
- Create: `src/barra_quant/data_universe.py`
- Modify: `tests/test_data_universe.py`

**Interfaces:**
- Consumes: commit-keyed DuckDB `prices` table and normalized `manifest.json`.
- Produces: `UniverseConfig`, `DataBundle`, `load_data`, `validate_prices`, and `make_signal_dates` with the signatures in the interface map.

- [ ] **Step 1: Add failing time-contract tests**

Append to `tests/test_data_universe.py`:

```python
import pandas as pd
import pytest

from barra_quant.data_universe import make_signal_dates, validate_prices


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
```

- [ ] **Step 2: Run the focused tests and observe the missing module**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_data_universe.py -k "availability or duplicate or signal_date" -v
```

Expected: collection fails because `barra_quant.data_universe` does not exist.

- [ ] **Step 3: Implement validation, loading, and signal dates**

Create `src/barra_quant/data_universe.py` with the two dataclasses from the interface map and these functions:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import duckdb
import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "symbol", "trade_date", "open", "close", "high", "low",
    "volume", "amount", "source_file", "source_commit",
}


@dataclass(frozen=True)
class UniverseConfig:
    initial_cash: float = 10_000_000.0
    top_n: int = 50
    min_cross_section: int = 100
    min_history_days: int = 60
    liquidity_window: int = 20
    max_position_to_adv: float = 0.01
    min_avg_amount: float = 20_000_000.0


@dataclass(frozen=True)
class DataBundle:
    prices: pd.DataFrame
    calendar: pd.DatetimeIndex
    source_manifest: dict[str, object]


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"missing price columns: {sorted(missing)}")
    if prices.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("duplicate symbol trade_date keys")
    result = prices.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.normalize()
    result = result.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)
    localized = result["trade_date"].dt.tz_localize("Asia/Shanghai")
    result["available_at"] = localized + pd.Timedelta(hours=15, minutes=30)
    result["availability_source"] = "inferred_for_smoke_test"
    return result


def make_signal_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    frame = pd.DataFrame({"trade_date": pd.DatetimeIndex(calendar).sort_values().unique()})
    frame["week"] = frame["trade_date"].dt.to_period("W-FRI")
    return pd.DatetimeIndex(frame.groupby("week", sort=True)["trade_date"].max())


def load_data(database: Path, source_manifest_path: Path) -> DataBundle:
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    with duckdb.connect(str(database), read_only=True) as connection:
        prices = connection.execute("SELECT * FROM prices ORDER BY trade_date, symbol").df()
    prices = validate_prices(prices)
    commits = set(prices["source_commit"].unique())
    if commits != {manifest["source_commit"]}:
        raise ValueError(f"source commit mismatch: prices={commits}, manifest={manifest['source_commit']}")
    calendar = pd.DatetimeIndex(prices["trade_date"].drop_duplicates().sort_values())
    manifest = dict(manifest)
    manifest["calendar_source"] = "inferred_from_prices"
    manifest["price_basis"] = "raw_unadjusted_smoke_test"
    manifest["corporate_actions_complete"] = False
    return DataBundle(prices=prices, calendar=calendar, source_manifest=manifest)
```

- [ ] **Step 4: Run the data contract tests**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_data_universe.py -v
```

Expected: all Task 1 and Task 3 tests pass.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/data_universe.py tests/test_data_universe.py
git commit -m "feat: enforce V1a market data time contract"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 4: Build the PIT-Approximate Universe and Effective Daily Constraints

**Files:**
- Modify: `src/barra_quant/data_universe.py`
- Modify: `tests/test_data_universe.py`

**Interfaces:**
- Consumes: validated price rows, inferred calendar, signal dates, and `UniverseConfig`.
- Produces: `classify_board`, `build_base_universe`, and `infer_constraints`; base-universe rows contain `signal_date`, `symbol`, `board`, `history_days`, `adv20_amount`, `adv20_volume`, `base_eligible`, and `universe_reason`.

- [ ] **Step 1: Add failing board, liquidity, and no-current-snapshot tests**

Append:

```python
from barra_quant.data_universe import (
    UniverseConfig,
    build_base_universe,
    classify_board,
    infer_constraints,
    make_signal_dates,
)


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
```

- [ ] **Step 2: Run the new tests and observe missing functions**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_data_universe.py -k "board or universe or constraints" -v
```

Expected: imports or assertions fail because the universe and constraints functions are not implemented.

- [ ] **Step 3: Implement board mapping, trailing-only liquidity, and inferred rules**

Add:

```python
import numpy as np


def classify_board(symbol: str) -> str:
    value = symbol.lower()
    if value.startswith("sh900") or value.startswith("sz200"):
        return "EXCLUDED_B_SHARE"
    if value.startswith(("sh688", "sh689")):
        return "STAR"
    if value.startswith(("sz300", "sz301")):
        return "CHINEXT"
    if value.startswith("bj"):
        return "BSE"
    if value.startswith(("sh", "sz")) and value[2:].isdigit() and len(value) == 8:
        return "MAIN"
    return "INVALID"


def build_base_universe(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: UniverseConfig,
) -> pd.DataFrame:
    ordered = prices.sort_values(["symbol", "trade_date"], kind="mergesort").copy()
    grouped = ordered.groupby("symbol", sort=False)
    ordered["history_days"] = grouped.cumcount() + 1
    ordered["adv20_amount"] = grouped["amount"].transform(
        lambda values: values.rolling(config.liquidity_window, min_periods=config.liquidity_window).mean()
    )
    ordered["adv20_volume"] = grouped["volume"].transform(
        lambda values: values.rolling(config.liquidity_window, min_periods=config.liquidity_window).mean()
    )
    panel = ordered[ordered["trade_date"].isin(signal_dates)].copy()
    panel = panel.rename(columns={"trade_date": "signal_date"})
    panel["board"] = panel["symbol"].map(classify_board)
    target_value = config.initial_cash / config.top_n
    panel["base_eligible"] = (
        panel["board"].isin(["MAIN", "STAR", "CHINEXT", "BSE"])
        & panel["history_days"].ge(config.min_history_days)
        & panel["adv20_amount"].ge(config.min_avg_amount)
        & target_value.le(panel["adv20_amount"] * config.max_position_to_adv)
    )
    panel["universe_reason"] = np.select(
        [
            panel["board"].isin(["EXCLUDED_B_SHARE", "INVALID"]),
            panel["history_days"].lt(config.min_history_days),
            panel["adv20_amount"].lt(config.min_avg_amount),
            target_value.gt(panel["adv20_amount"] * config.max_position_to_adv),
        ],
        ["MARKET_EXCLUDED", "INSUFFICIENT_HISTORY", "LOW_ABSOLUTE_LIQUIDITY", "LOW_CAPACITY"],
        default="ELIGIBLE",
    )
    return panel[[
        "signal_date", "symbol", "board", "history_days", "adv20_amount",
        "adv20_volume", "base_eligible", "universe_reason",
    ]].sort_values(["signal_date", "symbol"], kind="mergesort").reset_index(drop=True)


def infer_constraints(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    result = prices[["trade_date", "symbol", "open", "close"]].copy()
    result["board"] = result["symbol"].map(classify_board)
    previous = result.sort_values(["symbol", "trade_date"]).groupby("symbol")["close"].shift(1)
    rate = result["board"].map({"MAIN": 0.10, "CHINEXT": 0.20, "STAR": 0.20, "BSE": 0.30})
    result["is_tradable"] = True
    result["limit_up"] = (previous * (1.0 + rate)).round(2)
    result["limit_down"] = (previous * (1.0 - rate)).round(2)
    result["min_buy_qty"] = result["board"].map({"MAIN": 100, "CHINEXT": 100, "STAR": 200, "BSE": 100})
    result["qty_step"] = result["board"].map({"MAIN": 100, "CHINEXT": 100, "STAR": 1, "BSE": 1})
    result["price_tick"] = 0.01
    result["constraints_source"] = "inferred"
    result["source_version"] = result.get("source_commit", "v1a-inferred")
    return result[[
        "trade_date", "symbol", "is_tradable", "limit_up", "limit_down",
        "min_buy_qty", "qty_step", "price_tick", "constraints_source", "source_version",
    ]]
```

Do not query DuckDB inside either function; this ensures `company_snapshot_current` cannot enter the path.

- [ ] **Step 4: Run the full universe tests**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_data_universe.py -v
```

Expected: mapping, trailing-data invariance, liquidity, and constraints tests pass.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/data_universe.py tests/test_data_universe.py
git commit -m "feat: build PIT-guarded V1a universe"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 5: Compute Fixed Factors, Exact-Market-Day Labels, Scores, and Targets

**Files:**
- Create: `src/barra_quant/research.py`
- Create: `tests/test_research.py`

**Interfaces:**
- Consumes: validated prices, market calendar, base-universe panel, `UniverseConfig`, and `ResearchConfig`.
- Produces: factor/label rows keyed by `(signal_date, symbol)`, deterministic `signal_id`, `target_weight`, and the `ResearchResult` dataclass.

- [ ] **Step 1: Add failing formula, missing-endpoint, and target tests**

Create `tests/test_research.py`:

```python
import pandas as pd

from barra_quant.data_universe import UniverseConfig, build_base_universe, make_signal_dates
from barra_quant.research import ResearchConfig, compute_factor_label_panel, score_and_target


def test_factor_formulas_use_exact_market_offsets(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    signal_date = calendar[80]
    panel = compute_factor_label_panel(
        synthetic_prices,
        calendar,
        pd.DatetimeIndex([signal_date]),
        ResearchConfig(),
    )
    symbol = synthetic_prices["symbol"].iloc[0]
    row = panel.loc[panel["symbol"] == symbol].iloc[0]
    prices = synthetic_prices.set_index(["trade_date", "symbol"])
    expected_momentum = (
        prices.loc[(calendar[75], symbol), "close"]
        / prices.loc[(calendar[60], symbol), "close"] - 1.0
    )
    expected_reversal = -(
        prices.loc[(calendar[80], symbol), "close"]
        / prices.loc[(calendar[75], symbol), "close"] - 1.0
    )
    assert row["momentum_20_5"] == expected_momentum
    assert row["reversal_5"] == expected_reversal


def test_label_does_not_skip_missing_symbol_day(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    symbol = synthetic_prices["symbol"].iloc[0]
    signal_date = calendar[80]
    missing_start = calendar[81]
    prices = synthetic_prices.loc[
        ~((synthetic_prices["symbol"] == symbol) & (synthetic_prices["trade_date"] == missing_start))
    ]
    panel = compute_factor_label_panel(prices, calendar, pd.DatetimeIndex([signal_date]), ResearchConfig())
    assert pd.isna(panel.loc[panel["symbol"] == symbol, "forward_return_5"].iloc[0])


def test_target_requires_one_hundred_complete_candidates(synthetic_prices):
    calendar = pd.DatetimeIndex(synthetic_prices["trade_date"].unique()).sort_values()
    signal_dates = pd.DatetimeIndex([calendar[80]])
    base = build_base_universe(synthetic_prices, signal_dates, UniverseConfig())
    factors = compute_factor_label_panel(synthetic_prices, calendar, signal_dates, ResearchConfig())
    result = score_and_target(factors, base, UniverseConfig(), run_seed="fixed")
    selected = result.loc[result["target_weight"] > 0]
    assert len(selected) == 50
    assert selected["target_weight"].eq(0.02).all()
    assert selected["signal_id"].is_unique
```

- [ ] **Step 2: Run the research tests and observe missing interfaces**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_research.py -v
```

Expected: collection fails because `barra_quant.research` does not exist.

- [ ] **Step 3: Implement fixed vectorized formulas and cross-sectional scoring**

Create `src/barra_quant/research.py` with the `ResearchConfig` and `ResearchResult` dataclasses from the interface map, then implement:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib

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


def _long_at_signal_dates(wide: pd.DataFrame, name: str, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    selected = wide.reindex(signal_dates)
    return selected.rename_axis(index="signal_date", columns="symbol").stack(dropna=False).rename(name).reset_index()


def compute_factor_label_panel(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    signal_dates: pd.DatetimeIndex,
    config: ResearchConfig,
) -> pd.DataFrame:
    close = prices.pivot(index="trade_date", columns="symbol", values="close").reindex(calendar)
    open_price = prices.pivot(index="trade_date", columns="symbol", values="open").reindex(calendar)
    log_return = np.log(close).diff()
    series = {
        "momentum_20_5": close.shift(5) / close.shift(20) - 1.0,
        "reversal_5": -(close / close.shift(5) - 1.0),
        "low_volatility_20": -log_return.rolling(20, min_periods=20).std(ddof=1),
    }
    for horizon in config.horizons:
        series[f"forward_return_{horizon}"] = open_price.shift(-(horizon + 1)) / open_price.shift(-1) - 1.0
    frames = [_long_at_signal_dates(value, name, signal_dates) for name, value in series.items()]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["signal_date", "symbol"], how="outer", validate="one_to_one")
    return result.sort_values(["signal_date", "symbol"], kind="mergesort").reset_index(drop=True)


def _rank_factor(group: pd.DataFrame, factor: str, lower: float, upper: float) -> pd.Series:
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
    panel = factors.merge(base_universe, on=["signal_date", "symbol"], how="left", validate="one_to_one")
    for factor in FACTOR_COLUMNS:
        score_column = f"{factor}_score"
        panel[score_column] = np.nan
        for group_index in panel.groupby("signal_date", sort=True).groups.values():
            eligible_index = panel.index[group_index][panel.loc[group_index, "base_eligible"].fillna(False)]
            panel.loc[eligible_index, score_column] = _rank_factor(
                panel.loc[eligible_index], factor, 0.01, 0.99
            ).to_numpy()
    scores = [f"{factor}_score" for factor in FACTOR_COLUMNS]
    panel["factor_complete"] = panel[scores].notna().all(axis=1)
    panel["eligible"] = panel["base_eligible"].fillna(False) & panel["factor_complete"]
    panel["composite_score"] = panel[scores].mean(axis=1).where(panel["eligible"])
    panel["composite_rank"] = np.nan
    eligible = panel.loc[panel["eligible"]].sort_values(
        ["signal_date", "composite_score", "symbol"], ascending=[True, False, True], kind="mergesort"
    )
    panel.loc[eligible.index, "composite_rank"] = eligible.groupby("signal_date", sort=True).cumcount() + 1
    counts = panel.groupby("signal_date")["eligible"].transform("sum")
    panel["target_weight"] = np.where(
        panel["eligible"] & counts.ge(config.min_cross_section) & panel["composite_rank"].le(config.top_n),
        1.0 / config.top_n,
        0.0,
    )
    panel["signal_id"] = panel.apply(
        lambda row: hashlib.sha256(
            f"{run_seed}|{row['signal_date']:%Y-%m-%d}|{row['symbol']}".encode("utf-8")
        ).hexdigest()[:24],
        axis=1,
    )
    panel["signal_status"] = np.where(counts.ge(config.min_cross_section), "READY", "INSUFFICIENT_UNIVERSE")
    return panel.sort_values(["signal_date", "symbol"], kind="mergesort").reset_index(drop=True)
```

- [ ] **Step 4: Run formulas, no-skip labels, and target tests**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_research.py -v
```

Expected: formula values match exact market offsets, missing T+1 open yields a missing label, and exactly 50 of at least 100 candidates receive 2% targets.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/research.py tests/test_research.py
git commit -m "feat: compute fixed V1a factors and targets"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 6: Add RankIC, HAC, Quantiles, Benchmark, and Barra-Lite Diagnostics

**Files:**
- Modify: `src/barra_quant/research.py`
- Modify: `tests/test_research.py`

**Interfaces:**
- Consumes: scored research panel and validated prices.
- Produces: `compute_rank_ic`, `hac_ic_summary`, `compute_quantile_returns`, `compute_barra_lite`, `residualize_factor`, `build_benchmark`, and the completed `build_research` entry point.

- [ ] **Step 1: Add failing statistical and unavailable-exposure tests**

Append:

```python
from barra_quant.research import (
    build_benchmark,
    compute_barra_lite,
    compute_rank_ic,
    hac_ic_summary,
)


def test_rank_ic_is_cross_sectional_by_signal_date():
    panel = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-01-02"] * 4 + ["2026-01-09"] * 4),
        "symbol": ["a", "b", "c", "d"] * 2,
        "factor": [1, 2, 3, 4, 4, 3, 2, 1],
        "forward_return_5": [0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01],
    })
    result = compute_rank_ic(panel, "factor", 5)
    assert result["rank_ic"].tolist() == [1.0, 1.0]


def test_hac_lag_uses_weekly_overlap():
    ic = pd.Series([0.01, 0.03, -0.01, 0.02, 0.04])
    summary_5 = hac_ic_summary(ic, horizon=5)
    summary_20 = hac_ic_summary(ic, horizon=20)
    assert summary_5["max_lags"] == 0
    assert summary_20["max_lags"] == 3


def test_barra_lite_does_not_shorten_beta_window(synthetic_prices):
    short = synthetic_prices[synthetic_prices["trade_date"].isin(
        sorted(synthetic_prices["trade_date"].unique())[:90]
    )]
    result = compute_barra_lite(short, ResearchConfig(beta_window=120, beta_min_observations=100))
    assert result["beta_120"].isna().all()
    assert result["residual_volatility_120"].isna().all()


def test_benchmark_is_named_gross_research_benchmark():
    panel = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
        "eligible": [True, True],
        "forward_return_5": [0.01, 0.03],
    })
    result = build_benchmark(panel, primary_horizon=5)
    assert result.iloc[0]["benchmark_name"] == "pit_universe_equal_weight_gross"
    assert result.iloc[0]["benchmark_return"] == 0.02
```

- [ ] **Step 2: Run diagnostics tests and observe missing interfaces**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_research.py -k "rank_ic or hac or barra or benchmark" -v
```

Expected: imports fail for the new diagnostic functions.

- [ ] **Step 3: Implement the diagnostic calculations and research entry point**

Add these critical implementations:

```python
import math
from scipy.stats import spearmanr
import statsmodels.api as sm


def compute_rank_ic(panel: pd.DataFrame, factor: str, horizon: int) -> pd.DataFrame:
    outcome = f"forward_return_{horizon}"
    rows: list[dict[str, object]] = []
    for signal_date, group in panel.dropna(subset=[factor, outcome]).groupby("signal_date", sort=True):
        value = spearmanr(group[factor], group[outcome]).statistic
        rows.append({"signal_date": signal_date, "factor": factor, "horizon": horizon, "rank_ic": value})
    return pd.DataFrame(rows)


def hac_ic_summary(ic: pd.Series, horizon: int) -> dict[str, float | int]:
    clean = ic.dropna().astype(float)
    max_lags = max(0, math.ceil(horizon / 5) - 1)
    if len(clean) < max(3, max_lags + 2):
        standard_deviation = float(clean.std(ddof=1))
        return {
            "observations": len(clean), "mean_ic": float(clean.mean()),
            "standard_deviation": standard_deviation,
            "icir": float(clean.mean() / standard_deviation) if standard_deviation > 0 else float("nan"),
            "hac_t": float("nan"), "max_lags": max_lags,
        }
    model = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lags}
    )
    standard_deviation = float(clean.std(ddof=1))
    return {
        "observations": len(clean),
        "mean_ic": float(clean.mean()),
        "standard_deviation": standard_deviation,
        "icir": float(clean.mean() / standard_deviation) if standard_deviation > 0 else float("nan"),
        "hac_t": float(model.tvalues[0]),
        "max_lags": max_lags,
    }


def build_benchmark(panel: pd.DataFrame, primary_horizon: int) -> pd.DataFrame:
    column = f"forward_return_{primary_horizon}"
    result = panel.loc[panel["eligible"]].groupby("signal_date", as_index=False)[column].mean()
    result = result.rename(columns={column: "benchmark_return"})
    result["benchmark_name"] = "pit_universe_equal_weight_gross"
    return result


def compute_quantile_returns(panel: pd.DataFrame, factor: str, horizon: int) -> pd.DataFrame:
    outcome = f"forward_return_{horizon}"
    work = panel.dropna(subset=[factor, outcome]).copy()
    work["quantile"] = work.groupby("signal_date")[factor].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False) + 1
    )
    return work.groupby(["signal_date", "quantile"], as_index=False)[outcome].mean()
```

Implement `compute_barra_lite` with exact market-day pivots: compute daily stock returns and the contemporaneous equal-weight eligible-market return, then for each symbol/date with 120 trailing market days and at least100 paired values fit `sm.OLS(stock_return, sm.add_constant(market_return))`. Save slope as `beta_120` and the sample standard deviation of residuals as `residual_volatility_120`; never reduce the window. Save `momentum_style_20_5` from the fixed factor and `liquidity_style_20 = log(rolling_mean(amount, 20))` only when all20 observations exist.

Implement `residualize_factor` as a per-date OLS of a factor score on an intercept plus the available standardized Barra-lite exposures; if fewer than30 complete rows or no exposure is available, return missing residuals and status `UNAVAILABLE`.

For every fixed factor plus `composite_score`, calculate and retain these diagnostics without silently dropping failed dates:

- Coverage and missingness per signal date: non-null eligible count divided by eligible count, plus explicit missing count.
- Extremes per signal date: minimum, 1st percentile, 99th percentile, and maximum before winsorization.
- Five equal-count quantile returns plus `top_minus_bottom = Q5 - Q1`; monotonicity is the Spearman correlation between quantile number and that date's quantile return.
- Turnover at each rebalance: `0.5 * sum(abs(weight_t - weight_previous))` over the union of symbols, treating absent weights as zero.
- Pairwise factor correlation: per-date Spearman rank correlation, followed by its time-series mean and observation count.
- Decay: the RankIC summary at horizons 1, 5, 10, and 20.
- Stability: chronological first-half/second-half and calendar-year RankIC summaries. Set market-regime status to `UNAVAILABLE_IN_V1A` because no independently defined regime series exists.

Implement `build_research` by calling the Task 5 factor/scoring functions, adding Barra-lite columns, computing all diagnostics above for every fixed factor plus `composite_score`, building the benchmark, and returning:

```python
ResearchResult(
    panel=panel,
    diagnostics={
        "mode": "SMOKE_TEST",
        "rank_ic": rank_ic_frame,
        "hac": hac_rows,
        "quantiles": quantile_frame,
        "coverage": coverage_frame,
        "extremes": extremes_frame,
        "turnover": turnover_frame,
        "factor_correlation": correlation_frame,
        "stability": stability_frame,
        "regime_status": "UNAVAILABLE_IN_V1A",
        "style_status": style_status,
        "unavailable_styles": [
            "size", "nonlinear_size", "value", "profitability",
            "growth", "leverage", "industry",
        ],
    },
    benchmark=build_benchmark(panel, config.primary_horizon),
)
```

- [ ] **Step 4: Run all research tests**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_research.py -v
```

Expected: formulas, labels, targets, cross-sectional IC, HAC lags, quantiles, benchmark naming, and unavailable Beta behavior pass. Warnings about constant synthetic inputs must be eliminated by improving the fixture, not by suppressing statistical warnings globally.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/research.py tests/test_research.py
git commit -m "feat: add factor diagnostics and Barra-lite audit"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 7: Implement A-Share Fee, Lot, Limit, and Order Primitives

**Files:**
- Create: `src/barra_quant/execution.py`
- Create: `tests/test_execution.py`

**Interfaces:**
- Consumes: target weights, inferred constraints, T-day trailing volume, and effective-dated fee rules.
- Produces: `FeeRule`, `ExecutionConfig`, `AcquisitionLot`, `BacktestResult`, `fee_for_fill`, `legal_quantity`, `fill_price`, and deterministic order identifiers.

- [ ] **Step 1: Add failing primitive tests**

Create `tests/test_execution.py`:

```python
from datetime import date

from barra_quant.execution import ExecutionConfig, FeeRule, fee_for_fill, fill_price, legal_quantity


FEE_RULE = FeeRule(
    effective_from=date(2023, 8, 28),
    effective_to=None,
    commission_bps=3.0,
    minimum_commission=5.0,
    stamp_duty_sell_bps=5.0,
    transfer_bps=0.1,
)


def test_fee_rule_applies_minimum_commission_and_sell_tax():
    buy = fee_for_fill(10_000.0, "BUY", date(2026, 3, 2), (FEE_RULE,))
    sell = fee_for_fill(10_000.0, "SELL", date(2026, 3, 2), (FEE_RULE,))
    assert buy == {"commission": 5.0, "stamp_duty": 0.0, "transfer_fee": 0.1}
    assert sell == {"commission": 5.0, "stamp_duty": 5.0, "transfer_fee": 0.1}


def test_board_quantity_rules():
    assert legal_quantity(375, min_buy_qty=100, qty_step=100, side="BUY") == 300
    assert legal_quantity(375, min_buy_qty=200, qty_step=1, side="BUY") == 375
    assert legal_quantity(175, min_buy_qty=200, qty_step=1, side="BUY") == 0
    assert legal_quantity(75, min_buy_qty=100, qty_step=100, side="SELL", position_quantity=75) == 75


def test_slippage_does_not_cross_price_limit():
    assert fill_price(10.0, "BUY", slippage_bps=20.0, limit_up=10.01, limit_down=9.0) == 10.01
    assert fill_price(10.0, "SELL", slippage_bps=20.0, limit_up=11.0, limit_down=9.99) == 9.99
```

- [ ] **Step 2: Run primitive tests and observe the missing module**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_execution.py -v
```

Expected: collection fails because `barra_quant.execution` does not exist.

- [ ] **Step 3: Implement exact primitive types and calculations**

Create `src/barra_quant/execution.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Literal

import pandas as pd


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class FeeRule:
    effective_from: date
    effective_to: date | None
    commission_bps: float
    minimum_commission: float
    stamp_duty_sell_bps: float
    transfer_bps: float


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000_000.0
    slippage_bps: float = 5.0
    volume_cap: float = 0.05


@dataclass
class AcquisitionLot:
    symbol: str
    quantity: int
    acquired_on: pd.Timestamp
    unlock_on: pd.Timestamp
    cost: float


@dataclass(frozen=True)
class BacktestResult:
    orders: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    nav: pd.DataFrame


def _active_rule(trade_date: date, rules: tuple[FeeRule, ...]) -> FeeRule:
    matches = [rule for rule in rules if rule.effective_from <= trade_date and (rule.effective_to is None or trade_date <= rule.effective_to)]
    if len(matches) != 1:
        raise ValueError(f"expected one fee rule for {trade_date}, found {len(matches)}")
    return matches[0]


def fee_for_fill(notional: float, side: Side, trade_date: date, rules: tuple[FeeRule, ...]) -> dict[str, float]:
    rule = _active_rule(trade_date, rules)
    return {
        "commission": max(rule.minimum_commission, notional * rule.commission_bps / 10_000.0),
        "stamp_duty": notional * rule.stamp_duty_sell_bps / 10_000.0 if side == "SELL" else 0.0,
        "transfer_fee": notional * rule.transfer_bps / 10_000.0,
    }


def legal_quantity(requested: int, min_buy_qty: int, qty_step: int, side: Side, position_quantity: int = 0) -> int:
    if side == "SELL" and requested == position_quantity and position_quantity < min_buy_qty:
        return position_quantity
    if requested < min_buy_qty:
        return 0
    return min_buy_qty + ((requested - min_buy_qty) // qty_step) * qty_step


def fill_price(open_price: float, side: Side, slippage_bps: float, limit_up: float, limit_down: float) -> float:
    multiplier = 1.0 + slippage_bps / 10_000.0 if side == "BUY" else 1.0 - slippage_bps / 10_000.0
    slipped = open_price * multiplier
    return round(min(slipped, limit_up), 2) if side == "BUY" else round(max(slipped, limit_down), 2)


def deterministic_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]
```

- [ ] **Step 4: Run primitive execution tests**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_execution.py -v
```

Expected: fees, minimum quantities, odd-lot liquidation, effective dates, slippage, and limit clamps pass exactly.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/execution.py tests/test_execution.py
git commit -m "feat: add A-share execution primitives"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 8: Build the Stateful T+1 Simulator and Idempotent Replay

**Files:**
- Modify: `src/barra_quant/execution.py`
- Modify: `tests/test_execution.py`

**Interfaces:**
- Consumes: `prices`, `calendar`, target rows, effective constraints, fee rules, `ExecutionConfig`, and `run_id`.
- Produces: `simulate(...) -> BacktestResult` and `replay_fills(...)`; orders use deterministic `client_order_id`, and positions expose total, sellable, locked, market value, and `stale_days`.

- [ ] **Step 1: Add failing T+1, limit, no-lookahead, replay, and NAV tests**

Append tests that use a three-symbol, four-day hand-built panel:

```python
import pandas as pd
import pytest

from barra_quant.execution import BacktestResult, simulate, replay_fills


def _simulation_inputs():
    dates = pd.bdate_range("2026-03-02", periods=4)
    prices = pd.DataFrame([
        {"trade_date": day, "symbol": "sh600000", "open": 10.0, "close": 10.0,
         "high": 10.2, "low": 9.8, "volume": 1_000_000, "amount": 10_000_000.0}
        for day in dates
    ])
    targets = pd.DataFrame([
        {"signal_date": dates[0], "execution_date": dates[1], "symbol": "sh600000",
         "target_weight": 1.0, "signal_id": "signal-buy", "adv20_volume": 1_000_000.0},
        {"signal_date": dates[1], "execution_date": dates[2], "symbol": "sh600000",
         "target_weight": 0.0, "signal_id": "signal-sell", "adv20_volume": 1_000_000.0},
    ])
    constraints = pd.DataFrame([
        {"trade_date": day, "symbol": "sh600000", "is_tradable": True,
         "limit_up": 11.0, "limit_down": 9.0, "min_buy_qty": 100,
         "qty_step": 100, "price_tick": 0.01, "constraints_source": "inferred",
         "source_version": "synthetic"}
        for day in dates
    ])
    return prices, pd.DatetimeIndex(dates), targets, constraints


def test_simulator_enforces_t1_and_nav_identity():
    dates = pd.bdate_range("2026-03-02", periods=4)
    prices = pd.DataFrame([
        {"trade_date": day, "symbol": "sh600000", "open": 10.0, "close": 10.0,
         "high": 10.2, "low": 9.8, "volume": 1_000_000, "amount": 10_000_000.0}
        for day in dates
    ])
    targets = pd.DataFrame([
        {"signal_date": dates[0], "execution_date": dates[1], "symbol": "sh600000",
         "target_weight": 1.0, "signal_id": "signal-buy", "adv20_volume": 1_000_000.0},
        {"signal_date": dates[1], "execution_date": dates[2], "symbol": "sh600000",
         "target_weight": 0.0, "signal_id": "signal-sell", "adv20_volume": 1_000_000.0},
    ])
    constraints = pd.DataFrame([
        {"trade_date": day, "symbol": "sh600000", "is_tradable": True,
         "limit_up": 11.0, "limit_down": 9.0, "min_buy_qty": 100,
         "qty_step": 100, "price_tick": 0.01, "constraints_source": "inferred",
         "source_version": "synthetic"}
        for day in dates
    ])
    result = simulate(
        prices, pd.DatetimeIndex(dates), targets, constraints, (FEE_RULE,),
        ExecutionConfig(initial_cash=1_000_000.0, slippage_bps=0.0, volume_cap=0.05),
        "run",
    )
    assert result.fills["client_order_id"].is_unique
    assert (result.nav["cash"] + result.nav["positions_value"] - result.nav["nav"]).abs().max() < 1e-8
    sell = result.fills[result.fills["side"] == "SELL"].iloc[0]
    assert sell["trade_date"] == dates[2]


def test_open_fill_does_not_change_when_future_intraday_columns_change():
    prices, calendar, targets, constraints = _simulation_inputs()
    before = simulate(prices, calendar, targets, constraints, (FEE_RULE,), ExecutionConfig(), "run")
    changed = prices.copy()
    changed["high"] *= 10
    changed["low"] *= 0.1
    changed["volume"] *= 100
    changed["amount"] *= 100
    after = simulate(changed, calendar, targets, constraints, (FEE_RULE,), ExecutionConfig(), "run")
    pd.testing.assert_frame_equal(before.fills, after.fills)


def test_replay_is_idempotent():
    prices, calendar, targets, constraints = _simulation_inputs()
    simulation_result = simulate(prices, calendar, targets, constraints, (FEE_RULE,), ExecutionConfig(), "run")
    duplicated = pd.concat([simulation_result.fills, simulation_result.fills], ignore_index=True)
    cash, holdings = replay_fills(1_000_000.0, duplicated)
    unique_cash, unique_holdings = replay_fills(1_000_000.0, simulation_result.fills)
    assert cash == unique_cash
    pd.testing.assert_frame_equal(holdings, unique_holdings)
```

Add a parameterized `test_rejection_statuses` that copies `_simulation_inputs()` and mutates exactly one input per case: set open equal to `limit_up` for `LIMIT_UP`, set open equal to `limit_down` on a sell for `LIMIT_DOWN`, delete the execution bar for `NO_BAR`, request fewer than the minimum lot for `LOT_TOO_SMALL`, set trailing ADV below one legal lot for `VOLUME_CAP`, and reduce cash below one lot plus fees for `INSUFFICIENT_CASH`. Assert both the terminal status and that rejected orders create no fill.

- [ ] **Step 2: Run simulator tests and observe the missing functions**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_execution.py -k "simulator or open_fill or replay or LIMIT" -v
```

Expected: tests fail because `simulate` and `replay_fills` are not implemented.

- [ ] **Step 3: Implement a deterministic daily state loop**

Implement `simulate` in `execution.py` with the following fixed event order for every `trade_date` in `calendar`:

1. Snapshot opening cash and positions; value missing bars at the last known close and increment `stale_days`.
2. Read only targets whose `execution_date == trade_date`; compute target quantities from opening NAV and that day's open.
3. Generate sell intents before buy intents, each sorted by symbol. Use `client_order_id = deterministic_id(run_id, signal_id, trade_date, symbol)`, `order_id = deterministic_id("order", client_order_id)`, and `fill_id = deterministic_id("fill", order_id, trade_date)`; ignore an already-seen client identifier.
4. Mark `NO_BAR` when the price or constraint row is absent, `SUSPENDED` when `is_tradable` is false, `LIMIT_UP` for a buy with `open >= limit_up`, and `LIMIT_DOWN` for a sell with `open <= limit_down`.
5. For a surviving intent, cap requested quantity by the trailing value carried on the target row—not the execution day's volume—then apply board lot rules. Use these exact calculations:

```python
adv_cap = int(float(target.adv20_volume) * config.volume_cap)
quantity = min(abs(target_quantity - current_quantity), adv_cap)
sellable_quantity = sum(
    lot.quantity
    for lot in lots
    if lot.symbol == target.symbol and lot.unlock_on <= trade_date
)
if side == "SELL":
    quantity = min(quantity, sellable_quantity)
quantity = legal_quantity(
    requested=quantity,
    min_buy_qty=int(constraint.min_buy_qty),
    qty_step=int(constraint.qty_step),
    side=side,
    position_quantity=current_quantity,
)
```

6. Mark a zero legal quantity as `T1_LOCKED`, `LOT_TOO_SMALL`, or `VOLUME_CAP` according to the first binding condition. For buys, reduce by one legal lot until notional plus all fees is within cash; mark `INSUFFICIENT_CASH` if no legal quantity remains.
7. Create at most one fill per order. For buys, debit notional plus fees and append one `AcquisitionLot` with `unlock_on` equal to the next calendar entry. For sells, credit notional less fees and consume oldest unlocked lots by `(acquired_on, symbol)` without changing locked lots. Calculate the minimum commission once on the aggregate order/day fill.
8. Append an order row for every intent with `signal_id`, `order_id`, `client_order_id`, `as_of`, `requested_quantity`, `filled_quantity`, `cancelled_quantity`, `status`, and `reason`. Append fill rows with the three identifiers plus `fill_id` and `as_of`. The allowed terminal statuses are `FILLED`, `PARTIAL`, `NO_BAR`, `SUSPENDED`, `LIMIT_UP`, `LIMIT_DOWN`, `T1_LOCKED`, `LOT_TOO_SMALL`, `VOLUME_CAP`, and `INSUFFICIENT_CASH`.
9. After all open fills, update `last_close` from that day's close. Emit one position row per held symbol with `as_of`, total/sellable/locked quantity, mark price, market value, and stale days. Emit NAV with `as_of`, `gross_nav` before costs, and `nav` after costs, then assert `cash >= 0` and `abs(cash + positions_value - nav) < 1e-8`.
10. Return `BacktestResult` with every table stably sorted and reset to a RangeIndex. Empty outputs must still contain their declared columns.

Implement `replay_fills` by sorting fills on `(trade_date, fill_id)`, dropping duplicate `client_order_id`, applying signed quantity and all fee columns, and returning holdings sorted by symbol. The cash transition is `cash -= notional + fees` for buys and `cash += notional - fees` for sells. Raise `ValueError` if any holding becomes negative. This definition makes replay of a duplicated fill table identical to replay of the unique table.

- [ ] **Step 4: Run all execution tests**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_execution.py -v
```

Expected: T+1 lots, limits, missing bars, lot sizes, historical volume cap, cash, fees, stale valuations, idempotent replay, and the NAV identity all pass.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/execution.py tests/test_execution.py tests/conftest.py
git commit -m "feat: simulate stateful A-share execution"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 9: Compute Metrics and Atomically Publish Auditable Artifacts

**Files:**
- Create: `src/barra_quant/reporting.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `ResearchResult`, `BacktestResult`, frozen config, source manifest, and core source paths.
- Produces: `compute_metrics`, `compute_run_id`, five fixed figures, `report.md`, `_SUCCESS`, and `publish_run`.

- [ ] **Step 1: Add failing identity and atomic-publish tests**

Create `tests/test_reporting.py`:

```python
import json
from pathlib import Path

import pandas as pd
import pytest

import barra_quant.reporting as reporting
from barra_quant.execution import BacktestResult
from barra_quant.research import ResearchResult
from barra_quant.reporting import compute_run_id, publish_run


def _minimal_results() -> tuple[ResearchResult, BacktestResult]:
    dates = pd.bdate_range("2026-03-02", periods=2)
    research = ResearchResult(
        panel=pd.DataFrame({"signal_date": [dates[0]], "symbol": ["sh600000"], "composite_score": [0.5]}),
        diagnostics={"ic": pd.DataFrame(), "decay": pd.DataFrame(), "quantiles": pd.DataFrame(), "style": pd.DataFrame()},
        benchmark=pd.DataFrame({"trade_date": dates, "benchmark_return": [0.0, 0.01]}),
    )
    backtest = BacktestResult(
        orders=pd.DataFrame({"client_order_id": ["order-1"], "status": ["FILLED"]}),
        fills=pd.DataFrame({
            "fill_id": ["fill-1"], "client_order_id": ["order-1"], "trade_date": [dates[0]],
            "side": ["BUY"], "notional": [1000.0], "commission": [5.0],
            "stamp_duty": [0.0], "transfer_fee": [0.01],
        }),
        positions=pd.DataFrame({"trade_date": [dates[0]], "symbol": ["sh600000"], "market_value": [1000.0]}),
        nav=pd.DataFrame({
            "trade_date": dates, "cash": [8995.0, 8995.0], "positions_value": [1000.0, 1010.0],
            "gross_nav": [10000.0, 10010.0], "nav": [9995.0, 10005.0],
        }),
    )
    return research, backtest


def test_run_id_is_content_addressed(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = {"mode": "SMOKE_TEST", "initial_cash": 10_000_000.0}
    manifest = {"source_commit": "abc123", "files": []}
    first = compute_run_id(config, manifest, [source])
    second = compute_run_id(config, manifest, [source])
    assert first == second
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_run_id(config, manifest, [source]) != first


def test_publish_writes_success_last(tmp_path):
    research_result, backtest_result = _minimal_results()
    destination = publish_run(
        tmp_path, "run-1", {"mode": "SMOKE_TEST"}, {"source_commit": "abc123"},
        research_result, backtest_result, {"mode": "SMOKE_TEST"},
    )
    assert (destination / "_SUCCESS").is_file()
    assert (destination / "research_panel.parquet").is_file()
    assert json.loads((destination / "metrics.json").read_text(encoding="utf-8"))["mode"] == "SMOKE_TEST"


def test_failed_publish_never_creates_success(tmp_path, monkeypatch):
    research_result, backtest_result = _minimal_results()
    def fail_write(frame, path):
        raise RuntimeError("write failed")
    monkeypatch.setattr(reporting, "_write_parquet", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        publish_run(tmp_path, "run-2", {}, {}, research_result, backtest_result, {})
    assert not (tmp_path / "run-2" / "_SUCCESS").exists()
```

Because PyArrow is intentionally absent, `publish_run` writes Parquet through an in-memory DuckDB connection registered with each DataFrame, not with `DataFrame.to_parquet`.

- [ ] **Step 2: Run reporting tests and observe the missing module**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_reporting.py -v
```

Expected: collection fails because `barra_quant.reporting` does not exist.

- [ ] **Step 3: Implement content hashing, metrics, figures, and atomic publish**

Create `src/barra_quant/reporting.py` with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import json
import os
import shutil

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .execution import BacktestResult
from .research import ResearchResult


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def compute_run_id(config: dict[str, object], source_manifest: dict[str, object], source_files: Sequence[Path]) -> str:
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
        connection.execute(f"COPY frame_to_write TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def compute_metrics(backtest: BacktestResult, benchmark: pd.DataFrame, trading_days: int = 252) -> dict[str, object]:
    nav = backtest.nav.sort_values("trade_date").copy()
    returns = nav["nav"].pct_change().dropna()
    gross_returns = nav["gross_nav"].pct_change().dropna()
    annual_return = (nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (trading_days / max(1, len(returns))) - 1.0
    annual_volatility = returns.std(ddof=1) * np.sqrt(trading_days)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else float("nan")
    drawdown = nav["nav"] / nav["nav"].cummax() - 1.0
    return {
        "mode": "SMOKE_TEST",
        "annualization_days": trading_days,
        "risk_free_rate": 0.0,
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "gross_total_return": float(nav["gross_nav"].iloc[-1] / nav["gross_nav"].iloc[0] - 1.0),
        "net_total_return": float(nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1.0),
        "total_cost": float(backtest.fills[["commission", "stamp_duty", "transfer_fee"]].sum().sum()),
        "benchmark_name": "pit_universe_equal_weight_gross",
    }
```

Create `_write_figures(staging: Path, research: ResearchResult, backtest: BacktestResult)` and always write these files under `figures/`: `ic_series.png`, `decay.png`, `quantiles.png`, `nav_drawdown.png`, and `style_exposure.png`. Map them respectively to time-series RankIC, horizon mean RankIC, quantile cumulative gross returns, strategy/benchmark NAV plus drawdown, and Barra-lite exposure. Use this exact unavailable-panel helper whenever its required table is empty:

```python
FIGURE_NAMES = (
    "ic_series.png",
    "decay.png",
    "quantiles.png",
    "nav_drawdown.png",
    "style_exposure.png",
)


def _unavailable_figure(path: Path, title: str, reason: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(0.5, 0.5, f"UNAVAILABLE\n{reason}", ha="center", va="center")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
```

Create `_render_report(metrics, source_manifest, research)` so `report.md` begins with `> SMOKE_TEST — not production evidence.` and then reports the preregistered factor hypothesis/rationale/direction/formulas/parameters/primary horizon/rejection criteria, source version, inferred calendar/constraints/availability, unadjusted-price warning, PIT incompleteness, missing corporate actions, order rejection counts, costs, gross/net results, benchmark distinction, and unavailable full-Barra styles. It must label the V1b 60/20/20 split and `RESEARCH_CANDIDATE` gate `NOT_EVALUATED_IN_V1A`. Assert the first line before writing the report.

Implement `publish_run` by writing every artifact to `<output_root>/.<run_id>.staging`, verifying expected files and DataFrame primary keys, writing `_SUCCESS` last, then using `os.replace(staging, output_root/run_id)`. If the final run already exists with `_SUCCESS`, verify its config and manifest hashes and return it without rewriting. Preserve a failed staging directory as `<output_root>/<run_id>.failed` with `error.json` and no `_SUCCESS`.

- [ ] **Step 4: Run reporting tests and inspect the report warning**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_reporting.py -v
```

Expected: content changes alter `run_id`, successful publication contains every required file and `_SUCCESS`, failed publication has no success marker, and `report.md` starts with the smoke-test warning.

- [ ] **Step 5: Create the task checkpoint**

```powershell
git add src/barra_quant/reporting.py tests/test_reporting.py
git commit -m "feat: publish deterministic V1a audit artifacts"
```

Expected: commit succeeds when Git exists; otherwise record the non-Git checkpoint.

---

### Task 10: Wire the Thin Runner and Prove Synthetic and Real Smoke Runs

**Files:**
- Create: `src/barra_quant/run.py`
- Create: `tests/test_v1a_integration.py`
- Modify: `README.md`
- Modify: `configs/v1a_smoke.json`

**Interfaces:**
- Consumes: all stable interfaces from Tasks 2-9 and `configs/v1a_smoke.json`.
- Produces: `run_v1a(config_path) -> Path`, CLI `python -m barra_quant.run --config ...`, one deterministic current-data run, and final V1a documentation.

- [ ] **Step 1: Add failing orchestration and smoke-status tests**

Create `tests/test_v1a_integration.py`:

```python
import json

from barra_quant.run import run_v1a


def test_synthetic_end_to_end_is_deterministic(synthetic_v1a_config):
    first = run_v1a(synthetic_v1a_config)
    second = run_v1a(synthetic_v1a_config)
    assert first == second
    assert (first / "_SUCCESS").is_file()
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "SMOKE_TEST"
    assert metrics["risk_free_rate"] == 0.0


def test_current_data_smoke_run_has_no_performance_assertion(project_smoke_config):
    output = run_v1a(project_smoke_config)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "SMOKE_TEST"
    assert metrics["source_commit"] == "08c4692f2b56d3ae47feae93de5576d4c626be8a"
    assert (output / "report.md").read_text(encoding="utf-8").startswith("> SMOKE_TEST")
```

Add these fixtures to `tests/conftest.py`; the synthetic fixture creates a temporary DuckDB `prices` table plus normalized manifest, while the project fixture skips when the commit-keyed data has not been materialized so unit CI remains network-independent:

```python
def _smoke_config(database: str, manifest: str) -> dict[str, object]:
    return {
        "database": database, "source_manifest": manifest, "output_root": "artifacts/runs",
        "mode": "SMOKE_TEST", "initial_cash": 10_000_000.0, "top_n": 50,
        "min_cross_section": 100, "min_history_days": 60, "liquidity_window": 20,
        "max_position_to_adv": 0.01, "min_avg_amount": 20_000_000.0,
        "horizons": [1, 5, 10, 20], "primary_horizon": 5,
        "winsor_lower": 0.01, "winsor_upper": 0.99, "beta_window": 120,
        "beta_min_observations": 100, "volume_cap": 0.05, "slippage_bps": 5.0,
        "slippage_stress_bps": [10.0, 20.0], "annualization_days": 252,
        "risk_free_rate": 0.0,
    }


@pytest.fixture
def synthetic_v1a_config(tmp_path: Path, synthetic_prices: pd.DataFrame) -> Path:
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
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    config = _smoke_config(
        database.relative_to(tmp_path).as_posix(),
        manifest_path.relative_to(tmp_path).as_posix(),
    )
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
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
```

- [ ] **Step 2: Run integration tests and observe the missing runner**

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_v1a_integration.py -v
```

Expected: collection fails because `barra_quant.run` does not exist.

- [ ] **Step 3: Implement the thin runner without business logic**

Create `src/barra_quant/run.py`:

```python
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
    bundle = load_data(root / raw_config["database"], root / raw_config["source_manifest"])
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
    run_id = compute_run_id(raw_config, bundle.source_manifest, CORE_SOURCE_FILES)
    signal_dates = make_signal_dates(bundle.calendar)
    base_universe = build_base_universe(bundle.prices, signal_dates, universe_config)
    research = build_research(
        bundle.prices, bundle.calendar, base_universe, universe_config,
        research_config, run_id,
    )
    constraints = infer_constraints(bundle.prices, bundle.calendar)
    targets = research.panel.loc[research.panel["target_weight"] > 0, [
        "signal_date", "symbol", "target_weight", "signal_id", "adv20_volume",
    ]].copy()
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
        bundle.prices, bundle.calendar, targets, constraints, fee_rules,
        ExecutionConfig(
            initial_cash=raw_config["initial_cash"],
            slippage_bps=raw_config["slippage_bps"],
            volume_cap=raw_config["volume_cap"],
        ),
        run_id,
    )
    metrics = compute_metrics(backtest, research.benchmark, raw_config["annualization_days"])
    metrics.update({
        "source_commit": bundle.source_manifest["source_commit"],
        "calendar_source": bundle.source_manifest["calendar_source"],
        "availability_source": "inferred_for_smoke_test",
        "constraints_source": "inferred",
        "universe_pit_complete": False,
        "corporate_actions_complete": False,
    })
    return publish_run(
        root / raw_config["output_root"], run_id, raw_config,
        bundle.source_manifest, research, backtest, metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V1a A-share smoke pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_v1a(args.config))


if __name__ == "__main__":
    main()
```

Keep `run.py` free of factor formulas, fill decisions, SQL, plotting, and file-format logic.

- [ ] **Step 4: Run focused, full, and real-data verification**

First migrate/materialize the current source snapshot and build its versioned database using the Task 2 CLI. This operation needs network permission only if the current checkout cannot supply all files. Then update the two paths in `configs/v1a_smoke.json` to the exact commit-keyed outputs.

Run:

```powershell
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests\test_v1a_integration.py -v
& 'E:\0KD\envs\tsffids\python.exe' -m pytest tests -v
& 'E:\0KD\envs\tsffids\python.exe' -m barra_quant.run --config configs\v1a_smoke.json
```

Expected:

- All tests pass.
- The real run completes with `_SUCCESS`.
- `metrics.json` says `SMOKE_TEST`, `universe_pit_complete=false`, and `corporate_actions_complete=false`.
- `report.md` begins with the smoke warning.
- Appending synthetic future dates leaves historical universe, factors, labels inputs, and targets unchanged.
- No assertion depends on positive performance.

- [ ] **Step 5: Update operating documentation and create the final checkpoint**

Add to `README.md`:

````markdown
## V1a strategy smoke run

V1a validates engineering behavior only. It does not establish that any factor or
portfolio is effective. The bundled source lacks complete point-in-time security
status and corporate actions.

```powershell
conda activate tsffids
python scripts/china_stock_pipeline.py all
python -m barra_quant.run --config configs/v1a_smoke.json
python -m pytest tests -v
```

Consume a run only when `artifacts/runs/<run_id>/_SUCCESS` exists. Inspect
`report.md`, `metrics.json`, `orders.parquet`, `fills.parquet`, `positions.parquet`,
and `nav.parquet` together; a positive return in this short sample is not research evidence.
````

Run the checkpoint:

```powershell
git add README.md configs/v1a_smoke.json src/barra_quant/run.py tests/test_v1a_integration.py tests/conftest.py
git commit -m "feat: complete deterministic V1a smoke pipeline"
```

Expected: commit succeeds when Git exists; otherwise record `Task 10 checkpoint: commit skipped because .git is absent` and report that version-control history remains unavailable.

---

## Final Verification Checklist

- [ ] `E:\0KD\envs\tsffids\python.exe -m pytest tests -v` passes.
- [ ] Existing raw-to-Parquet-to-DuckDB tests still pass.
- [ ] Current source commit is stored in an immutable directory with per-file SHA-256 values.
- [ ] Current real-data run finishes as `SMOKE_TEST` and creates `_SUCCESS`.
- [ ] Same config, source manifest, and source code produce the same `run_id` and byte-equivalent tabular values.
- [ ] Appending future synthetic observations does not alter historical universe, factors, or signals.
- [ ] T+1, limit-open rejection, no-bar rejection, board lot sizes, volume cap, fee dates, minimum commission, and NAV identity have exact synthetic assertions.
- [ ] T+1 open fill decisions do not read T+1 high, low, close, amount, or full-day volume.
- [ ] `company_snapshot_current` is absent from the historical strategy query path.
- [ ] Beta and residual volatility remain unavailable when the 120-day/100-observation rule is unmet.
- [ ] Report separates gross strategy, net strategy, and the gross research benchmark.
- [ ] Report labels size, nonlinear size, value, profitability, growth, leverage, and industry as `UNAVAILABLE`.
- [ ] Report lists unadjusted prices, incomplete PIT universe, inferred calendar/availability/constraints, missing corporate actions, and data-license limitation.
- [ ] No V1a output calls a factor valid, deployable, or investment-ready.

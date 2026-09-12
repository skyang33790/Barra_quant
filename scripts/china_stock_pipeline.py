from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb


REPO_URL = "https://github.com/guidebee/china-stock-data.git"
PRICE_COLUMNS = {
    "symbol": "VARCHAR",
    "trade_date": "DATE",
    "open": "DOUBLE",
    "close": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "volume": "BIGINT",
    "amount": "DOUBLE",
}


@dataclass(frozen=True)
class SourceVersion:
    commit: str
    committed_at: str


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _git(raw_repo: Path, *args: str, capture: bool = False) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={raw_repo.resolve().as_posix()}",
        "-C",
        str(raw_repo),
        *args,
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def download(raw_repo: Path) -> None:
    """Create or fast-forward the shallow raw-data checkout."""
    if (raw_repo / ".git").is_dir():
        _git(raw_repo, "pull", "--ff-only")
    elif raw_repo.exists():
        raise RuntimeError(f"Raw path exists but is not a Git checkout: {raw_repo}")
    else:
        raw_repo.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                str(raw_repo),
            ]
        )
    _git(raw_repo, "sparse-checkout", "set", "data")


def source_version(raw_repo: Path) -> SourceVersion:
    output = _git(raw_repo, "show", "-s", "--format=%H%n%cI", "HEAD", capture=True)
    commit, committed_at = output.splitlines()
    return SourceVersion(commit=commit, committed_at=committed_at)


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


def _create_raw_price_view(con: duckdb.DuckDBPyConnection, raw_repo: Path) -> None:
    price_glob = raw_repo / "data" / "price" / "*" / "*" / "*.csv"
    if not list((raw_repo / "data" / "price").glob("*/*/*.csv")):
        raise FileNotFoundError(f"No raw price CSV files found below {price_glob}")
    columns = "{" + ", ".join(
        f"'{name}': '{data_type}'" for name, data_type in PRICE_COLUMNS.items()
    ) + "}"
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW raw_prices AS
        SELECT
            lower(trim(symbol)) AS symbol,
            substr(lower(trim(symbol)), 3) AS code,
            upper(substr(trim(symbol), 1, 2)) AS exchange,
            trade_date,
            open,
            close,
            high,
            low,
            volume,
            amount,
            replace(filename, '\\', '/') AS source_file
        FROM read_csv(
            {_sql_string(price_glob)},
            header = false,
            columns = {columns},
            filename = true,
            strict_mode = true
        )
        """
    )


def _price_quality(con: duckdb.DuckDBPyConnection, relation: str) -> dict[str, object]:
    row = con.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count(DISTINCT trade_date) AS trading_days,
            count(DISTINCT symbol) AS symbols,
            min(trade_date) AS min_trade_date,
            max(trade_date) AS max_trade_date,
            count(*) - count(DISTINCT (symbol, trade_date)) AS duplicate_keys,
            count_if(
                symbol IS NULL OR trade_date IS NULL OR open IS NULL OR close IS NULL
                OR high IS NULL OR low IS NULL OR volume IS NULL OR amount IS NULL
            ) AS null_rows,
            count_if(NOT regexp_matches(symbol, '^(sh|sz|bj)[0-9]{{6}}$')) AS invalid_symbols,
            count_if(volume < 0 OR amount < 0) AS negative_activity,
            count_if(
                high < greatest(open, close, low)
                OR low > least(open, close, high)
            ) AS invalid_ohlc,
            count_if(
                regexp_extract(source_file, 'stock_price_([0-9]{{4}}_[0-9]{{2}}_[0-9]{{2}})\\.csv$', 1)
                <> strftime(trade_date, '%Y_%m_%d')
            ) AS filename_date_mismatches
        FROM {relation}
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    return dict(zip(names, row, strict=True))


def _assert_price_quality(metrics: dict[str, object]) -> None:
    failures = {
        name: metrics[name]
        for name in (
            "duplicate_keys",
            "null_rows",
            "invalid_symbols",
            "negative_activity",
            "invalid_ohlc",
            "filename_date_mismatches",
        )
        if metrics[name] != 0
    }
    if metrics["row_count"] == 0:
        failures["row_count"] = 0
    if failures:
        raise ValueError(f"Raw price quality checks failed: {failures}")


def normalize(
    raw_repo: Path,
    normalized_dir: Path,
    version: SourceVersion | None = None,
) -> dict[str, object]:
    """Rebuild deterministic normalized Parquet files from the raw snapshot."""
    raw_manifest_path = raw_repo / "manifest.json"
    raw_manifest: dict[str, object] = {}
    if raw_manifest_path.is_file():
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        snapshot_version = SourceVersion(
            commit=str(raw_manifest["source_commit"]),
            committed_at=str(raw_manifest["source_committed_at"]),
        )
        if version is not None and version != snapshot_version:
            raise ValueError("snapshot manifest version mismatch")
        version = snapshot_version
    version = version or source_version(raw_repo)
    company_json = raw_repo / "data" / "company" / "companies.json"
    if not company_json.is_file():
        raise FileNotFoundError(f"Missing raw company snapshot: {company_json}")

    staging = normalized_dir.with_name(normalized_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "prices").mkdir(parents=True)

    con = duckdb.connect()
    try:
        _create_raw_price_view(con, raw_repo)
        metrics = _price_quality(con, "raw_prices")
        _assert_price_quality(metrics)

        con.execute(
            f"""
            COPY (
                SELECT
                    symbol,
                    code,
                    exchange,
                    trade_date,
                    open,
                    close,
                    high,
                    low,
                    volume,
                    amount,
                    source_file,
                    {_sql_string(version.commit)} AS source_commit,
                    year(trade_date) AS trade_year,
                    month(trade_date) AS trade_month
                FROM raw_prices
                ORDER BY trade_date, symbol
            ) TO {_sql_string(staging / 'prices')}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (trade_year, trade_month))
            """
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    lower(trim(symbol)) AS symbol,
                    code::VARCHAR AS code,
                    upper(substr(trim(symbol), 1, 2)) AS exchange,
                    name::VARCHAR AS name,
                    stock_type::VARCHAR AS stock_type,
                    trade::DOUBLE AS last_trade,
                    mktcap::DOUBLE AS market_cap_cny_thousand,
                    nmc::DOUBLE AS float_market_cap_cny_thousand,
                    turnoverratio::DOUBLE AS turnover_ratio_pct,
                    {_sql_string(version.committed_at)}::TIMESTAMPTZ AS snapshot_at,
                    {_sql_string(version.commit)} AS source_commit,
                    false AS pit_safe
                FROM read_json_auto({_sql_string(company_json)})
                ORDER BY symbol
            ) TO {_sql_string(staging / 'company_snapshot_current.parquet')}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        company_metrics = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                count(*) - count(DISTINCT symbol) AS duplicate_symbols,
                count_if(NOT regexp_matches(symbol, '^(sh|sz|bj)[0-9]{{6}}$')) AS invalid_symbols
            FROM read_parquet({_sql_string(staging / 'company_snapshot_current.parquet')})
            """
        ).fetchone()
        if company_metrics[1] or company_metrics[2]:
            raise ValueError(
                "Company snapshot quality checks failed: "
                f"duplicate_symbols={company_metrics[1]}, invalid_symbols={company_metrics[2]}"
            )

        manifest = {
            **raw_manifest,
            "source_repo": REPO_URL,
            "source_commit": version.commit,
            "source_committed_at": version.committed_at,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "prices": {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in metrics.items()
            },
            "company_snapshot": {
                "rows": company_metrics[0],
                "pit_safe": False,
                "warning": "Current weekly snapshot; do not use it as a historical universe.",
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        con.close()

    if normalized_dir.exists():
        shutil.rmtree(normalized_dir)
    os.replace(staging, normalized_dir)
    return manifest


def load_duckdb(normalized_dir: Path, database: Path) -> dict[str, object]:
    """Load normalized Parquet into a compact DuckDB warehouse."""
    manifest_path = normalized_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Normalize first; missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    database.parent.mkdir(parents=True, exist_ok=True)
    staging = database.with_suffix(database.suffix + ".staging")
    if staging.exists():
        staging.unlink()

    con = duckdb.connect(str(staging))
    try:
        con.execute(
            f"""
            CREATE TABLE prices AS
            SELECT *
            FROM read_parquet(
                {_sql_string(normalized_dir / 'prices' / '**' / '*.parquet')},
                hive_partitioning = true
            )
            """
        )
        con.execute("CREATE UNIQUE INDEX prices_symbol_date ON prices(symbol, trade_date)")
        con.execute(
            f"""
            CREATE TABLE company_snapshot_current AS
            SELECT * FROM read_parquet({_sql_string(normalized_dir / 'company_snapshot_current.parquet')})
            """
        )
        con.execute(
            "CREATE UNIQUE INDEX company_snapshot_symbol "
            "ON company_snapshot_current(symbol)"
        )
        con.execute(
            """
            CREATE TABLE pipeline_metadata (
                source_repo VARCHAR,
                source_commit VARCHAR,
                source_committed_at TIMESTAMPTZ,
                built_at_utc TIMESTAMPTZ,
                price_rows BIGINT,
                min_trade_date DATE,
                max_trade_date DATE,
                company_rows BIGINT,
                company_snapshot_pit_safe BOOLEAN
            )
            """
        )
        con.execute(
            "INSERT INTO pipeline_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                manifest["source_repo"],
                manifest["source_commit"],
                manifest["source_committed_at"],
                manifest["built_at_utc"],
                manifest["prices"]["row_count"],
                manifest["prices"]["min_trade_date"],
                manifest["prices"]["max_trade_date"],
                manifest["company_snapshot"]["rows"],
                False,
            ],
        )
        db_metrics = _price_quality(con, "prices")
        _assert_price_quality(db_metrics)
        if db_metrics["row_count"] != manifest["prices"]["row_count"]:
            raise ValueError("DuckDB price row count does not match normalized Parquet")
        con.execute("CHECKPOINT")
    finally:
        con.close()

    if database.exists():
        database.unlink()
    os.replace(staging, database)
    return db_metrics


def project_paths(
    project_root: Path,
    source_commit: str,
) -> tuple[Path, Path, Path]:
    return (
        project_root / "data" / "raw" / "china-stock-data" / source_commit,
        project_root / "data" / "normalized" / "china-stock-data" / source_commit,
        project_root / "data" / "warehouse" / f"china_stock_{source_commit}.duckdb",
    )


def _checkout_path(project_root: Path) -> Path:
    checkout = project_root / "data" / "raw" / "_checkouts" / "china-stock-data"
    legacy = project_root / "data" / "raw" / "china-stock-data"
    if (legacy / ".git").is_dir():
        root = project_root.resolve()
        legacy_resolved = legacy.resolve()
        checkout_resolved = checkout.resolve()
        if not legacy_resolved.is_relative_to(root) or not checkout_resolved.is_relative_to(root):
            raise RuntimeError("checkout migration paths must stay under project root")
        if checkout.exists():
            raise RuntimeError(f"Checkout destination already exists: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(checkout))
    return checkout


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal china-stock-data pipeline")
    parser.add_argument(
        "command",
        choices=("download", "normalize", "load", "all"),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--source-commit",
        help="Exact source commit for normalize or load",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    source_commit = args.source_commit

    if args.command in {"download", "all"}:
        checkout = _checkout_path(project_root)
        download(checkout)
        version = source_version(checkout)
        raw_repo = materialize_snapshot(
            checkout / "data",
            project_root / "data" / "raw" / "china-stock-data",
            version,
        )
        source_commit = version.commit
        if args.command == "download":
            print((raw_repo / "manifest.json").read_text(encoding="utf-8"), end="")
    if source_commit is None:
        parser.error("--source-commit is required for normalize and load")
    raw_repo, normalized_dir, database = project_paths(project_root, source_commit)
    if args.command in {"normalize", "all"}:
        manifest = normalize(raw_repo, normalized_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.command in {"load", "all"}:
        metrics = load_duckdb(normalized_dir, database)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

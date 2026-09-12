from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import pytest

from scripts.china_stock_pipeline import (
    SourceVersion,
    load_duckdb,
    materialize_snapshot,
    normalize,
    project_paths,
    sha256_file,
)


class ChinaStockPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.raw = self.root / "raw"
        price_dir = self.raw / "data" / "price" / "2026" / "01"
        price_dir.mkdir(parents=True)
        (price_dir / "stock_price_2026_01_05.csv").write_text(
            "sh600000,2026-01-05,10.0,10.2,10.3,9.9,1000,10100.0\n"
            "sz000001,2026-01-05,12.0,11.8,12.1,11.7,2000,23800.0\n",
            encoding="utf-8",
        )
        company_dir = self.raw / "data" / "company"
        company_dir.mkdir(parents=True)
        (company_dir / "companies.json").write_text(
            json.dumps(
                [
                    {
                        "symbol": "sh600000",
                        "code": "600000",
                        "name": "浦发银行",
                        "stock_type": "sh_a",
                        "trade": 10.2,
                        "mktcap": 100.0,
                        "nmc": 90.0,
                        "turnoverratio": 1.0,
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.version = SourceVersion(
            commit="test-commit",
            committed_at="2026-01-05T15:30:00+08:00",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_normalize_and_load(self) -> None:
        normalized = self.root / "normalized"
        database = self.root / "warehouse" / "test.duckdb"

        manifest = normalize(self.raw, normalized, self.version)
        metrics = load_duckdb(normalized, database)

        self.assertEqual(manifest["prices"]["row_count"], 2)
        self.assertEqual(metrics["duplicate_keys"], 0)
        with duckdb.connect(str(database), read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM prices").fetchone()[0], 2)
            self.assertEqual(
                con.execute("SELECT pit_safe FROM company_snapshot_current").fetchone()[0],
                False,
            )

    def test_duplicate_symbol_date_is_rejected(self) -> None:
        price_file = next((self.raw / "data" / "price").glob("*/*/*.csv"))
        with price_file.open("a", encoding="utf-8") as handle:
            handle.write("sh600000,2026-01-05,10.0,10.2,10.3,9.9,1000,10100.0\n")

        with self.assertRaisesRegex(ValueError, "duplicate_keys"):
            normalize(self.raw, self.root / "normalized", self.version)


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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.china_stock_pipeline import SourceVersion, load_duckdb, normalize


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


if __name__ == "__main__":
    unittest.main()


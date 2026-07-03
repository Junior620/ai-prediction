"""Tests du parser CSV Investing.com pour le café robusta."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_coffee_csv import parse_investing_csv

ROBUSTA_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "cafe" / "ROBUSTA"
    / "London Robusta Coffee Futures Historical Data.csv"
)


class TestParseInvestingCsv:
    def test_parse_sample_csv(self, tmp_path):
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text(
            'Date,"Price","Open","High","Low","Vol.","Change %"\n'
            '07/02/2026,"3,745.00","3,700.00","3,760.00","3,690.00","9.03K","1.22%"\n'
            '07/01/2026,"3,700.00","3,680.00","3,710.00","3,650.00","8.10K","0.54%"\n'
            '07/01/2026,"3,699.00","3,680.00","3,710.00","3,650.00","8.10K","0.54%"\n',
            encoding="utf-8",
        )
        df = parse_investing_csv(str(csv_file))

        # Doublons de dates supprimés, tri croissant
        assert len(df) == 2
        assert list(df.columns) == ["date", "price"]
        assert df["date"].iloc[0] == pd.Timestamp("2026-07-01")
        assert df["date"].iloc[1] == pd.Timestamp("2026-07-02")
        # Séparateur de milliers correctement parsé
        assert df["price"].iloc[1] == 3745.0

    def test_parse_real_robusta_csv(self):
        if not ROBUSTA_CSV.exists():
            import pytest
            pytest.skip("CSV robusta absent")
        df = parse_investing_csv(str(ROBUSTA_CSV))
        assert len(df) > 4000
        assert df["date"].is_monotonic_increasing
        assert not df["date"].duplicated().any()
        # Prix robusta plausibles (USD/tonne)
        assert df["price"].between(500, 8000).all()

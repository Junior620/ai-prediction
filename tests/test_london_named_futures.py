"""Tests mois nommes Londres (Databento symbologie + labels ICE)."""

from src.data_collection.databento_london_collector import (
    contract_label,
    databento_raw_symbol,
    upcoming_named_contracts,
)
from src.data_collection.ice_london_collector import _normalize_contract_label


def test_databento_raw_symbol_format():
    assert databento_raw_symbol(12, 2026) == "C   FMZ0026!"
    assert databento_raw_symbol(3, 2027) == "C   FMH0027!"
    assert databento_raw_symbol(5, 2027) == "C   FMK0027!"


def test_contract_label():
    assert contract_label(12, 2026) == "DEC26"
    assert contract_label(3, 2027) == "MAR27"


def test_upcoming_named_contracts_count():
    specs = upcoming_named_contracts(count=7)
    assert len(specs) == 7
    assert all("FM" in s["symbol"] for s in specs)
    assert specs[0]["contract"]


def test_normalize_ice_contract_label():
    assert _normalize_contract_label("DEC26") == "DEC26"
    assert _normalize_contract_label("Dec 26") == "DEC26"
    assert _normalize_contract_label("MAR27") == "MAR27"
    assert _normalize_contract_label("Cash") is None

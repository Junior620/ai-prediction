"""Unit tests for futures curve helpers (no network)."""

from src.models.futures_curve_predictor import (
    FuturesCurvePredictor,
    investing_to_yahoo,
    yahoo_to_label,
)


def test_investing_to_yahoo():
    assert investing_to_yahoo("CCZ26") == "CCZ26.NYB"
    assert investing_to_yahoo("CCY00") is None
    assert investing_to_yahoo("CCZ26.NYB") == "CCZ26.NYB"


def test_yahoo_to_label():
    assert yahoo_to_label("CCZ26.NYB") == "Dec 26"
    assert yahoo_to_label("CCH27.NYB") == "Mar 27"


def test_parse_model_filename():
    assert FuturesCurvePredictor._parse_model_filename("xgb_CCZ26_NYB_h7") == ("CCZ26.NYB", 7)

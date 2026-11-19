"""Tableau de bord : pas de signal ni de valeur par défaut présentés comme une mesure."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "frontend" / "src" / "components" / "dashboard"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_signal_and_change_only_on_a_validated_model_price():
    card = _read(DASH / "PredictionHorizonCard.tsx")
    assert "const showSignal = status === 'validated' && !servedAsClose;" in card
    # Le badge BUY/SELL/HOLD et la flèche de variation sont tous deux sous showSignal.
    signal = card.index("Signal : {sigStyle.label}")
    assert card.rindex("showSignal ?", 0, signal) > card.index("const showSignal")
    arrow = card.index("{formatPercentage(pct)}")
    assert card.rindex("{showSignal && (", 0, arrow) > card.index("const showSignal")
    assert "Pas de signal — le modèle ne bat pas le cours inchangé" in card


def test_model_price_is_folded_away_from_the_decision_view():
    card = _read(DASH / "PredictionHorizonCard.tsx")
    model = card.index("{fmt(modelPrice)}")
    assert card.rindex("<details", 0, model) > card.index("const modelPrice")
    assert "À ne pas utiliser pour décider." in card


def test_no_default_volatility_or_momentum_tile():
    kpi = _read(DASH / "MarketKPIBar.tsx")
    assert "garchVol != null ? `${Number(garchVol).toFixed(1)}%` : '—'" not in kpi
    assert re.search(r"\.\.\.\(garchVol != null", kpi)
    analytics = _read(ROOT / "frontend" / "src" / "lib" / "marketAnalytics.ts")
    assert "changePct7d: number | null;" in analytics
    panel = _read(DASH / "AnalysisPanel.tsx")
    assert "pred7Validated" in panel


def test_freshness_banner_and_served_lot_are_shown():
    dashboard = _read(ROOT / "frontend" / "src" / "components" / "SimplifiedMarketDashboard.tsx")
    assert "marketInfo?.freshness?.banner" in dashboard
    assert "Lot servi : {marketInfo.model_version}" in dashboard


def test_model_derived_blocks_hidden_until_a_horizon_is_validated():
    # 19/11/2026 : le brief affichait « SELL » et une exposition vendeuse tirée des prix du modèle,
    # et les tableaux par contrat des projections J+1…J+30 non validées.
    dashboard = _read(ROOT / "frontend" / "src" / "components" / "SimplifiedMarketDashboard.tsx")
    assert "const modelValidated = (marketInfo?.validated_horizons?.length ?? 0) > 0;" in dashboard
    history = dashboard.index("<PredictionHistoryPanel")
    assert dashboard.rindex("{modelValidated && (", 0, history) > dashboard.index("const modelValidated")
    assert "showModelColumns={modelValidated}" in dashboard
    assert dashboard.count("modelValidated={modelValidated}") == 2

    brief = _read(ROOT / "frontend" / "src" / "components" / "MarketBrief.tsx")
    assert "modelValidated = false," in brief
    badge = brief.index("{brief.signal}</span>")
    assert brief.rindex("{modelValidated && (", 0, badge) > brief.index("function BriefBody")
    assert "{modelValidated && brief.recommendation && (" in brief
    assert "${support.toLocaleString()}" not in brief

    futures = _read(DASH / "FuturesCurvePanel.tsx")
    assert "showModelColumns = false," in futures
    for key in ("j1", "j7", "j14", "j30"):
        line = futures.index(f'dataKey="{key}"')
        assert futures.rindex("{showModel && (", 0, line) > futures.index("function CurveBlock")
    cell = futures.index("pickPred(c, 1)} currency")
    assert futures.rindex("{showModel && (", 0, cell) > futures.index("function CurveBlock")

    analytics = _read(ROOT / "frontend" / "src" / "lib" / "marketAnalytics.ts")
    assert "if (!opts.modelValidated) {" in analytics
    assert "toLocaleString('fr-FR')} $`" not in analytics

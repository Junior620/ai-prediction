"""Vérification bout en bout des predictors cacao + robusta (sans charger l'API FastAPI)."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

import pickle
import yaml
from config.settings import get_settings
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.market_registry import load_all_markets, resolve_api_market
from src.nlp.nlp_analyzer import NLPAnalyzer


def load_predictor_for_market(market_cfg, settings, nlp, pred_cfg, skip_nhits=False):
  model_dir = Path(market_cfg.models_dir)
  prophet_files = sorted(model_dir.glob("prophet_improved_*.pkl"))
  xgb_files = sorted(model_dir.glob("xgboost_improved_*.pkl"))
  if not prophet_files or not xgb_files:
    return None

  with open(prophet_files[-1], "rb") as f:
    prophet = pickle.load(f)
  with open(xgb_files[-1], "rb") as f:
    xgb = pickle.load(f)

  nhits = None
  if not skip_nhits:
    nhits_dirs = [d for d in sorted(model_dir.glob("nhits_*")) if d.is_dir()]
    if nhits_dirs:
      try:
        from neuralforecast import NeuralForecast
        nhits = NeuralForecast.load(path=str(nhits_dirs[-1]))
      except Exception as exc:
        print(f"  [WARN] N-HiTS non chargé pour {market_cfg.market_id}: {exc}")

  return ImprovedPricePredictor(
    prophet_model=prophet,
    xgboost_model=xgb,
    nlp_analyzer=nlp,
    sentiment_weight=pred_cfg.get("sentiment_weight", 0.05),
    model_version=prophet_files[-1].stem.replace("prophet_", ""),
    supabase_url=settings.supabase_url,
    supabase_key=settings.supabase_key,
    nhits_model=nhits,
    ensemble_weights_file=market_cfg.ensemble_weights_file,
    ensemble_fallback=pred_cfg.get("ensemble_fallback"),
    multi_step_mode=pred_cfg.get("multi_step_mode", "recursive"),
    conformal_intervals_file=market_cfg.conformal_intervals_file,
    confidence_level=pred_cfg.get("confidence_level", 0.90),
    price_bounds=market_cfg.price_bounds,
    price_table=market_cfg.price_table,
    nhits_unique_id=market_cfg.nhits_unique_id,
    garch_enabled=market_cfg.garch_enabled,
    models_dir=str(model_dir),
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--skip-nhits", action="store_true", help="Évite conflit DLL Windows FinBERT+N-HiTS")
  args = parser.parse_args()

  settings = get_settings()
  with open(ROOT / "config/config.yaml", encoding="utf-8") as f:
    pred_cfg = (yaml.safe_load(f) or {}).get("prediction", {})

  nlp = NLPAnalyzer()
  ok = True

  for market_id, cfg in load_all_markets().items():
    print(f"\n=== {cfg.display_name} ({market_id}) ===")
    predictor = load_predictor_for_market(cfg, settings, nlp, pred_cfg, skip_nhits=args.skip_nhits)
    if predictor is None:
      print("  ERREUR: modèles introuvables")
      ok = False
      continue

    preds = predictor.predict(horizons=[1, 7, 30], recent_news=[])
    p1 = preds[0]
    print(f"  API market: {cfg.api_markets[0]}")
    print(f"  J+1: ${p1.price:,.2f}  IC: ${p1.confidence_interval[0]:,.0f} - ${p1.confidence_interval[1]:,.0f}")
    if cfg.garch_enabled:
      vol = p1.components.get("garch_annualized_volatility")
      print(f"  GARCH vol ann.: {vol:.1f}%" if vol else "  GARCH: N/A")
    print(f"  N-HiTS actif: {p1.components.get('nhits') is not None}")

  assert resolve_api_market("COFFEE_ROBUSTA").market_id == "coffee_robusta"
  assert resolve_api_market("ICE_NY").market_id == "cocoa"
  print("\n[OK] Vérification bout en bout réussie")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())

"""
Pre-genere les briefs quotidiens (cron 1x/jour).
Usage: python scripts/generate_daily_briefs.py [--skip-nhits]
"""
import argparse
import pickle
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv()

from config.settings import get_settings
from src.api.cache import RedisCache
from src.intelligence.brief_service import BriefService
from src.models.improved_price_predictor import ImprovedPricePredictor
from src.models.market_registry import load_all_markets
from src.nlp.nlp_analyzer import NLPAnalyzer
from supabase import create_client


def load_predictor(market_cfg, settings, nlp, pred_cfg, skip_nhits=False):
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
                print(f"  [WARN] N-HiTS non charge: {exc}")

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
    parser = argparse.ArgumentParser(description="Genere les briefs Claude quotidiens")
    parser.add_argument("--skip-nhits", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    with open(ROOT / "config/config.yaml", encoding="utf-8") as f:
        pred_cfg = (yaml.safe_load(f) or {}).get("prediction", {})

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY manquante — arret.")
        sys.exit(1)

    redis = RedisCache()
    supabase = create_client(settings.supabase_url, settings.supabase_key)
    brief_service = BriefService(redis_cache=redis)
    nlp = NLPAnalyzer()

    ok = True
    for market_id, cfg in load_all_markets().items():
        api_market = cfg.api_markets[0]
        print(f"\n=== Brief {cfg.display_name} ({api_market}) ===")
        predictor = load_predictor(cfg, settings, nlp, pred_cfg, skip_nhits=args.skip_nhits)
        if predictor is None:
            print("  [SKIP] Modeles absents")
            ok = False
            continue
        try:
            result = brief_service.generate(
                api_market=api_market,
                predictor=predictor,
                supabase=supabase,
                user_id="cron",
                advanced=False,
                force_refresh=True,
            )
            sig = result["brief"].get("signal", "?")
            print(f"  OK signal={sig} cached={result.get('cached')}")
        except Exception as exc:
            print(f"  [ERR] {exc}")
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

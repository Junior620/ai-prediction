"""Calibrate ensemble + conformal for coffee_robusta from existing walk-forward CSV."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
wf_csv = str(ROOT / "reports/walk_forward/coffee_robusta/20260703_141553_walk_forward_predictions.csv")
ens_path = ROOT / "config/coffee_robusta/ensemble_weights.json"
conf_path = ROOT / "config/coffee_robusta/conformal_intervals.json"
ens_path.parent.mkdir(parents=True, exist_ok=True)

from src.validation.ensemble_calibrator import EnsembleCalibrator
from src.validation.conformal_interval_calibrator import (
    ConformalIntervalCalibrator,
    ConformalIntervalCalibratorConfig,
)

cal = EnsembleCalibrator()
res = cal.run(walk_forward_csv=wf_csv, nhits_csv=None, xgb_only=True)
cal.save(res, str(ens_path))
print(f"ENSEMBLE OK -> {ens_path}")

conf_cal = ConformalIntervalCalibrator(ConformalIntervalCalibratorConfig(coverage_level=0.90))
conf_res = conf_cal.run(walk_forward_csv=wf_csv, nhits_csv=None, ensemble_weights_file=str(ens_path))
conf_cal.save(conf_res, str(conf_path))
print(f"CONFORMAL OK -> {conf_path}")
for h, meta in conf_res.intervals_payload.get("by_horizon", {}).items():
    print(
        f"  h{h}: margin +/-${meta['margin_lower']:.0f}  "
        f"coverage={meta['empirical_coverage']:.1%}"
    )

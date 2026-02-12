"""Model drift observability using Evidently AI with PSI fallback.

Detects distribution shift between training baseline and current production data.
Uses Evidently DataDriftPreset when available; falls back to PSI (Population Stability Index)
when Evidently fails (e.g. Python 3.14 / Pydantic v1 incompatibility).
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .price_change_model import FEATURE_COLS

# Path to save/load reference (training) feature data for drift comparison
BASELINE_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "drift_reference.parquet"
# Dataset-level drift: share of columns that must drift to flag "drift detected" (default 50%)
DRIFT_SHARE_THRESHOLD = 0.5
# Per-column PSI threshold: PSI > 0.25 suggests significant drift for that column
PSI_DRIFT_THRESHOLD = 0.25


def compute_baseline(X: pd.DataFrame, output_path: Path | None = None) -> None:
    """Save reference (baseline) feature data for drift detection.
    Called from train.py after fitting; stored as parquet for later comparison.
    """
    output_path = output_path or BASELINE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in FEATURE_COLS if c in X.columns]
    X[cols].to_parquet(output_path, index=False)


def load_baseline(path: Path | None = None) -> pd.DataFrame | None:
    """Load reference dataframe from disk."""
    path = path or BASELINE_PATH
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _psi_score(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions.
    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct)).
    Higher PSI indicates greater distribution shift. Returns 0 on error.
    """
    try:
        min_val = min(float(np.nanmin(expected)), float(np.nanmin(actual)))
        max_val = max(float(np.nanmax(expected)), float(np.nanmax(actual)))
        if max_val == min_val:
            return 0.0
        edges = np.linspace(min_val, max_val, bins + 1)
        exp_hist, _ = np.histogram(np.nan_to_num(expected, 0), bins=edges)
        act_hist, _ = np.histogram(np.nan_to_num(actual, 0), bins=edges)
        exp_pct = (exp_hist + 1e-6) / (exp_hist.sum() + 1e-6 * len(exp_hist))
        act_pct = (act_hist + 1e-6) / (act_hist.sum() + 1e-6 * len(act_hist))
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    except Exception:
        return 0.0


def _compute_drift_fallback(
    ref_sub: pd.DataFrame,
    cur_sub: pd.DataFrame,
    drift_share_threshold: float,
) -> dict[str, Any]:
    """PSI-based drift when Evidently is unavailable (e.g. Python 3.14).
    Computes PSI per column; flags column as drifted if PSI > PSI_DRIFT_THRESHOLD.
    Dataset drift = share of drifted columns >= drift_share_threshold.
    """
    feature_drifts: dict[str, dict[str, Any]] = {}
    drifted_count = 0
    for col in ref_sub.columns:
        ref_vals = ref_sub[col].dropna().values
        cur_vals = cur_sub[col].dropna().values
        if len(ref_vals) < 2 or len(cur_vals) < 2:
            continue
        psi = _psi_score(ref_vals.astype(float), cur_vals.astype(float))
        drifted = psi > PSI_DRIFT_THRESHOLD
        if drifted:
            drifted_count += 1
        feature_drifts[col] = {
            "drifted": drifted,
            "drift_score": round(psi, 4),
            "method": "PSI",
            "threshold": PSI_DRIFT_THRESHOLD,
        }
    n_cols = len(feature_drifts)
    share = drifted_count / n_cols if n_cols else 0.0
    drift_detected = share >= drift_share_threshold
    summary = (
        f"Drift detected ({share:.1%} of columns drifted, threshold {drift_share_threshold:.0%})"
        if drift_detected
        else f"No significant drift ({share:.1%} of columns drifted)"
    )
    return {
        "drift_detected": drift_detected,
        "summary": summary,
        "share_of_drifting_columns": round(share, 4),
        "number_drifted_columns": drifted_count,
        "number_of_columns": n_cols,
        "drift_share_threshold": drift_share_threshold,
        "feature_drifts": feature_drifts,
        "n_current_samples": len(cur_sub),
        "n_reference_samples": len(ref_sub),
        "psi_score": share,
        "_fallback": True,
    }


def _parse_evidently_metrics(dump: dict) -> dict[str, Any]:
    """Extract drift metrics from Evidently Report dump_dict.
    Parses DriftedColumnsCount (overall share/count) and ValueDrift (per-column).
    """
    mr = dump.get("metric_results") or {}
    number_drifted = 0
    share_drifted = 0.0
    feature_drifts: dict[str, dict[str, Any]] = {}

    for _mid, m in mr.items() if isinstance(mr, dict) else []:
        params = (m.get("metric_value_location") or {}).get("metric", {}).get("params") or {}
        ptype = params.get("type", "")

        # Overall dataset drift: number and share of columns that drifted
        if "DriftedColumnsCount" in ptype:
            if "count" in m and isinstance(m["count"], dict) and "value" in m["count"]:
                number_drifted = int(float(m["count"]["value"]))
            if "share" in m and isinstance(m["share"], dict) and "value" in m["share"]:
                share_drifted = float(m["share"]["value"])
            continue

        # Per-column drift: K-S, chi-square, or other stat test result
        if "ValueDrift" in ptype:
            col = params.get("column")
            if not col:
                continue
            drifted = False
            drift_score: float | None = None
            method = params.get("method", "")
            threshold = params.get("threshold", 0.05)
            widgets = m.get("widget") or []
            for w in widgets:
                for c in (w.get("params") or {}).get("counters") or []:
                    label = str(c.get("label", ""))
                    val = str(c.get("value", ""))
                    if "Data drift detected" in label or "Drift in column" in val:
                        drifted = True
                    match = re.search(r"Drift score:\s*([\d.]+)", label)
                    if match:
                        drift_score = float(match.group(1))
            feature_drifts[col] = {
                "drifted": drifted,
                "drift_score": drift_score,
                "method": method,
                "threshold": threshold,
            }

    return {
        "number_drifted_columns": number_drifted,
        "share_of_drifting_columns": share_drifted,
        "feature_drifts": feature_drifts,
    }


def compute_drift(
    current: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> dict[str, Any]:
    """
    Run Evidently DataDriftPreset and return drift report.
    Dataset drift is detected when share_of_drifting_columns >= drift_share_threshold.
    """
    reference = reference or load_baseline()
    if reference is None or len(reference) == 0:
        return {
            "drift_detected": False,
            "summary": "No baseline available. Run train.py to create one.",
            "share_of_drifting_columns": None,
            "number_drifted_columns": None,
            "number_of_columns": None,
            "feature_drifts": {},
            "n_current_samples": len(current),
            "n_reference_samples": None,
        }

    # Align columns
    cols = [c for c in FEATURE_COLS if c in current.columns and c in reference.columns]
    if not cols:
        return {
            "drift_detected": False,
            "summary": "No overlapping features.",
            "share_of_drifting_columns": 0.0,
            "number_drifted_columns": 0,
            "number_of_columns": 0,
            "feature_drifts": {},
            "n_current_samples": len(current),
            "n_reference_samples": len(reference),
        }

    ref_sub = reference[cols].fillna(0)
    cur_sub = current[cols].fillna(0)

    try:
        # Evidently may fail on Python 3.14 (Pydantic v1 incompat) → fallback below
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
        snapshot = report.run(reference_data=ref_sub, current_data=cur_sub)
        dump = snapshot.dump_dict()
        parsed = _parse_evidently_metrics(dump)
        share = parsed["share_of_drifting_columns"]
        n_drifted = parsed["number_drifted_columns"]
        drift_detected = share >= drift_share_threshold
        summary = (
            f"Drift detected ({share:.1%} of columns drifted, threshold {drift_share_threshold:.0%})"
            if drift_detected
            else f"No significant drift ({share:.1%} of columns drifted)"
        )
        return {
            "drift_detected": drift_detected,
            "summary": summary,
            "share_of_drifting_columns": round(share, 4),
            "number_drifted_columns": n_drifted,
            "number_of_columns": len(cols),
            "drift_share_threshold": drift_share_threshold,
            "feature_drifts": parsed["feature_drifts"],
            "n_current_samples": len(cur_sub),
            "n_reference_samples": len(ref_sub),
            "psi_score": share,
        }
    except Exception as e:
        # Evidently fails on Python 3.14 (Pydantic v1 incompat) - use PSI fallback
        return _compute_drift_fallback(ref_sub, cur_sub, drift_share_threshold)

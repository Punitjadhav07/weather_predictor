"""
Fuse XGBoost (class + confidence) with fuzzy weather_decision for UI / DSS.

Uses confidence-aware weighting: high ML confidence follows ML; low confidence
leans on fuzzy; disagreements on wet/dry produce graded “possible …” outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from weather_dss.fuzzy_weather import coarse_bucket_from_ml_label


@dataclass
class HybridResult:
    ml_prediction: str
    ml_confidence: float
    fuzzy_score: float
    fuzzy_label: str
    final_decision: str
    fusion_mode: str
    recommendations: list[str]
    conf_low_threshold: float
    conf_high_threshold: float


def _is_wet_coarse(bucket: str) -> bool:
    return bucket in ("rain", "storm")


def fuse_hybrid(
    ml_label: str,
    ml_confidence: float,
    fuzzy_label: str,
    fuzzy_score: float,
    *,
    conf_low: float,
    conf_high: float,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
    precipitation_mm: float = 0.0,
    wind_speed_kph: float = 10.0,
    visibility_km: float = 10.0,
) -> HybridResult:
    """
    Combine ML and fuzzy outputs.

    ``conf_low`` / ``conf_high`` are typically validation-set percentiles (e.g. 25th / 75th)
    of max class probability — computed at model export, not hardcoded.

    - High ML confidence: trust ML for headline; adjust wording if fuzzy disagrees on wet/dry.
    - Low ML confidence: fuzzy-led headline.
    - Mid: explicit blend when wet flags disagree.
    """
    ml_coarse = coarse_bucket_from_ml_label(ml_label)
    wet_ml = _is_wet_coarse(ml_coarse)
    wet_fuzzy = fuzzy_label in ("rain", "storm")
    sunny_fuzzy = fuzzy_label == "sunny"

    mode = "blend"
    final = ""

    if ml_confidence >= conf_high:
        mode = "ml_high_confidence"
        if wet_ml and wet_fuzzy:
            final = (
                f"Wet conditions likely — ML: {ml_label!r} ({ml_confidence:.0%} confidence); "
                f"fuzzy agrees ({fuzzy_label})."
            )
        elif wet_ml and not wet_fuzzy:
            final = (
                f"Possible rain / wet spell — ML favors {ml_label!r} ({ml_confidence:.0%}), "
                f"but fuzzy suggests {fuzzy_label} from current T/RH/precip; treat as unsettled."
            )
            mode = "possible_rain"
        elif not wet_ml and wet_fuzzy and ml_confidence >= conf_high:
            final = (
                f"Watch for worsening weather — ML says {ml_label!r} ({ml_confidence:.0%}), "
                f"while fuzzy leans {fuzzy_label}; carry precautions."
            )
            mode = "ml_clear_fuzzy_wet"
        elif sunny_fuzzy and ml_coarse == "sunny":
            final = (
                f"Mostly fair — ML: {ml_label!r} ({ml_confidence:.0%}); fuzzy supports clear/dry ({fuzzy_label})."
            )
        else:
            final = f"{ml_label.title()} — ML confident ({ml_confidence:.0%}); fuzzy read: {fuzzy_label}."
    elif ml_confidence <= conf_low:
        mode = "fuzzy_led_low_ml"
        final = (
            f"Uncertain ML ({ml_confidence:.0%} on {ml_label!r}) — leaning on fuzzy: {fuzzy_label} "
            f"(score {fuzzy_score:.1f}/100)."
        )
    else:
        mode = "mid_confidence_blend"
        if wet_ml != wet_fuzzy:
            final = (
                f"Mixed signals — ML: {ml_label!r} ({ml_confidence:.0%}), fuzzy: {fuzzy_label} "
                f"(score {fuzzy_score:.1f}). Prefer cautious planning."
            )
        else:
            final = (
                f"{ml_label.title()} — ML ({ml_confidence:.0%}) and fuzzy ({fuzzy_label}) broadly agree."
            )

    recs = _dynamic_recommendations(
        mode=mode,
        ml_label=ml_label,
        ml_confidence=ml_confidence,
        fuzzy_label=fuzzy_label,
        fuzzy_score=fuzzy_score,
        wet_ml=wet_ml,
        wet_fuzzy=wet_fuzzy,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        precipitation_mm=precipitation_mm,
        wind_speed_kph=wind_speed_kph,
        visibility_km=visibility_km,
        conf_low=conf_low,
        conf_high=conf_high,
    )

    return HybridResult(
        ml_prediction=ml_label,
        ml_confidence=ml_confidence,
        fuzzy_score=fuzzy_score,
        fuzzy_label=fuzzy_label,
        final_decision=final,
        fusion_mode=mode,
        recommendations=recs,
        conf_low_threshold=conf_low,
        conf_high_threshold=conf_high,
    )


def _dynamic_recommendations(
    *,
    mode: str,
    ml_label: str,
    ml_confidence: float,
    fuzzy_label: str,
    fuzzy_score: float,
    wet_ml: bool,
    wet_fuzzy: bool,
    temperature_c: float,
    humidity_pct: float,
    precipitation_mm: float,
    wind_speed_kph: float,
    visibility_km: float,
    conf_low: float,
    conf_high: float,
) -> list[str]:
    out: list[str] = []

    if wet_ml:
        if ml_confidence >= conf_high:
            out.append(
                "Strong ML signal for wet-type conditions — prioritize waterproofs and check official alerts."
            )
        elif ml_confidence <= conf_low:
            out.append(
                "Wet-type ML class is uncertain (low confidence); treat rain readiness as precautionary only."
            )
        else:
            out.append(
                "Moderate ML confidence on wet-type weather — reasonable to pack light rain protection."
            )

    if mode == "possible_rain" or (
        wet_ml and conf_low < ml_confidence < conf_high and not wet_fuzzy
    ):
        out.append(
            "Precautionary: compact umbrella or shell; verify sky/radar if planning long outdoors."
        )

    if wet_fuzzy or precipitation_mm > 1.0:
        out.append("Carry rain gear; roads may be slick if intensity picks up.")

    if ml_confidence <= conf_low:
        out.append(
            "Overall low ML confidence on the top class; lean on fuzzy output and live conditions."
        )

    if temperature_c >= 32:
        out.append("High heat: limit strenuous outdoor activity; hydrate and seek shade.")
    elif temperature_c <= 2:
        out.append("Cold: dress in layers; watch for ice if moisture is present.")

    if wind_speed_kph >= 40:
        out.append(
            "High wind — travel caution: secure loose items; allow extra stopping distance when driving."
        )
    elif wind_speed_kph >= 25:
        out.append("Breezy conditions: stay aware of gusts if cycling or towing.")

    if visibility_km < 3:
        out.append(
            "Low visibility safety alert: slow down, increase following distance, use lights as appropriate."
        )

    if not out:
        out.append("Conditions appear moderate; routine precautions are enough.")

    return out

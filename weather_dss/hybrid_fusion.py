"""
Hybrid fusion (ML + fuzzy) for decision support.

ML uses XGBoost multiclass to predict 27 fine-grained weather labels.
Fuzzy logic provides an interpretable 4-class wetness decision (sunny/cloudy/rain/storm)
using quantile-derived LOW/MEDIUM/HIGH membership functions.

Hybrid fusion combines both with validation-calibrated confidence bands so the
system can explicitly handle uncertainty and conflicting wet/dry signals.

Coarse-category mapping
-----------------------
Both ML and fuzzy outputs are projected onto four coarse categories so they
can be compared on equal footing:

    storm-like  — thunder, storm, hurricane, tornado, tropical
    rain-like   — rain, drizzle, shower; also snow/ice/sleet/pellets
                  (mapped here because they produce wet surfaces and share
                   umbrella / caution recommendations with rain)
    dry         — sunny, clear
    neutral     — cloudy, overcast, mist, fog, haze, "other", fallback

Fusion logic
------------
1. If coarse categories agree → trust ML (highest granularity).
2. Else if ML confidence is HIGH → trust ML.
3. Else if ML confidence is LOW  → fall back to fuzzy.
4. Else → directional uncertainty based on ML rain probability.

Directional uncertainty
-----------------------
When the system cannot confidently choose ML or fuzzy, it reports a
*directional* uncertain verdict rather than a generic "uncertain":
    ml_rain_probability ≥ 0.5 → "Possible Rain ⚠️"
    ml_rain_probability <  0.5 → "Possibly Clear 🌤️"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """All outputs produced by a single hybrid-fusion call."""

    ml_prediction: str
    ml_confidence: float
    fuzzy_score: float
    fuzzy_label: str
    ml_rain_probability: float
    disagreement: float
    ml_coarse: str
    fuzzy_coarse: str
    confidence_level: str
    final_source: str
    final_decision: str
    fusion_mode: str
    reasoning: list[str]
    recommendations: list[str]
    conf_low_threshold: float
    conf_high_threshold: float
    disagreement_high_threshold: float


# ---------------------------------------------------------------------------
# Coarse-category mapping (deterministic, no overlap)
# ---------------------------------------------------------------------------

def ml_label_to_coarse_category(ml_label: str) -> str:
    """Map any of the 27 fine-grained ML classes to one of 4 coarse categories.

    Priority order (first match wins):
        1. storm-like  — thunder, storm, hurricane, tornado, tropical
        2. rain-like   — rain, drizzle, shower, sleet, snow, pellets, ice, freezing
                         (snow/ice are "cold-wet": they share wet-surface safety
                          recommendations with rain, so a 5th category is unnecessary)
        3. dry         — sun, clear
        4. neutral     — cloud, overcast, mist, fog, haze, other, everything else

    Returns one of: "storm-like", "rain-like", "dry", "neutral".
    """
    s = str(ml_label).lower().strip()

    # "other" is the merged rare-class bucket → neutral
    if s == "other":
        return "neutral"

    # --- 1. Storm-like (checked first: "thunderstorm" contains "storm") ---
    if any(k in s for k in ("thunder", "storm", "hurricane", "tornado", "tropical")):
        return "storm-like"

    # --- 2. Rain-like (includes cold-wet: snow, ice, sleet, pellets, freezing) ---
    if any(
        k in s
        for k in (
            "rain",
            "drizzle",
            "shower",
            "sleet",
            "snow",
            "pellets",
            "ice",
            "freezing",
        )
    ):
        return "rain-like"

    # --- 3. Dry ---
    if any(k in s for k in ("sun", "clear")):
        return "dry"

    # --- 4. Neutral (cloudy, fog, mist, haze, and anything unrecognized) ---
    if any(k in s for k in ("cloud", "overcast", "mist", "fog", "haze")):
        return "neutral"

    # Fallback: treat unknown labels as neutral rather than crashing
    return "neutral"


def fuzzy_label_to_coarse_category(fuzzy_label: str) -> str:
    """Map the 4 fuzzy output labels to the same coarse categories used by ML.

    sunny  → dry
    cloudy → neutral
    rain   → rain-like
    storm  → storm-like
    """
    mapping = {
        "sunny": "dry",
        "cloudy": "neutral",
        "rain": "rain-like",
        "storm": "storm-like",
    }
    return mapping.get(fuzzy_label, "neutral")


# ---------------------------------------------------------------------------
# Decision text helpers
# ---------------------------------------------------------------------------

def _decision_text_from_coarse(coarse: str) -> str:
    """Convert a resolved coarse category to a user-friendly decision string."""
    if coarse == "dry":
        return "Sunny ☀️"
    if coarse == "neutral":
        return "Cloudy"
    if coarse == "rain-like":
        return "Possible Rain ⚠️"
    if coarse == "storm-like":
        return "Possible Storm ⚠️"
    # Should not reach here after directional-uncertainty logic,
    # but kept as a safety fallback.
    return "Uncertain / Transitional ⚠️"


def _confidence_level(
    ml_confidence: float, *, conf_low: float, conf_high: float
) -> str:
    """Classify ML max-probability into Low / Medium / High using
    validation-calibrated percentile bands (p25 = conf_low, p75 = conf_high).
    """
    if ml_confidence <= conf_low:
        return "Low"
    if ml_confidence >= conf_high:
        return "High"
    return "Medium"


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------

def fuse_hybrid(
    ml_label: str,
    ml_confidence: float,
    ml_proba: np.ndarray,
    label_encoder: LabelEncoder,
    fuzzy_label: str,
    fuzzy_score: float,
    *,
    conf_low: float,
    conf_high: float,
    disagreement_high_threshold: float,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
    precipitation_mm: float = 0.0,
    wind_speed_kph: float = 10.0,
    visibility_km: float = 10.0,
) -> HybridResult:
    """Explicit hybrid fusion with validation-calibrated confidence bands.

    Steps:
        1. Project both ML and fuzzy outputs to coarse categories.
        2. Compute disagreement = |ML(rain-prob) − fuzzy_wet_norm|.
        3. Select final source (ML / Fuzzy / Uncertain) via confidence bands.
        4. If uncertain, apply directional uncertainty based on ML rain prob.
        5. Generate reasoning (2–3 bullets) and priority-ordered recommendations.

    Args:
        ml_label: Fine-grained ML predicted class (e.g. "light rain").
        ml_confidence: Max predicted class probability.
        ml_proba: Full probability vector from predict_proba.
        label_encoder: Fitted LabelEncoder (maps indices ↔ class names).
        fuzzy_label: Fuzzy output label (sunny/cloudy/rain/storm).
        fuzzy_score: Fuzzy crisp wet score (0–100).
        conf_low: Validation p25 of max-probability (Low threshold).
        conf_high: Validation p75 of max-probability (High threshold).
        disagreement_high_threshold: Validation p75 of |ML−fuzzy| disagreement.
        temperature_c: Input temperature in Celsius.
        humidity_pct: Input humidity percentage.
        precipitation_mm: Input precipitation in mm.
        wind_speed_kph: Input wind speed in kph.
        visibility_km: Input visibility in km.
    """
    # --- Coarse projection ---
    ml_coarse = ml_label_to_coarse_category(ml_label)
    fuzzy_coarse = fuzzy_label_to_coarse_category(fuzzy_label)

    # --- Confidence band classification ---
    confidence_level = _confidence_level(
        ml_confidence, conf_low=conf_low, conf_high=conf_high
    )

    # --- ML rain-like probability ---
    # Sum probabilities of all rain-like + storm-like classes to get
    # an aggregate "wet" probability from the ML side.
    ml_classes = list(label_encoder.classes_)
    rain_mask = np.array(
        [ml_label_to_coarse_category(cls) in ("rain-like", "storm-like") for cls in ml_classes],
        dtype=bool,
    )
    ml_rain_probability = float(np.asarray(ml_proba)[rain_mask].sum())

    # --- Disagreement metric ---
    # Fuzzy wet score is 0..100; normalize to 0..1 for direct comparison
    # with the ML probability.  High disagreement ≥ p75 threshold.
    fuzzy_wet_norm = float(fuzzy_score) / 100.0
    disagreement = abs(ml_rain_probability - fuzzy_wet_norm)

    # --- Fusion decision ---
    # Priority:
    #   1. Coarse match → ML (unless medium-conf + high disagreement)
    #   2. High confidence → ML
    #   3. Low confidence → Fuzzy
    #   4. Otherwise → Uncertain (directional)
    coarse_match = ml_coarse == fuzzy_coarse

    final_source = "Uncertain"
    if coarse_match:
        final_source = "ML"
        # Even when categories match, high disagreement with only medium
        # confidence suggests unreliable intensity estimates.
        if confidence_level == "Medium" and disagreement >= disagreement_high_threshold:
            final_source = "Uncertain"
    elif ml_confidence >= conf_high:
        final_source = "ML"
    elif ml_confidence <= conf_low:
        final_source = "Fuzzy"
    else:
        final_source = "Uncertain"

    # Never surface ML's "other" (rare-class bucket) as a decision.
    if final_source == "ML" and ml_label == "other":
        final_source = "Uncertain"

    # --- Final decision text ---
    if final_source == "ML":
        final_coarse = ml_coarse
        final_decision = _decision_text_from_coarse(final_coarse)
    elif final_source == "Fuzzy":
        final_coarse = fuzzy_coarse
        final_decision = _decision_text_from_coarse(final_coarse)
    else:
        # Directional uncertainty: give the user a hint about the likely
        # direction rather than a generic "uncertain" label.
        final_coarse = "uncertain"
        if ml_rain_probability >= 0.5:
            final_decision = "Possible Rain ⚠️"
        else:
            final_decision = "Possibly Clear 🌤️"

    # --- Reasoning (2–3 concise, non-overlapping bullets) ---
    reasoning = _build_reasoning(
        ml_label=ml_label,
        ml_coarse=ml_coarse,
        fuzzy_label=fuzzy_label,
        fuzzy_coarse=fuzzy_coarse,
        confidence_level=confidence_level,
        ml_rain_probability=ml_rain_probability,
        fuzzy_wet_norm=fuzzy_wet_norm,
        disagreement=disagreement,
        disagreement_high_threshold=disagreement_high_threshold,
        coarse_match=coarse_match,
        final_source=final_source,
    )

    # --- Recommendations (priority-ordered) ---
    recommendations = _build_recommendations(
        final_coarse=final_coarse,
        confidence_level=confidence_level,
        disagreement=disagreement,
        disagreement_high_threshold=disagreement_high_threshold,
        fuzzy_label=fuzzy_label,
        fuzzy_wet_norm=fuzzy_wet_norm,
        precipitation_mm=precipitation_mm,
        wind_speed_kph=wind_speed_kph,
        visibility_km=visibility_km,
        temperature_c=temperature_c,
    )

    return HybridResult(
        ml_prediction=ml_label,
        ml_confidence=ml_confidence,
        fuzzy_score=fuzzy_score,
        fuzzy_label=fuzzy_label,
        ml_rain_probability=ml_rain_probability,
        disagreement=disagreement,
        ml_coarse=ml_coarse,
        fuzzy_coarse=fuzzy_coarse,
        confidence_level=confidence_level,
        final_source=final_source,
        final_decision=final_decision,
        fusion_mode=final_source,
        reasoning=reasoning,
        recommendations=recommendations,
        conf_low_threshold=conf_low,
        conf_high_threshold=conf_high,
        disagreement_high_threshold=disagreement_high_threshold,
    )


# ---------------------------------------------------------------------------
# Reasoning builder (Task 4)
# ---------------------------------------------------------------------------

def _build_reasoning(
    *,
    ml_label: str,
    ml_coarse: str,
    fuzzy_label: str,
    fuzzy_coarse: str,
    confidence_level: str,
    ml_rain_probability: float,
    fuzzy_wet_norm: float,
    disagreement: float,
    disagreement_high_threshold: float,
    coarse_match: bool,
    final_source: str,
) -> list[str]:
    """Generate 2–3 concise reasoning bullets explaining the fusion decision.

    Cases:
        - Agreement → "ML and fuzzy agree"
        - High disagreement → "High disagreement detected"
        - Low confidence → "Low ML confidence — deferring to fuzzy"
        - Medium uncertainty → "Moderate uncertainty handled via confidence bands"
    """
    bullets: list[str] = []

    # Bullet 1: describe the agreement / disagreement status
    if coarse_match and disagreement < disagreement_high_threshold:
        pattern = ml_coarse.replace("-like", "").replace("-", " ")
        bullets.append(
            f"ML and fuzzy agree: both indicate **{pattern}** conditions."
        )
    elif disagreement >= disagreement_high_threshold:
        bullets.append(
            f"High disagreement detected: ML rain-like probability "
            f"({ml_rain_probability:.0%}) vs fuzzy wet score ({fuzzy_wet_norm:.0%})."
        )
    else:
        bullets.append(
            "Moderate uncertainty: confidence bands used to select "
            "between ML and fuzzy signals."
        )

    # Bullet 2: explain what was chosen and why
    if final_source == "ML":
        if ml_label == "other":
            bullets.append(
                f"ML confidence is **{confidence_level}** "
                "(rare-class prediction merged into transitional bucket)."
            )
        else:
            bullets.append(
                f"ML prediction (**{ml_label}**) selected with "
                f"**{confidence_level}** confidence."
            )
    elif final_source == "Fuzzy":
        bullets.append(
            f"Low ML confidence — deferring to fuzzy logic (**{fuzzy_label}**)."
        )
    else:
        # Uncertain / directional
        if ml_rain_probability >= 0.5:
            bullets.append(
                "Leaning toward rain based on ML probability distribution."
            )
        else:
            bullets.append(
                "Leaning toward clear/dry based on ML probability distribution."
            )

    # Return exactly 2–3 bullets (no more)
    return bullets[:3]


# ---------------------------------------------------------------------------
# Recommendation builder (Task 5 — priority tiers)
# ---------------------------------------------------------------------------

def _build_recommendations(
    *,
    final_coarse: str,
    confidence_level: str,
    disagreement: float,
    disagreement_high_threshold: float,
    fuzzy_label: str,
    fuzzy_wet_norm: float,
    precipitation_mm: float,
    wind_speed_kph: float,
    visibility_km: float,
    temperature_c: float,
) -> list[str]:
    """Build priority-ordered recommendations capped at 7 items.

    Priority tiers:
        1. Safety   — storm, high wind (≥40 kph), low visibility (<3 km)
        2. Weather  — umbrella, rain gear, precipitation awareness
        3. Comfort  — heat (≥32 °C), cold (≤2 °C)

    Each tier is appended in order so safety items always appear first.
    """
    out: list[str] = []

    wetish = final_coarse in ("rain-like", "storm-like") or fuzzy_label in ("rain", "storm")

    # ── Tier 1: Safety ────────────────────────────────────────────────
    if final_coarse == "storm-like":
        out.append(
            "⛈️ Possible storm: avoid travel where possible; secure loose items."
        )
    if wind_speed_kph >= 40:
        out.append(
            "💨 High wind (≥40 kph): travel caution; secure loose items."
        )
    if visibility_km < 3:
        out.append(
            "🌫️ Low visibility (<3 km): slow down and increase following distance."
        )

    # ── Tier 2: Weather preparation ───────────────────────────────────
    if wetish and final_coarse != "storm-like":
        # Storm already covered in safety tier
        out.append("🌧️ Carry an umbrella / rain gear; roads may get slick.")
    if wetish and (precipitation_mm > 1.0 or fuzzy_wet_norm >= 0.6):
        out.append("Higher precipitation indicated: plan extra travel time.")
    if not wetish:
        if final_coarse == "dry":
            out.append(
                "☀️ Outdoor activities likely OK; stay hydrated and use sun protection."
            )
        else:
            out.append(
                "🌥️ Cloudy/transitional: dress in light layers and keep plans flexible."
            )

    if confidence_level == "Low":
        out.append("⚠️ Low confidence: check local forecast before important decisions.")
    if disagreement >= disagreement_high_threshold:
        out.append(
            "⚠️ Conflicting wet/dry signals: carry an umbrella as a precaution."
        )

    # ── Tier 3: Comfort ───────────────────────────────────────────────
    if temperature_c >= 32:
        out.append("🔥 High heat (≥32 °C): limit strenuous activity; hydrate.")
    elif temperature_c <= 2:
        out.append("🥶 Cold conditions (≤2 °C): dress in warm layers.")

    # Breezy note (below safety threshold but still notable)
    if 25 <= wind_speed_kph < 40:
        out.append("🍃 Breezy: expect gusts; be careful while driving or cycling.")

    # Cap at 7 recommendations to avoid overwhelming the user
    return out[:7]

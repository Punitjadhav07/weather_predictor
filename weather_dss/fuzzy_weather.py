"""
Phase 3 — Fuzzy weather decision using scikit-fuzzy (fuzzification, rules, defuzzification).

Universe bounds for antecedents are derived from dataset percentiles (no hardcoded outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class FuzzyWeatherSystem:
    """Holds the control system and variable references for inference + interpretation."""

    control: ctrl.ControlSystem
    temperature: ctrl.Antecedent
    humidity: ctrl.Antecedent
    precipitation: ctrl.Antecedent
    wind_speed: ctrl.Antecedent
    weather_decision: ctrl.Consequent


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _fallback_crisp_score(
    temperature: float,
    humidity: float,
    precipitation: float,
    wind_speed: float = 0.0,
) -> float:
    """Deterministic backup if rule aggregation yields no defuzzified output."""
    s = 48.0
    if precipitation > 4:
        s += 28
    elif precipitation > 0.5:
        s += 14
    elif precipitation > 0.05:
        s += 6
    if humidity > 82:
        s += 12
    elif humidity < 32:
        s -= 12
    if temperature > 29:
        s -= 10
    if temperature < 4:
        s += 6
    if wind_speed > 45 and precipitation > 1.0:
        s += 12
    elif wind_speed > 55:
        s += 8
    return float(np.clip(s, 0.0, 100.0))


def build_fuzzy_system_from_dataframe(df: pd.DataFrame) -> FuzzyWeatherSystem:
    """
    Build antecedents (temp, humidity, precip, wind_speed) with LOW/MEDIUM/HIGH MFs
    from column quantiles; consequent weather_decision on 0–100 (sunny/cloudy/rain/storm).
    """
    t = df["temperature"].astype(float)
    h = df["humidity"].astype(float)
    p = df["precipitation"].astype(float).clip(lower=0)
    wnd = df["wind_speed"].astype(float).clip(lower=0)

    t_lo = float(t.quantile(0.02))
    t_hi = float(t.quantile(0.98))
    t_q25, t_q50, t_q75 = (float(t.quantile(q)) for q in (0.25, 0.5, 0.75))
    t_pad = max(2.0, (t_hi - t_lo) * 0.08)
    t_min_u = min(t_lo - t_pad, t_q25 - 5)
    t_max_u = max(t_hi + t_pad, t_q75 + 5)
    temp_universe = np.arange(t_min_u, t_max_u + 0.1, 0.2)

    humidity_universe = np.arange(0, 101, 1)
    h_med = float(h.quantile(0.5))
    p_max_u = float(max(20.0, p.quantile(0.995) * 1.1, float(p.max()) if len(p) else 20.0))
    precip_universe = np.linspace(0.0, p_max_u, 200)
    # Percentile anchors for precip (dataset has many zeros; use positive subset for mid/high)
    p_pos = p[p > 0.01]
    if len(p_pos) > 200:
        e1 = float(np.clip(p_pos.quantile(0.35), 0.05, p_max_u * 0.25))
        e2 = float(np.clip(p_pos.quantile(0.65), e1 + 0.05, p_max_u * 0.55))
        e3 = float(np.clip(p_pos.quantile(0.88), e2 + 0.05, p_max_u * 0.92))
    else:
        e1, e2, e3 = 0.2, min(2.0, p_max_u * 0.2), min(8.0, p_max_u * 0.6)
    e1, e2, e3 = sorted([e1, e2, e3])

    w_lo = float(wnd.quantile(0.02))
    w_hi = float(wnd.quantile(0.98))
    w25, w50, w75 = (float(wnd.quantile(q)) for q in (0.25, 0.5, 0.75))
    w_pad = max(1.0, (w_hi - w_lo) * 0.1)
    w_min_u = 0.0
    w_max_u = float(min(max(w_hi + w_pad, w75 + 8.0, 25.0), 130.0))
    wind_universe = np.linspace(w_min_u, w_max_u, 140)
    w_u_top = float(wind_universe[-1])

    temperature = ctrl.Antecedent(temp_universe, "temperature")
    temperature["low"] = fuzz.trimf(temperature.universe, [t_min_u, t_min_u, t_q50])
    temperature["medium"] = fuzz.trimf(temperature.universe, [t_q25, t_q50, t_q75])
    temperature["high"] = fuzz.trimf(temperature.universe, [t_q50, t_max_u, t_max_u])

    humidity = ctrl.Antecedent(humidity_universe, "humidity")
    humidity["low"] = fuzz.trimf(humidity.universe, [0, 0, min(48, h_med)])
    humidity["medium"] = fuzz.trimf(
        humidity.universe, [max(30, h_med - 20), h_med, min(85, h_med + 22)]
    )
    humidity["high"] = fuzz.trimf(humidity.universe, [max(60, h_med + 5), 100, 100])

    precipitation = ctrl.Antecedent(precip_universe, "precipitation")
    precipitation["low"] = fuzz.trapmf(
        precipitation.universe, [0.0, 0.0, max(0.01, e1 * 0.35), e1]
    )
    precipitation["medium"] = fuzz.trimf(
        precipitation.universe, [max(0.01, e1 * 0.55), e2, min(e3, p_max_u * 0.85)]
    )
    precipitation["high"] = fuzz.trapmf(
        precipitation.universe,
        [max(e2, e3 * 0.45), e3, p_max_u, p_max_u],
    )

    wind_speed = ctrl.Antecedent(wind_universe, "wind_speed")
    wind_speed["low"] = fuzz.trimf(wind_universe, [w_min_u, w_min_u, max(w50, 4.0)])
    wind_speed["medium"] = fuzz.trimf(
        wind_universe,
        [max(1.0, w25 * 0.85), w50, min(max(w75, w50 + 3.0), w_u_top * 0.88)],
    )
    wind_speed["high"] = fuzz.trimf(
        wind_universe,
        [
            max(w50 * 0.9, w75 * 0.85),
            min(max(w75 * 1.05, w_hi), w_u_top * 0.95),
            w_u_top,
        ],
    )

    weather_decision = ctrl.Consequent(np.arange(0, 101, 1), "weather_decision")
    weather_decision.defuzzify_method = "centroid"
    weather_decision["sunny"] = fuzz.trimf(weather_decision.universe, [0, 18, 38])
    weather_decision["cloudy"] = fuzz.trimf(weather_decision.universe, [32, 50, 68])
    weather_decision["rain"] = fuzz.trimf(weather_decision.universe, [58, 74, 88])
    weather_decision["storm"] = fuzz.trimf(weather_decision.universe, [78, 92, 100])

    rules = [
        ctrl.Rule(humidity["high"] & precipitation["high"], weather_decision["rain"]),
        ctrl.Rule(temperature["high"] & humidity["low"], weather_decision["sunny"]),
        ctrl.Rule(humidity["medium"] & precipitation["low"], weather_decision["cloudy"]),
        ctrl.Rule(precipitation["high"], weather_decision["rain"]),
        ctrl.Rule(temperature["low"] & humidity["high"], weather_decision["cloudy"]),
        ctrl.Rule(temperature["low"] & precipitation["high"], weather_decision["storm"]),
        ctrl.Rule(
            temperature["high"] & precipitation["low"] & humidity["medium"],
            weather_decision["sunny"],
        ),
        ctrl.Rule(
            temperature["medium"] & humidity["high"] & precipitation["medium"],
            weather_decision["rain"],
        ),
        ctrl.Rule(wind_speed["high"] & precipitation["high"], weather_decision["storm"]),
        ctrl.Rule(wind_speed["high"] & humidity["low"], weather_decision["sunny"]),
        ctrl.Rule(
            wind_speed["medium"] & precipitation["low"] & temperature["medium"],
            weather_decision["cloudy"],
        ),
        # Coverage for dry air (common dead-zone without these rules)
        ctrl.Rule(
            humidity["low"] & precipitation["low"] & temperature["low"],
            weather_decision["sunny"],
        ),
        ctrl.Rule(
            humidity["low"] & precipitation["low"] & temperature["medium"],
            weather_decision["sunny"],
        ),
        ctrl.Rule(
            humidity["low"] & precipitation["low"] & temperature["high"],
            weather_decision["sunny"],
        ),
        ctrl.Rule(
            temperature["low"] & humidity["medium"] & precipitation["low"],
            weather_decision["cloudy"],
        ),
        ctrl.Rule(
            wind_speed["low"] & humidity["medium"] & precipitation["low"],
            weather_decision["cloudy"],
        ),
    ]

    control = ctrl.ControlSystem(rules)
    return FuzzyWeatherSystem(
        control=control,
        temperature=temperature,
        humidity=humidity,
        precipitation=precipitation,
        wind_speed=wind_speed,
        weather_decision=weather_decision,
    )


def infer_fuzzy_decision(
    system: FuzzyWeatherSystem,
    temperature: float,
    humidity: float,
    precipitation: float,
    wind_speed: float,
) -> tuple[float, str]:
    """
    Run fuzzification → rule aggregation → defuzzification.
    Returns (crisp score 0–100, winning linguistic output label).
    """
    sim = ctrl.ControlSystemSimulation(system.control)
    t_lo, t_hi = float(system.temperature.universe[0]), float(system.temperature.universe[-1])
    sim.input["temperature"] = _clip(temperature, t_lo, t_hi)
    sim.input["humidity"] = _clip(humidity, 0, 100)
    p_lo, p_hi = float(system.precipitation.universe[0]), float(system.precipitation.universe[-1])
    sim.input["precipitation"] = _clip(max(0.0, precipitation), p_lo, p_hi)
    w_lo, w_hi = float(system.wind_speed.universe[0]), float(system.wind_speed.universe[-1])
    sim.input["wind_speed"] = _clip(max(0.0, wind_speed), w_lo, w_hi)
    sim.compute()
    raw = sim.output.get("weather_decision")
    if raw is None or not np.isfinite(raw):
        score = _fallback_crisp_score(temperature, humidity, precipitation, wind_speed)
    else:
        score = float(raw)
    label = winning_output_term(system.weather_decision, score)
    return score, label


def winning_output_term(decision: ctrl.Consequent, score: float) -> str:
    """Pick consequent term with highest membership at the defuzzified score."""
    best_l, best_mu = "cloudy", -1.0
    for name, term in decision.terms.items():
        mu = float(fuzz.interp_membership(decision.universe, term.mf, score))
        if mu > best_mu:
            best_mu, best_l = mu, name
    return best_l


def coarse_bucket_from_ml_label(ml_label: str) -> str:
    """Map fine-grained repository label to a coarse bucket for comparison."""
    s = str(ml_label).lower()
    if any(k in s for k in ("thunder", "storm", "hurricane")):
        return "storm"
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
        )
    ):
        return "rain"
    if any(k in s for k in ("sun", "clear")):
        return "sunny"
    if "cloud" in s or "overcast" in s:
        return "cloudy"
    if s in ("mist", "fog", "haze"):
        return "cloudy"
    return "cloudy"


def hybrid_ml_fuzzy_decision(
    ml_predicted_label: str,
    temperature: float,
    humidity: float,
    precipitation: float,
    fuzzy: FuzzyWeatherSystem,
    *,
    wind_speed: float = 10.0,
) -> dict[str, Any]:
    """
    Combine XGBoost class label with fuzzy inference for a final narrative + actions.

    Fuzzy output can reinforce or contradict the ML label; recommendations prioritize
    safety (rain/wind-style cues from precip + fuzzy).
    """
    fuzzy_score, fuzzy_label = infer_fuzzy_decision(
        fuzzy, temperature, humidity, precipitation, wind_speed
    )
    ml_bucket = coarse_bucket_from_ml_label(ml_predicted_label)

    agree = ml_bucket == fuzzy_label
    notes: list[str] = []
    if not agree:
        notes.append(
            f"ML coarse bucket ({ml_bucket}) differs from fuzzy output ({fuzzy_label}); "
            f"using both for guidance."
        )
    else:
        notes.append(f"ML coarse view ({ml_bucket}) agrees with fuzzy ({fuzzy_label}).")

    # Refinement: if fuzzy strongly indicates rain from precip/humidity, surface that
    if fuzzy_label in ("rain", "storm") and fuzzy_score >= 62:
        final = f"{fuzzy_label} (fuzzy-strong; ML={ml_predicted_label!r})"
        notes.append("Fuzzy logic emphasizes wet conditions given inputs.")
    elif fuzzy_label == "sunny" and fuzzy_score <= 35:
        final = f"sunny (fuzzy-strong; ML={ml_predicted_label!r})"
        notes.append("Fuzzy logic supports clear/dry conditions.")
    else:
        final = f"{fuzzy_label} (blend; ML={ml_predicted_label!r})"

    recs: list[str] = []
    if ml_bucket == "rain" and fuzzy_label not in ("rain", "storm"):
        recs.append(
            "ML indicates rain-type conditions nearby even if current precip is low; "
            "consider an umbrella."
        )
    if fuzzy_label in ("rain", "storm") or precipitation > 1.0:
        recs.append("Carry an umbrella or rain gear; roads may be slick.")
    if temperature >= 32:
        recs.append("High heat: limit strenuous outdoor activity; stay hydrated.")
    elif temperature <= 2:
        recs.append("Cold conditions: dress warmly; watch for ice if precip is present.")
    if humidity > 85 and precipitation > 0.5:
        recs.append("Very humid with rain: allow extra travel time.")
    if not recs:
        recs.append("Conditions look moderate; usual outdoor precautions apply.")

    return {
        "ml_prediction": ml_predicted_label,
        "ml_coarse_bucket": ml_bucket,
        "fuzzy_crisp_score": round(fuzzy_score, 2),
        "fuzzy_weather_decision": fuzzy_label,
        "final_decision": final,
        "recommendations": recs,
        "notes": notes,
    }


def print_hybrid_report(result: dict[str, Any]) -> None:
    print("\n--- Hybrid (ML + fuzzy) ---")
    print(f"ML prediction:        {result['ml_prediction']!r}")
    print(f"ML coarse bucket:     {result['ml_coarse_bucket']}")
    print(f"Fuzzy crisp score:    {result['fuzzy_crisp_score']} (0=drier, 100=wetter/stormy)")
    print(f"Fuzzy decision:       {result['fuzzy_weather_decision']}")
    print(f"Final decision:       {result['final_decision']}")
    for n in result["notes"]:
        print(f"  Note: {n}")
    print("Recommendations:")
    for r in result["recommendations"]:
        print(f"  • {r}")

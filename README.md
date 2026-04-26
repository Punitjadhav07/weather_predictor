# 🌦️ Hybrid Weather Prediction Decision Support System:-

A **decision support system under uncertainty** that combines **XGBoost** (27-class classifier) with **Fuzzy Logic** to provide actionable weather recommendations — not just predictions.

> This is **not** a standard prediction tool. It is a **decision system** that explicitly handles conflicting signals, uncertain confidence, and ambiguous conditions using calibrated thresholds and hybrid fusion.

---

## 🚀 What This System Does

| Capability | How |
|---|---|
| **Predicts** fine-grained weather | XGBoost trained on 27 normalised weather classes |
| **Handles ambiguity** | Fuzzy logic provides a parallel interpretable 4-class wet/dry assessment |
| **Detects conflicting signals** | Disagreement metric compares ML rain probability vs fuzzy wet score |
| **Calibrates confidence** | Validation-set percentiles (p25/p75) define Low/Medium/High bands — no hardcoded thresholds |
| **Gives directional uncertainty** | When uncertain, tells you *which direction* (rain vs clear) rather than a generic "unknown" |
| **Prioritises safety** | Recommendations ordered: Safety → Weather Prep → Comfort |

---

## 🧠 System Architecture

```
                 ┌──────────────┐
  User Inputs ──►│   Features   │
                 │  Engineering │
                 └──────┬───────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   ┌────────────────┐     ┌─────────────────┐
   │   XGBoost ML   │     │   Fuzzy Logic   │
   │  (27 classes)  │     │  (4 categories) │
   │                │     │  sunny/cloudy/  │
   │  predict_proba │     │  rain/storm     │
   └───────┬────────┘     └────────┬────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              ┌────────────────┐
              │ Hybrid Fusion  │
              │                │
              │ • Coarse map   │
              │ • Confidence   │
              │   bands        │
              │ • Disagreement │
              │   metric       │
              │ • Directional  │
              │   uncertainty  │
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │  Streamlit UI  │
              │                │
              │ Decision +     │
              │ Reasoning +    │
              │ Recommendations│
              └────────────────┘
```

---

## 📂 Project Structure

```
Weather_prediction/
├── streamlit_app.py              # Streamlit UI (entry point)
├── requirements.txt              # Python dependencies
├── README.md
│
├── weather_dss/                  # Core package
│   ├── __init__.py
│   ├── data_processing.py        # Phase 1: load, clean, engineer features
│   ├── ml_xgboost.py             # Phase 2: XGBoost training + evaluation
│   ├── fuzzy_weather.py          # Phase 3: fuzzy control system (scikit-fuzzy)
│   ├── hybrid_fusion.py          # Phase 4: ML↔fuzzy fusion + recommendations
│   ├── prediction_utils.py       # Feature vector builder + predict helpers
│   └── artifacts.py              # Model artifact paths and metadata I/O
│
├── scripts/                      # Offline / training scripts
│   ├── save_model_artifacts.py   # Train + export model bundle with calibrated thresholds
│   ├── run_phase2_ml.py          # Standalone Phase 2 runner
│   ├── run_phase3_fuzzy.py       # Standalone Phase 3 runner
│   └── verify_phase1.py          # Data processing sanity checks
│
├── Dataset/                      # GlobalWeatherRepository CSV
├── models/                       # Saved model bundle (xgboost_bundle.joblib)
└── outputs/                      # Feature importance plots, logs
```

---

## ⚙️ Technologies

| Component | Technology |
|---|---|
| ML classifier | XGBoost (multiclass, 27 labels) |
| Fuzzy reasoning | scikit-fuzzy (trimf/trapmf MFs, centroid defuzzification) |
| Data processing | pandas, NumPy, scikit-learn |
| Confidence calibration | Validation-set percentiles (p25 = Low, p75 = High) |
| Disagreement threshold | 75th percentile of `\|ML_rain_prob − fuzzy_wet_norm\|` on validation set |
| UI | Streamlit |
| Visualisation | Matplotlib |

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Punitjadhav07/weather_predictor.git
cd weather_predictor
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** If XGBoost fails to import, install OpenMP first: `brew install libomp`

### 4. Train the model and export artifacts

```bash
python scripts/save_model_artifacts.py
```

This trains XGBoost with validation tuning, computes calibrated thresholds, and saves everything to `models/xgboost_bundle.joblib`.

**What gets saved in the bundle:**
- Trained XGBoost model
- LabelEncoder (27 classes)
- Feature column order
- Confidence bands: `confidence_low_threshold` (p25), `confidence_high_threshold` (p75)
- Disagreement threshold: `disagreement_high_threshold` (p75 of ML vs fuzzy disagreement)

### 5. Launch the app

```bash
streamlit run streamlit_app.py
```

---

## 🔀 How Hybrid Fusion Works

### Step 1 — Coarse Category Mapping

Both ML and fuzzy outputs are projected to 4 coarse categories for comparison:

| Coarse Category | ML Keywords | Fuzzy Label |
|---|---|---|
| **storm-like** | thunder, storm, hurricane, tornado, tropical | storm |
| **rain-like** | rain, drizzle, shower, sleet, snow, ice, pellets, freezing | rain |
| **dry** | sun, clear | sunny |
| **neutral** | cloud, overcast, mist, fog, haze, other | cloudy |

> **Why is snow under "rain-like"?** Snow/ice produce wet surfaces and require the same safety precautions (carry gear, travel caution). Keeping 4 categories avoids complicating the fuzzy comparison.

### Step 2 — Confidence Classification

ML confidence (`max(predict_proba)`) is classified using validation-calibrated bands:

```
Low    ←  confidence ≤ p25 of validation max-probabilities
Medium ←  p25 < confidence < p75
High   ←  confidence ≥ p75
```

No hardcoded thresholds — bands adapt to the trained model's actual behaviour.

### Step 3 — Disagreement Metric

```
disagreement = | ML_rain_probability − fuzzy_wet_score / 100 |
```

- `ML_rain_probability` = sum of predict_proba over all rain-like + storm-like classes
- `fuzzy_wet_score` = crisp defuzzified output (0–100)
- Threshold = 75th percentile of disagreement on validation set

### Step 4 — Fusion Decision

```
IF coarse categories match → trust ML
  (UNLESS medium confidence + high disagreement → uncertain)
ELIF ML confidence is HIGH → trust ML
ELIF ML confidence is LOW  → trust fuzzy
ELSE → directional uncertainty
```

### Step 5 — Directional Uncertainty

When the system cannot confidently choose, it gives a **directional hint** instead of a generic "uncertain":

```
ML rain probability ≥ 50% → "Possible Rain ⚠️"
ML rain probability <  50% → "Possibly Clear 🌤️"
```

---

## 🖥️ UI Output

The Streamlit interface shows exactly 4 sections:

| Section | Description |
|---|---|
| **Final Decision** | Large, colour-coded verdict (🟢 sunny/cloudy, 🟡 rain, 🔴 storm) |
| **Confidence Level** | Badge with colour: 🔴 Low, 🟡 Medium, 🟢 High |
| **Reasoning** | 2–3 concise bullets explaining *why* the decision was made |
| **Recommendations** | Priority-ordered: Safety → Weather Prep → Comfort (max 7 items) |

All technical details (ML probabilities, disagreement metric, feature vector) are hidden in an expandable **🔍 Technical Details** section.

---

## 🧪 Example Scenarios

| Inputs | Decision | Reasoning |
|---|---|---|
| High humidity + heavy rain | **Possible Rain ⚠️** | ML and fuzzy agree: both indicate rain conditions |
| Hot + dry + clear sky | **Sunny ☀️** | ML prediction (*sunny*) selected with High confidence |
| Moderate temp + mixed signals | **Possibly Clear 🌤️** | Moderate uncertainty: confidence bands used to choose |
| Low temp + high wind + precip | **Possible Storm ⚠️** | High disagreement detected between ML and fuzzy |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Validation accuracy | ~63% |
| Test accuracy (holdout) | ~63% |
| Number of classes | 27 (after merging rare labels into "other") |
| Train/Val/Test split | 60% / 20% / 20% (stratified) |

> Accuracy is affected by the high number of fine-grained weather labels. The hybrid system compensates by using fuzzy reasoning and calibrated confidence bands to handle misclassifications gracefully.

---

## 🎯 Future Improvements

- Label simplification strategies for higher base accuracy
- Advanced probability calibration (Platt scaling / isotonic regression)
- Real-time weather API integration for live inference
- Cloud deployment (Streamlit Cloud / Docker)
- Temporal features (hour-of-day, day-of-week patterns)

---

## 👨‍💻 Author

**Punit Jadhav**
AI/ML Enthusiast

---

## ⭐ Key Takeaway

This project demonstrates **decision intelligence under uncertainty** — not just maximising prediction accuracy, but building a system that knows *when it doesn't know* and still provides actionable, safety-prioritised guidance.

# 🌦️ Hybrid Weather Prediction + Decision Support System

A **machine learning + soft computing based system** that predicts weather conditions and provides actionable recommendations using a hybrid approach combining **XGBoost** and **Fuzzy Logic**.

---

## 🚀 Project Overview

This project addresses the limitations of traditional weather prediction models by integrating:-

* **Machine Learning (XGBoost)** for probabilistic prediction
* **Fuzzy Logic** for handling uncertainty and ambiguity
* **Hybrid Decision Fusion** for intelligent recommendations

The system does not rely solely on prediction accuracy but improves **decision reliability under uncertainty**.

---

## 🧠 Key Features

* 📊 Data-driven weather prediction using XGBoost
* 🌫️ Fuzzy logic system for uncertainty modeling
* 🔀 Hybrid fusion of ML confidence + fuzzy reasoning
* 📉 Confidence calibration using validation percentiles
* 🌐 Interactive UI built with Streamlit
* ⚡ Real-time decision support (umbrella, travel caution, etc.)

---

## 🏗️ System Architecture

```
Data → Feature Engineering → XGBoost Model → Fuzzy Logic → Hybrid Fusion → UI Output
```

---

## ⚙️ Technologies Used

* Python
* XGBoost
* Scikit-learn
* Scikit-fuzzy
* Streamlit
* Pandas, NumPy, Matplotlib

---

## 📂 Project Structure

```
weather_dss/
├── data_processing.py
├── ml_xgboost.py
├── fuzzy_weather.py
├── hybrid_fusion.py
├── prediction_utils.py

scripts/
├── save_model_artifacts.py
├── run_phase2_ml.py
├── run_phase3_fuzzy.py

streamlit_app.py
requirements.txt
```

---

## ▶️ How to Run

### 1. Clone repository

```bash
git clone https://github.com/Punitjadhav07/weather_predictor.git
cd weather_predictor
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train model & generate artifacts

```bash
python scripts/save_model_artifacts.py
```

### 5. Run the application

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Example Output

| Input Condition                    | Output                             |
| ---------------------------------- | ---------------------------------- |
| High humidity + high precipitation | Rain (Carry umbrella)              |
| High temperature + low humidity    | Sunny (Outdoor safe)               |
| Conflicting signals                | Possible rain (Precaution advised) |

---

## 🔬 Hybrid Decision Logic

The system combines:

* **ML Prediction + Confidence Score**
* **Fuzzy Inference Output**
* **Dynamic Fusion Rules**

This ensures:

* Better handling of uncertain conditions
* Reduced overconfidence in noisy predictions
* More reliable recommendations

---

## 📊 Model Performance

* Validation Accuracy: ~63%
* Test Accuracy: ~63%

Note: Accuracy is affected by fine-grained weather labels. The hybrid system compensates using fuzzy reasoning.

---

## 🎯 Future Improvements

* Label simplification for higher accuracy
* Advanced probability calibration
* Real-time weather API integration
* Deployment on cloud

---

## 👨‍💻 Author

**Punit Jadhav**
AI/ML Enthusiast

---

## ⭐ Final Note

This project focuses on **decision intelligence**, not just prediction accuracy, making it more practical for real-world uncertain environments.


# 🌾 Eco-Harvesters: AI-Driven Agricultural Optimization Platform

**Eco-Harvesters** is a machine learning-based platform designed to recommend optimal crops for farming based on soil composition, weather conditions, and engineered environmental interactions.

---

## ✅ Features

- **ETL pipeline** using Pandas and in-memory SQL (SQLite)
- **Exploratory Data Analysis** and ANOVA for statistical significance of rainfall on crop yield
- **Feature Engineering**: 
  - NPK sum
  - Temperature-Humidity interaction
- **XGBoost classifier** for crop prediction
- **Scikit-learn pipeline** for data preprocessing and evaluation
- **Model persistence** using joblib
- **Live crop prediction script (`predict_crop.py`)**

---

## 🚀 Pipeline Output Snapshot

```bash
🔍 Loading dataset...
🧼 Cleaning and preprocessing...
🧪 Performing statistical tests...
ANOVA result on rainfall by label: F = 605.53, p = 0.00000
📊 Running ETL via SQL...
🚀 Splitting and scaling features...
🌱 Training XGBoost model...
🧾 Evaluating model...
Model R² Score: 0.96
Model MSE: 1.51
📈 Feature Importances:
N: 0.1734
P: 0.1412
K: 0.1592
temperature: 0.0901
humidity: 0.1399
ph: 0.0570
rainfall: 0.1298
NPK_sum: 0.0397
temp_humidity_interaction: 0.0698
✅ Pipeline completed successfully.
```

---

## 📂 Project Structure

```
eco_harvesters_full_project/
│
├── main.py               # Runs full training pipeline
├── pipeline.py           # Core logic for ETL, training, evaluation
├── predict_crop.py       # Interactive script to predict crop from user input
├── xgb_model.pkl         # Trained XGBoost model
├── scaler.pkl            # Trained StandardScaler
├── label_encoder.pkl     # Trained LabelEncoder
└── Crop_recommendation.csv # Dataset used
```

---

## ⚙️ How to Use

1. **Run training pipeline:**

```bash
python main.py
```

2. **Predict a crop based on your input:**

```bash
python predict_crop.py
```

3. **Sample Prediction Output:**

```bash
🌾 Recommended crop for the given conditions: rice
```

---

## 📊 Model Performance

- **R² Score:** `0.96`
- **Mean Squared Error:** `1.51`
- **Top Features:**
  - N (Nitrogen)
  - P (Phosphorus)
  - K (Potassium)
  - Humidity
  - Rainfall

---

## 📌 Next Steps

- Add location-wise crop yield data
- Build a Streamlit-based user interface
- Visualize crop suitability by region

---

> Designed with care by Atharva Patil 💡

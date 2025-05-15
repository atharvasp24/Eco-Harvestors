import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f_oneway
import sqlite3
import joblib




def run_pipeline():
    print("🔍 Loading dataset...")
    df = pd.read_csv("Crop_recommendation.csv")  # Make sure this file exists in the same directory

    print("🧼 Cleaning and preprocessing...")
    df.dropna(inplace=True)

    # Feature Engineering
    df["NPK_sum"] = df["N"] + df["P"] + df["K"]
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]

    # Store in SQLite for simulated SQL-based ETL
    conn = sqlite3.connect(":memory:")
    df.to_sql("crop_data", conn, index=False, if_exists="replace")

    print("🧪 Performing statistical tests...")
    # ANOVA for rainfall grouped by label
    groups = [group["rainfall"].values for name, group in df.groupby("label")]
    anova_result = f_oneway(*groups)
    print(f"ANOVA result on rainfall by label: F = {anova_result.statistic:.2f}, p = {anova_result.pvalue:.5f}")

    print("📊 Running ETL via SQL...")
    etl_df = pd.read_sql_query("""
        SELECT N, P, K, temperature, humidity, ph, rainfall,
               N+P+K AS NPK_sum,
               temperature * humidity AS temp_humidity_interaction,
               label
        FROM crop_data
    """, conn)

    print("🚀 Splitting and scaling features...")
    X = etl_df.drop("label", axis=1)
    y = etl_df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    print("🌱 Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    print("🧾 Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model R² Score: {r2:.2f}")
    print(f"Model MSE: {mse:.2f}")

    print("📈 Feature Importances:")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    import joblib
    _ = joblib.dump(model, "xgb_model.pkl")
    _ = joblib.dump(scaler, "scaler.pkl")
    _ = joblib.dump(label_encoder, "label_encoder.pkl")

    print("✅ Pipeline completed successfully.")


import joblib
import numpy as np

# Load the saved model and preprocessing tools
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Feature Engineering
    NPK_sum = N + P + K
    temp_humidity_interaction = temperature * humidity

    # Create the feature array
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall, NPK_sum, temp_humidity_interaction]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict and decode label
    pred_encoded = model.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return pred_label

# Example usage
if __name__ == "__main__":
    # Test input
    N = 90
    P = 42
    K = 43
    temperature = 25.0
    humidity = 80.0
    ph = 6.5
    rainfall = 200.0

    crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    print(f"🌾 Recommended crop for the given conditions: {crop}")

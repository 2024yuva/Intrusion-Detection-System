import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

MODEL_PATH = 'models/svm_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
DATA_PATH = 'data/simulated_google_traffic.csv'  # ✅ Corrected path

def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def preprocess_data(df, features=['length', 'src_port', 'dst_port']):
    try:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(df[features])
        return X_scaled
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def predict_anomalies(X_scaled):
    try:
        model = joblib.load(MODEL_PATH)
        raw_preds = model.predict(X_scaled)
        preds = [0 if p == 1 else 1 for p in raw_preds]
        return preds
    except Exception as e:
        print(f"Error during prediction: {e}")
        return []

def run_prediction():
    df = load_data()
    if df.empty:
        return []

    X_scaled = preprocess_data(df)
    if X_scaled is None:
        return []

    df['predicted'] = predict_anomalies(X_scaled)
    return df[['timestamp', 'length', 'predicted']].tail(50).to_dict(orient='records')

if __name__ == '__main__':
    results = run_prediction()
    for record in results:
        print(record)
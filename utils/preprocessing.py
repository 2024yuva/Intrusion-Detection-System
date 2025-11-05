import pandas as pd
import joblib

def load_data(path='data/simulated_google_traffic.csv'):
    return pd.read_csv(path)

def preprocess(df, features=['length', 'src_port', 'dst_port']):
    scaler = joblib.load('models/scaler.pkl')
    return scaler.transform(df[features])

def predict_svm(X_scaled):
    model = joblib.load('models/svm_model.pkl')
    return [0 if p == 1 else 1 for p in model.predict(X_scaled)]
import pandas as pd
import numpy as np
import joblib
import os
from xai import explain_scores

DATA_PATH = "data/simulated_google_traffic.csv"
MODEL_DIR = "models"

def load_models():
    """Load trained models and scaler"""
    try:
        svm = joblib.load(os.path.join(MODEL_DIR, 'model_svm.pkl'))
        iforest = joblib.load(os.path.join(MODEL_DIR, 'model_if.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        return svm, iforest, scaler
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        return None, None, None


def load_default_dataframe():
    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            raise ValueError("Dataset is empty")

        # ensure required cols exist
        for col in ["src_port", "dst_port", "length"]:
            if col not in df.columns:
                df[col] = np.random.randint(100, 10000, size=len(df))

        df.fillna(0, inplace=True)
        return df

    except Exception as e:
        print("⚠️ Error loading dataset:", e)
        return pd.DataFrame({
            "src_port": np.random.randint(1000, 9000, 100),
            "dst_port": np.random.choice([80, 443, 22, 53], 100),
            "length": np.random.randint(40, 1500, 100)
        })


def predict_all(df):
    if df.empty:
        return {}

    # ----- FEATURE EXTRACTION -----
    feat_cols = [c for c in df.columns if c not in ["timestamp"]]
    # Ensure we use the same columns as training
    train_cols = ['src_port', 'dst_port', 'length']
    available_cols = [c for c in train_cols if c in df.columns]
    
    if not available_cols:
        return {}
        
    X = df[available_cols].values
    
    # Load models
    svm, iforest, scaler = load_models()
    
    if svm is None or iforest is None:
        # Fallback if models not found
        return {"error": "Models not loaded"}

    # Scale features
    X_scaled = scaler.transform(X)

    # ----- ENSEMBLE PREDICTION (SOFT VOTING) -----
    # Get decision function scores (higher is better/normal)
    # SVM: positive for inliers, negative for outliers
    score_svm = svm.decision_function(X_scaled)
    
    # Isolation Forest: positive for inliers, negative for outliers
    score_if = iforest.decision_function(X_scaled)
    
    # Normalize scores to 0-1 range (approximate) for combination
    # Sigmoid function to map (-inf, inf) to (0, 1)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    prob_svm = sigmoid(score_svm)
    prob_if = sigmoid(score_if)
    
    # Combined score (Average)
    combined_score = (prob_svm + prob_if) / 2
    
    # Threshold for anomaly (0.5 is the decision boundary for sigmoid)
    # If combined probability < 0.5, it's an anomaly (1), else normal (0)
    preds = (combined_score < 0.5).astype(int)
    
    # Invert scores for visualization (higher = more anomalous)
    # We want 0 to 1, where 1 is high anomaly score
    vis_scores = 1 - combined_score

    # ----- RANDOM CATEGORY LABELS -----
    possible_threats = ["DDoS", "SQL Injection", "Brute Force", "Normal"]
    categories = []
    for _ in range(10):
        label = np.random.choice(possible_threats, p=[0.3, 0.2, 0.2, 0.3])
        categories.append({
            "label": label,
            "score": round(np.random.uniform(0.7, 0.99), 2)
        })

    # ----- XAI -----
    # Use SVM for explanation as it's distance-based and intuitive
    xai = explain_scores("svm", X_scaled, vis_scores)

    # ====================================================
    # 🔥 DENSITY HEATMAP (src_port vs length)
    # ====================================================
    bin_x = pd.cut(df["src_port"], bins=20, labels=False)
    bin_y = pd.cut(df["length"], bins=20, labels=False)

    df["bin_x"] = bin_x
    df["bin_y"] = bin_y

    heat = (
        df.groupby(["bin_x", "bin_y"])
        .size()
        .reset_index(name="count")
        .dropna()
    )

    heatmap_data = []
    for _, r in heat.iterrows():
        heatmap_data.append({
            "x": int(r["bin_x"]),
            "y": int(r["bin_y"]),
            "v": int(r["count"])
        })

    # ====================================================
    # 🔥 NORMAL vs ANOMALY SPLIT
    # ====================================================
    normal_vals = df.loc[preds == 0, "length"].tolist()
    anomaly_vals = df.loc[preds == 1, "length"].tolist()

    # Get individual model predictions
    pred_svm = (prob_svm < 0.5).astype(int)
    pred_if = (prob_if < 0.5).astype(int)
    
    # Calculate agreement percentage
    agreement = np.mean(pred_svm == pred_if) * 100
    
    # Calculate statistics
    total_packets = len(preds)
    anomaly_count = int(np.sum(preds))
    normal_count = total_packets - anomaly_count
    anomaly_percentage = (anomaly_count / total_packets * 100) if total_packets > 0 else 0

    # ====================================================
    # FINAL RETURN
    # ====================================================
    return {
        "if": {"scores": vis_scores.tolist(), "labels": preds.tolist()},
        "normal": normal_vals,
        "anomaly": anomaly_vals,
        "categories": categories,
        "xai_proxy": xai,
        "features": available_cols,
        "heatmap": heatmap_data,
        # NEW: Individual model predictions
        "model_predictions": {
            "svm": pred_svm.tolist(),
            "isolation_forest": pred_if.tolist(),
            "ensemble": preds.tolist(),
            "agreement_percentage": round(agreement, 2)
        },
        # NEW: Statistics
        "statistics": {
            "total_packets": total_packets,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_percentage": round(anomaly_percentage, 2)
        }
    }


if __name__ == "__main__":
    df = load_default_dataframe()
    out = predict_all(df)
    print("Preview:", list(out.keys()))

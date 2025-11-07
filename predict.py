import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xai import explain_scores
from intel import check_ip_threat
from threat_categorizer import zero_shot_category  # 👈 Threat classification
import random
import time

DATA_PATH = "data/simulated_google_traffic.csv"


def load_default_dataframe():
    """Load dataset or fallback to simulated data."""
    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            raise ValueError("Dataset is empty!")

        df = df.dropna(subset=["src_ip", "dst_ip"])
        df.fillna(0, inplace=True)
        return df

    except Exception as e:
        print("⚠️ Error loading CSV:", e)
        # fallback dummy traffic
        return pd.DataFrame({
            "timestamp": pd.date_range(start="2025-01-01", periods=100, freq="s"),
            "src_ip": [f"192.168.1.{i%255}" for i in range(100)],
            "dst_ip": [f"10.0.0.{i%255}" for i in range(100)],
            "length": np.random.randint(40, 1500, 100),
            "src_port": np.random.randint(1000, 9000, 100),
            "dst_port": np.random.choice([80, 443, 22, 53], 100),
            "protocol": np.random.choice(["TCP", "UDP", "ICMP"], 100),
        })


def predict_all(df):
    """Perform anomaly detection, threat classification & enrichment."""
    if df.empty:
        return {}

    start_time = time.time()

    # === 1️⃣ Features for model ===
    feat_cols = ["length", "src_port", "dst_port"]
    X = df[feat_cols].values

    # === 2️⃣ Isolation Forest — detect anomalies ===
    iforest = IsolationForest(contamination=0.12, random_state=int(time.time()) % 1000)
    iforest.fit(X)
    raw_scores = -iforest.score_samples(X)

    # Add noise for dynamic simulation
    noise = np.random.normal(0, 0.05, len(raw_scores))
    scores = np.clip(raw_scores + noise, 0, 1)

    # Pick adaptive threshold
    threshold = np.percentile(scores, 88)
    preds = (scores > threshold).astype(int)

    # Random live spikes for visualization
    for i in random.sample(range(len(preds)), k=min(10, len(preds))):
        preds[i] = 1

    # === 3️⃣ Explainability (proxy SHAP values) ===
    xai = explain_scores("if", X, scores)

    # === 4️⃣ Threat Categorization ===
    categories = []
    for _, row in df.iterrows():
        try:
            threat = zero_shot_category(row.to_dict())
        except Exception as e:
            threat = {"label": "Normal", "confidence": 0.0}
            print(f"⚠️ Threat classification failed for {row.get('src_ip')}: {e}")
        categories.append(threat)

    # === 5️⃣ Threat Intelligence (only for anomalies) ===
    anomalies = df[preds == 1]
    intel_results = []
    if not anomalies.empty:
        unique_ips = anomalies["src_ip"].unique()[:10]
        for ip in unique_ips:
            info = check_ip_threat(ip)
            intel_results.append(info)
    else:
        print("ℹ️ No anomalies detected in current batch.")

    end_time = time.time()
    print(f"✅ Prediction completed in {end_time - start_time:.2f}s - {len(df)} records processed")

    # === 6️⃣ Final JSON payload ===
    return {
        "if": {"scores": scores.tolist(), "labels": preds.tolist()},
        "xai_proxy": xai,
        "features": feat_cols,
        "categories": categories,
        "intel": intel_results,
    }

# End of predict.py ✅

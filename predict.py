import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xai import explain_scores

DATA_PATH = "data/simulated_google_traffic.csv"


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
    X = df[feat_cols].values

    # ----- ISOLATION FOREST -----
    iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iforest.fit(X)
    scores = -iforest.score_samples(X)
    preds = (scores > np.percentile(scores, 95)).astype(int)

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
    xai = explain_scores("if", X, scores)

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

    # ====================================================
    # FINAL RETURN
    # ====================================================
    return {
        "if": {"scores": scores.tolist(), "labels": preds.tolist()},
        "normal": normal_vals,
        "anomaly": anomaly_vals,
        "categories": categories,
        "xai_proxy": xai,
        "features": feat_cols,
        "heatmap": heatmap_data
    }


if __name__ == "__main__":
    df = load_default_dataframe()
    out = predict_all(df)
    print("Preview:", list(out.keys()))

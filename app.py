from flask import Flask, render_template, jsonify
import pandas as pd
from predict import predict_all, load_default_dataframe
from gemini_ai import generate_summary

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-stream", methods=["GET"])
def predict_stream():
    try:
        df = load_default_dataframe().tail(200)
        out = predict_all(df)
        print("📡 Sent Data Snapshot:", {k: type(v) for k, v in out.items()})
        return jsonify(out)
    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)})

@app.route("/dataset-preview", methods=["GET"])
def dataset_preview():
    try:
        df = load_default_dataframe().head(20)
        cols = [c for c in ["timestamp", "length", "src_port", "dst_port", "anomaly"] if c in df.columns]
        return jsonify(df[cols].to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/ai-summary")
def ai_summary():
    try:
        summary = generate_summary()
        return jsonify({"summary": summary})
    except Exception as e:
        print("Gemini Route Error:", e)
        return jsonify({"summary": "Failed to load AI summary."}), 500



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

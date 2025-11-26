from flask import Flask, render_template, jsonify
import pandas as pd
from predict import predict_all, load_default_dataframe
from gemini_ai import generate_summary
from utils.alerts import send_email_alert, get_alert_status
from utils.path_utils import get_resource_path
from datetime import datetime

app = Flask(__name__, 
            template_folder=get_resource_path('templates'),
            static_folder=get_resource_path('static'))

# Alert cooldown tracking
last_alert_time = None
ALERT_COOLDOWN = 60  # seconds

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-stream", methods=["GET"])
def predict_stream():
    global last_alert_time
    try:
        df = load_default_dataframe().tail(200)
        out = predict_all(df)
        
        # Check if we should send an alert
        stats = out.get("statistics", {})
        anomaly_rate = stats.get("anomaly_percentage", 0)
        
        # Trigger alert if anomaly rate > 5% and cooldown passed
        if anomaly_rate > 5.0:
            now = datetime.now()
            should_alert = True
            
            if last_alert_time:
                time_diff = (now - last_alert_time).total_seconds()
                should_alert = time_diff >= ALERT_COOLDOWN
            
            if should_alert:
                send_email_alert(
                    anomaly_count=stats.get("anomaly_count", 0),
                    total_packets=stats.get("total_packets", 0),
                    anomaly_rate=anomaly_rate
                )
                last_alert_time = now
        
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

@app.route("/alert-status")
def alert_status():
    """Get current alert status and history"""
    try:
        status = get_alert_status()
        return jsonify(status)
    except Exception as e:
        print("Alert Status Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
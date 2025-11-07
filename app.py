from flask import Flask, jsonify, render_template
from predict import load_default_dataframe, predict_all
from gemini_ai import generate_summary

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-stream")
def predict_stream():
    try:
        df = load_default_dataframe()
        out = predict_all(df)

        print("=== Predict Stream Called ===")
        print("Scores:", len(out.get("if", {}).get("scores", [])))
        print("Categories:", len(out.get("categories", [])))
        print("Intel:", len(out.get("intel", [])))
        print("==============================")

        return jsonify(out)
    except Exception as e:
        print("❌ ERROR in /predict-stream:", e)
        return jsonify({"error": str(e)}), 500
 

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


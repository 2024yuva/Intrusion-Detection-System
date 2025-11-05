from flask import Flask, render_template, jsonify
from utils.preprocessing import load_data, preprocess, predict_svm

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = load_data()
    X_scaled = preprocess(df)
    df['predicted'] = predict_svm(X_scaled)
    result = df[['timestamp', 'length', 'predicted']].tail(50).to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
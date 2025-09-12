import json
import os
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load
from detectors import analyze_text

ARTIFACTS_DIR = Path('artifacts')
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'
THRESHOLD_PATH = ARTIFACTS_DIR / 'threshold.json'
STATIC_DIR = Path('static')
ALERTS_LOG = ARTIFACTS_DIR / 'alerts.jsonl'

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

model = None
optimal_threshold = 0.5  # Default threshold

# Load model and threshold
if MODEL_PATH.exists():
    model = load(MODEL_PATH)
    print("✅ Model loaded successfully")
    
# Load optimal threshold if available
if THRESHOLD_PATH.exists():
    try:
        with open(THRESHOLD_PATH, 'r') as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
        print(f"✅ Optimal threshold loaded: {optimal_threshold:.3f}")
    except Exception as e:
        print(f"⚠️ Failed to load threshold: {e}")
        optimal_threshold = 0.5

ALERT_THRESHOLD = int(os.getenv('ALERT_THRESHOLD', '6'))
ARTIFACTS_DIR.mkdir(exist_ok=True)


@app.get('/')
def index():
    index_path = STATIC_DIR / 'index.html'
    if index_path.exists():
        return send_from_directory(str(STATIC_DIR), 'index.html')
    return jsonify({"message": "UI not found. Use /predict."}), 404


@app.get('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "alert_threshold": ALERT_THRESHOLD,
    })


def classify(text: str):
    global model, optimal_threshold
    if model is None:
        raise ValueError("Model not loaded")
    
    # Get prediction probabilities
    try:
        proba = float(model.predict_proba([text])[0][1])
    except AttributeError:
        # Fallback for models without predict_proba
        decision = model.decision_function([text])[0]
        # Normalize to probability range
        proba = 1 / (1 + np.exp(-decision))  # Sigmoid function
        proba = float(proba)
    
    # Use optimal threshold instead of fixed 0.5
    label = int(proba >= optimal_threshold)
    
    return label, proba


def log_alert(payload: dict):
    try:
        # Ensure the artifacts directory exists
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        with open(ALERTS_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        # Log the error but don't crash the application
        print(f"Warning: Failed to log alert: {e}")


@app.get('/alerts')
def get_alerts():
    if not ALERTS_LOG.exists():
        return jsonify([])
    try:
        with open(ALERTS_LOG, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
        return jsonify(items[-200:])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.delete('/alerts')
def delete_alerts():
    try:
        if ALERTS_LOG.exists():
            ALERTS_LOG.unlink()  # Delete the alerts file
        return jsonify({"message": "Alerts database cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/predict')
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get('text', '')

    try:
        label, proba = classify(text)
        rules = analyze_text(text)
        severity = int(rules.get('severity', 0))
        alert = bool(severity >= ALERT_THRESHOLD or (label == 1 and proba >= 0.9))

        response = {
            "label": label,
            "probability_fraud": proba,
            "rules": rules,
            "severity": severity,
            "alert": alert,
        }

        if alert:
            log_alert({
                "text": text[:1000],
                "probability_fraud": proba,
                "severity": severity,
                "rules": rules,
            })

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.post('/analyze')
def analyze():
    return predict()



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)

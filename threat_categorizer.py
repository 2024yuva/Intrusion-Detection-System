from transformers import pipeline

# Load lightweight model (Zero-shot classification)
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    print("⚠️ Could not load zero-shot model:", e)
    classifier = None

# Define labels for categories
THREAT_LABELS = [
    "DDoS Attack",
    "Port Scanning",
    "SQL Injection",
    "Phishing Attempt",
    "Brute Force Login",
    "Data Exfiltration",
    "Normal Traffic"
]

def zero_shot_category(row_dict):
    """
    Classify network behavior into threat types using zero-shot classification.
    Accepts a row (dict) and returns the top category.
    """

    try:
        # Convert dictionary row to text input
        text_input = " ".join([f"{k}: {v}" for k, v in row_dict.items()])

        if classifier:
            result = classifier(text_input, candidate_labels=THREAT_LABELS)
            top_label = result["labels"][0]
            return {"label": top_label, "score": float(result["scores"][0])}
        else:
            # Fallback if model not loaded
            return {"label": "Normal", "score": 0.0}

    except Exception as e:
        print("Threat categorization error:", e)
        return {"label": "Normal", "score": 0.0}

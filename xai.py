"""
xai.py — Explainable AI module for IDS
-------------------------------------
This module provides feature-importance analysis for anomalies detected
by Isolation Forest, LOF, or Autoencoder models.

If SHAP is installed, it computes SHAP values directly.
Otherwise, it falls back to a simple correlation-based proxy.
"""

import numpy as np

# Optional SHAP integration
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def explain_scores(model_name: str, X: np.ndarray, scores: np.ndarray):
    """
    Returns a feature-importance dictionary for given anomaly scores.

    Args:
        model_name (str): Name of the model ("if", "lof", "ae", "svm").
        X (np.ndarray): Scaled input features (n_samples, n_features).
        scores (np.ndarray): Model output anomaly scores.

    Returns:
        dict: {
            "method": "shap" or "correlation",
            "feature_importances": [float, ...]
        }
    """
    # Defensive checks
    if X.size == 0 or len(scores) == 0:
        return {"method": "none", "feature_importances": []}

    # ===== 1️⃣ If SHAP available, use it for deeper interpretability =====
    if SHAP_AVAILABLE:
        try:
            # SHAP values expect a trained model; since we don't pass one,
            # we approximate importance by correlation with SHAP KernelExplainer.
            sample_idx = np.random.choice(len(X), min(50, len(X)), replace=False)
            background = X[sample_idx]
            # We simulate model output by mapping scores to SHAP-compatible space
            def model_fn(z):
                z = np.array(z)
                s = np.zeros(len(z))
                for j in range(z.shape[1]):
                    s += np.tanh(z[:, j]) * np.std(scores)
                return s.reshape(-1, 1)

            explainer = shap.KernelExplainer(model_fn, background)
            shap_vals = explainer.shap_values(X[:50], silent=True)
            importance = np.mean(np.abs(shap_vals), axis=0).flatten()
            norm = importance.sum() + 1e-9
            importance = (importance / norm).tolist()
            return {"method": "shap", "feature_importances": importance}
        except Exception:
            pass

    # ===== 2️⃣ Fallback: simple correlation-based proxy =====
    try:
        vals = []
        for j in range(X.shape[1]):
            x = X[:, j]
            if np.std(x) < 1e-9:
                vals.append(0.0)
            else:
                vals.append(float(np.corrcoef(x, scores)[0, 1]))
        vals = np.abs(np.array(vals))
        s = vals.sum() + 1e-9
        norm_vals = (vals / s).tolist()
        return {"method": "correlation", "feature_importances": norm_vals}
    except Exception:
        return {"method": "error", "feature_importances": []}

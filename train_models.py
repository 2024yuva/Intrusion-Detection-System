"""
Training script for SVM and Isolation Forest models
Run this to create model_svm.pkl, model_if.pkl and scaler.pkl files
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def load_training_data(filepath='data/simulated_google_traffic.csv'):
    """Load and prepare training data"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} samples from {filepath}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return None


def prepare_features(df):
    """Extract and prepare features for training"""
    # Select numeric features
    feature_cols = ['src_port', 'dst_port', 'length']
    
    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        print("✗ No suitable feature columns found")
        return None, None, None
    
    X = df[available_cols].fillna(0).values
    
    # Get labels if available
    y = df['anomaly'].values if 'anomaly' in df.columns else None
    
    print(f"✓ Extracted features: {available_cols}")
    print(f"  Shape: {X.shape}")
    
    if y is not None:
        anomaly_count = np.sum(y)
        print(f"  Anomalies: {anomaly_count} ({anomaly_count/len(y)*100:.1f}%)")
    
    return X, y, available_cols


def train_one_class_svm(X, contamination=0.1, nu=0.1, kernel='rbf', gamma='auto'):
    """Train One-Class SVM"""
    print(f"\n=== Training One-Class SVM ===")
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    print("✓ SVM trained successfully")
    return model


def train_isolation_forest(X, contamination=0.1, n_estimators=100):
    """Train Isolation Forest"""
    print(f"\n=== Training Isolation Forest ===")
    model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
    model.fit(X)
    print("✓ Isolation Forest trained successfully")
    return model


def evaluate_model(model, X, y=None, name="Model"):
    """Evaluate trained model"""
    print(f"\n=== {name} Evaluation ===")
    
    predictions = model.predict(X)
    # Convert to binary: -1 (outlier) -> 1, 1 (inlier) -> 0
    pred_binary = np.where(predictions == -1, 1, 0)
    
    anomalies_detected = np.sum(pred_binary)
    print(f"  Anomalies detected: {anomalies_detected} ({anomalies_detected/len(X)*100:.1f}%)")
    
    if y is not None:
        print(f"  Confusion Matrix:")
        cm = confusion_matrix(y, pred_binary)
        print(f"    TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")


def save_models(svm, iforest, scaler, model_dir='models'):
    """Save trained models to disk"""
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(svm, os.path.join(model_dir, 'model_svm.pkl'))
    joblib.dump(iforest, os.path.join(model_dir, 'model_if.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print(f"\n✓ Models saved to {model_dir}/")


def main():
    print("=" * 60)
    print("AI-IDS Ensemble Model Training")
    print("=" * 60)
    
    # Load data
    df = load_training_data()
    if df is None: return
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    if X is None: return
    
    # Scale features
    print(f"\n=== Scaling Features ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Features scaled")
    
    # Calculate contamination
    contamination = 0.1
    if y is not None:
        contamination = np.sum(y) / len(y)
        contamination = max(0.01, min(0.5, contamination))
    
    # Train models
    svm = train_one_class_svm(X_scaled, contamination=contamination, nu=contamination)
    iforest = train_isolation_forest(X, contamination=contamination) # IF usually works better on unscaled data, but consistent with pipeline
    
    # Evaluate
    evaluate_model(svm, X_scaled, y, "One-Class SVM")
    evaluate_model(iforest, X, y, "Isolation Forest") # Note: IF using unscaled X here for variety, or X_scaled? Let's use X_scaled for both to keep predict.py simple. Actually, let's use X_scaled for both to keep predict.py simple.
    
    # Re-train IF on scaled data to match predict.py expectation of single scaler
    iforest_scaled = train_isolation_forest(X_scaled, contamination=contamination)
    evaluate_model(iforest_scaled, X_scaled, y, "Isolation Forest (Scaled)")

    # Save models
    save_models(svm, iforest_scaled, scaler)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
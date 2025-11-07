"""
Training script for SVM and other models
Run this to create model.pkl and scaler.pkl files
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import OneClassSVM
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
    """
    Train One-Class SVM for anomaly detection
    
    Args:
        X: Training features
        contamination: Expected proportion of outliers
        nu: SVM parameter (should be close to contamination)
        kernel: Kernel type
        gamma: Kernel coefficient
    
    Returns:
        model: Trained SVM model
    """
    print(f"\n=== Training One-Class SVM ===")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Contamination: {contamination}")
    print(f"  Nu: {nu}")
    print(f"  Kernel: {kernel}")
    
    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma,
        verbose=False
    )
    
    model.fit(X)
    print("✓ Model trained successfully")
    
    return model


def evaluate_model(model, X, y=None):
    """Evaluate trained model"""
    print(f"\n=== Model Evaluation ===")
    
    # Get predictions
    predictions = model.predict(X)
    
    # Convert to binary: -1 (outlier) -> 1, 1 (inlier) -> 0
    pred_binary = np.where(predictions == -1, 1, 0)
    
    anomalies_detected = np.sum(pred_binary)
    print(f"  Anomalies detected: {anomalies_detected} ({anomalies_detected/len(X)*100:.1f}%)")
    
    # Get decision scores
    if hasattr(model, 'score_samples'):
        scores = model.score_samples(X)
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Compare with ground truth if available
    if y is not None:
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(y, pred_binary)
        print(f"    TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"    FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
        
        print(f"\n  Classification Report:")
        print(classification_report(y, pred_binary, 
                                   target_names=['Normal', 'Anomaly'],
                                   zero_division=0))


def save_models(model, scaler, model_dir='models'):
    """Save trained models to disk"""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Models saved:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("AI-IDS Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    if X is None:
        return
    
    # Scale features
    print(f"\n=== Scaling Features ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Features scaled")
    print(f"  Mean: {X_scaled.mean(axis=0)}")
    print(f"  Std:  {X_scaled.std(axis=0)}")
    
    # Calculate contamination from data if labels available
    if y is not None:
        contamination = np.sum(y) / len(y)
        contamination = max(0.01, min(0.5, contamination))  # Clamp between 1% and 50%
    else:
        contamination = 0.1  # Default
    
    # Train model
    model = train_one_class_svm(
        X_scaled, 
        contamination=contamination,
        nu=contamination,
        kernel='rbf',
        gamma='auto'
    )
    
    # Evaluate
    evaluate_model(model, X_scaled, y)
    
    # Save models
    save_models(model, scaler)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start the Flask app: python app.py")
    print("  2. Open http://localhost:5000")
    print("  3. The SVM model will now be used in predictions")


if __name__ == "__main__":
    main()
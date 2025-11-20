"""
Training script for various Anomaly Detection models (One-Class SVM, Isolation Forest, LOF, Autoencoder).
Run this to create model.pkl (or specific model file) and scaler.pkl files.

Usage examples:
python utm.py --model ocsvm
python utm.py --model iforest
python utm.py --model lof
python utm.py --model autoencoder
"""
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import NotFittedError

# New dependencies for Autoencoder
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    # This warning is suppressed if TensorFlow is available but helps the user if it's missing.
    # We still define tf = None for conditional checks.
    tf = None

# Use os.path.join for robust path handling across Windows/Linux
# This expects the data file at: [current_directory]/data/anomaly_detection_dataset.csv
DATA_FILEPATH = os.path.join('data', 'anomaly_detection_dataset.csv')
MODEL_DIR = 'models'
AUTOENCODER_MODEL_NAME = 'autoencoder_model.h5'


def load_training_data(filepath=DATA_FILEPATH):
    """Load and prepare training data"""
    print(f"Attempting to load data from: {os.path.abspath(filepath)}")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} samples from {filepath}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        print("Please ensure the 'data' folder and 'anomaly_detection_dataset.csv' are correctly placed.")
        return None


def prepare_features(df):
    """
    Extract and prepare features for training.
    Uses 'src_port', 'dst_port', 'length' as features.
    Infers binary label (anomaly) from either 'anomaly' or 'label' column.
    """
    # Select numeric features as specified in your base code
    feature_cols = ['src_port', 'dst_port', 'length']
    
    # Check which columns exist in the DataFrame
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if not available_cols:
        print("✗ No suitable feature columns found. Required: src_port, dst_port, length.")
        return None, None, None
    
    X = df[available_cols].fillna(0).values
    
    # --- Label Handling: Check for 'anomaly' first, then infer from 'label' ---
    y = None
    if 'anomaly' in df.columns:
        y = df['anomaly'].values
    elif 'label' in df.columns:
        # Convert the 'label' column (e.g., 'Normal' to 0, everything else to 1)
        y = (df['label'] != 'Normal').astype(int).values
        print("Note: Found 'label' column. Converting non-'Normal' values to 1 (Anomaly) for evaluation.")
    else:
        print("Warning: Could not find 'anomaly' or 'label' column for evaluation.")
        
    print(f"✓ Extracted features: {available_cols}")
    print(f"  Shape: {X.shape}")
    
    if y is not None:
        anomaly_count = np.sum(y)
        print(f"  True Anomalies: {anomaly_count} ({anomaly_count/len(y)*100:.1f}%)")
    
    return X, y, available_cols


def build_autoencoder(input_dim):
    """Builds a simple Keras Sequential Autoencoder model."""
    print(f"  Building Autoencoder with input dimension: {input_dim}")
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(int(input_dim * 2), activation='relu')(input_layer)
    encoder = Dense(int(input_dim), activation='relu')(encoder)
    # Bottleneck/Code Layer (Half of the input dimension for aggressive compression)
    bottleneck = Dense(max(2, input_dim // 2), activation='relu', name='bottleneck')(encoder)
    
    # Decoder
    decoder = Dense(int(input_dim), activation='relu')(bottleneck)
    decoder = Dense(int(input_dim * 2), activation='relu')(decoder)
    output_layer = Dense(input_dim, activation='linear')(decoder) # Output dimension equals input
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder


def train_model(model_name, X, contamination):
    """
    Trains the specified unsupervised anomaly detection model.
    """
    print(f"\n=== Training {model_name} Model ===")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    
    model = None
    
    if model_name == 'ocsvm':
        # One-Class SVM parameters
        nu = contamination
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='auto', verbose=False)
        model.fit(X)
    
    elif model_name == 'iforest':
        # Isolation Forest parameters
        model = IsolationForest(
            contamination=contamination, 
            random_state=42, 
            n_estimators=100,
            warm_start=False,
            n_jobs=-1 # Use all cores
        )
        model.fit(X)
        
    elif model_name == 'lof':
        # Local Outlier Factor parameters
        model = LocalOutlierFactor(
            n_neighbors=20, 
            contamination=contamination, 
            novelty=True # Required to use predict/decision_function on new data
        )
        model.fit(X)
        
    elif model_name == 'autoencoder':
        if tf is None:
            print("✗ Error: TensorFlow/Keras libraries are required for Autoencoder but not found.")
            return None
            
        # NOTE: We train on the full X_scaled set, relying on the majority of data being 'Normal'. 
        
        input_dim = X.shape[1]
        model = build_autoencoder(input_dim)
        
        # Training the Autoencoder
        # X is used as both input (X) and target (y)
        history = model.fit(
            X, X,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=2
        )
        
        print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        
    else:
        print(f"✗ Error: Unknown model name '{model_name}'")
        return None

    print(f"✓ {model_name.upper()} Model trained successfully")
    return model


def evaluate_model(model, X, y=None, contamination=0.1):
    """Evaluate trained model"""
    model_name = type(model).__name__
    print(f"\n=== Model Evaluation (on {model_name}) ===")
    
    try:
        if model_name in ['OneClassSVM', 'IsolationForest', 'LocalOutlierFactor']:
            # Get predictions: 1 (inlier) or -1 (outlier)
            predictions = model.predict(X)
            # Convert to binary: -1 (outlier) -> 1, 1 (inlier) -> 0
            pred_binary = np.where(predictions == -1, 1, 0)
            
            # Get scores for AUC calculation
            scores = None
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
            elif hasattr(model, 'score_samples'):
                scores = model.score_samples(X) * -1
            
        elif model_name == 'Functional': # Keras Model
            # 1. Predict reconstruction error
            reconstructions = model.predict(X, verbose=0)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            
            # 2. Use MSE as the anomaly score
            scores = mse
            
            # 3. Determine threshold (based on contamination or 95th percentile of errors)
            # A higher score is an anomaly, so we use 1 - contamination quantile
            threshold = np.quantile(scores, 1 - contamination)
            
            # 4. Binary predictions based on threshold
            pred_binary = (scores > threshold).astype(int)
            print(f"  Autoencoder Threshold (based on 1-Contamination): {threshold:.4f}")

        else:
            print("✗ Error: Model type not recognized for evaluation.")
            return

    except NotFittedError:
        print("✗ Error: Model is not fitted. Cannot evaluate.")
        return
    except AttributeError as e:
        print(f"✗ Error during prediction/evaluation: {e}. Check model compatibility.")
        return
    
    anomalies_detected = np.sum(pred_binary)
    print(f"  Anomalies detected: {anomalies_detected} ({anomalies_detected/len(X)*100:.1f}%)")
    
    if scores is not None:
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Compare with ground truth if available
        if y is not None:
            # AUC calculation: Higher score should correspond to the positive class (Anomaly = 1)
            auc_scores = scores if model_name != 'OneClassSVM' else scores * -1
            auc = roc_auc_score(y, auc_scores)
            print(f"  ROC-AUC Score: {auc:.4f}")
            
            print(f"\n  Confusion Matrix:")
            cm = confusion_matrix(y, pred_binary)
            print(f"    TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
            print(f"    FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
            
            print(f"\n  Classification Report:")
            print(classification_report(y, pred_binary, 
                                         target_names=['Normal', 'Anomaly'],
                                         zero_division=0))


def save_models(model, scaler, model_dir=MODEL_DIR):
    """Save trained models to disk"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Use the model name in the file name for clarity when running multiple models
    model_name_short = model.__class__.__name__.lower().replace('oneclasssvm', 'ocsvm').replace('isolationforest', 'iforest').replace('localoutlierfactor', 'lof')
    
    model_path = os.path.join(model_dir, f'{model_name_short}_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl') # Scaler is the same regardless of model
    
    if model_name_short == 'functional': # Keras Autoencoder model
        model_path = os.path.join(model_dir, AUTOENCODER_MODEL_NAME)
        model.save(model_path) # Keras native save format (H5 or SavedModel)
        print(f"Note: Autoencoder saved in Keras native format (.h5)")
    else:
        joblib.dump(model, model_path)
    
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Models saved:")
    print(f"  {model_path}")
    print(f"  {scaler_path} (Scaler remains constant)")


def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description="Train an Anomaly Detection model.")
    parser.add_argument('--model', 
                        type=str, 
                        default='ocsvm', 
                        choices=['ocsvm', 'iforest', 'lof', 'autoencoder'],
                        help="The model to train: ocsvm, iforest, lof, or autoencoder. Default is ocsvm.")
    args = parser.parse_args()

    # Check for TensorFlow dependency if Autoencoder is requested
    if args.model == 'autoencoder' and tf is None:
        print("\nFATAL ERROR: Autoencoder requires TensorFlow/Keras. Please install them (e.g., pip install tensorflow).")
        return

    print("=" * 60)
    print(f"AI-IDS Model Training Pipeline - Model: {args.model.upper()}")
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
    print(f"  Mean: {[f'{m:.4f}' for m in X_scaled.mean(axis=0)]}")
    print(f"  Std:  {[f'{s:.4f}' for s in X_scaled.std(axis=0)]}")
    
    # Calculate contamination from data if labels available
    if y is not None:
        # Use the true anomaly fraction as the contamination parameter
        contamination = np.sum(y) / len(y)
        contamination = max(0.01, min(0.5, contamination))  # Clamp between 1% and 50%
        print(f"Note: Using calculated contamination: {contamination:.4f}")
    else:
        # Default fallback
        contamination = 0.1  
        print(f"Note: Using default contamination: {contamination:.4f}")
    
    # Train model
    model = train_model(args.model, X_scaled, contamination)
    
    if model is None:
        return
    
    # Evaluate (passing contamination for Autoencoder thresholding)
    evaluate_model(model, X_scaled, y, contamination=contamination)
    
    # Save models
    save_models(model, scaler)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
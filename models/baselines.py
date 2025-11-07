"""
Baseline anomaly detection models: Isolation Forest and LOF
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class IsolationForestWrapper:
    """Wrapper for Isolation Forest anomaly detection"""
    
    def __init__(self, contamination=0.1, random_state=42, n_estimators=100):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples='auto',
            bootstrap=False
        )
        self.fitted = False
    
    def fit(self, X):
        """Fit the Isolation Forest model"""
        self.model.fit(X)
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict anomalies
        Returns:
            labels: 1 for anomaly, 0 for normal
            scores: anomaly scores (lower = more anomalous)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions: -1 for outliers, 1 for inliers
        predictions = self.model.predict(X)
        
        # Convert to binary: 1 for anomaly (outlier), 0 for normal (inlier)
        labels = np.where(predictions == -1, 1, 0)
        
        # Get anomaly scores (negative = more anomalous)
        scores = self.model.score_samples(X)
        
        return labels, scores


class LOFWrapper:
    """Wrapper for Local Outlier Factor anomaly detection"""
    
    def __init__(self, contamination=0.1, n_neighbors=20):
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=False  # Use for training data
        )
        self.fitted = False
        self.X_train = None
    
    def fit(self, X):
        """Fit LOF model"""
        self.X_train = X
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Predict anomalies
        Returns:
            labels: 1 for anomaly, 0 for normal
            scores: anomaly scores (lower = more anomalous)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Fit and predict on the same data
        predictions = self.model.fit_predict(X)
        
        # Convert to binary: 1 for anomaly, 0 for normal
        labels = np.where(predictions == -1, 1, 0)
        
        # Get negative outlier factor scores
        scores = self.model.negative_outlier_factor_
        
        return labels, scores


if __name__ == "__main__":
    # Test the models
    from sklearn.datasets import make_blobs
    
    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=1, random_state=42)
    # Add some outliers
    outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
    X = np.vstack([X, outliers])
    
    print("Testing Isolation Forest...")
    iso = IsolationForestWrapper(contamination=0.1)
    iso.fit(X)
    labels_if, scores_if = iso.predict(X)
    print(f"Detected {labels_if.sum()} anomalies out of {len(X)} samples")
    print(f"Score range: [{scores_if.min():.3f}, {scores_if.max():.3f}]")
    
    print("\nTesting LOF...")
    lof = LOFWrapper(contamination=0.1)
    lof.fit(X)
    labels_lof, scores_lof = lof.predict(X)
    print(f"Detected {labels_lof.sum()} anomalies out of {len(X)} samples")
    print(f"Score range: [{scores_lof.min():.3f}, {scores_lof.max():.3f}]")
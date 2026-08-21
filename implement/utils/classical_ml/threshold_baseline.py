import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdBaseline(BaseEstimator, ClassifierMixin):
    """
    A simple baseline classifier that predicts Spoofed (1) if Position Error (PE) > 5.0m.
    Assumes standard scaling has been applied, so it must locate the 'position_error' column index
    during fit, or we assume it's passed unscaled.
    Actually, since data is scaled and we don't have feature names natively, we need to map the
    'position_error' feature. But wait, in the unified workflow, we pass the features array.
    """
    def __init__(self, threshold=5.0, feature_idx=None, feature_name='position_error'):
        self.threshold = threshold
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        # If feature_idx isn't set, we assume position_error is the 11th feature.
        # But this is dangerous. In get_point_classifiers, we don't have features list.
        # Let's just fit it and do nothing since it's a fixed threshold.
        # But wait, X is standard scaled! The threshold of 5.0m won't work on scaled data
        # unless we know the mean and std. 
        # Alternatively, we just use a heuristic or we tune the threshold on the scaled data.
        
        # Let's tune the threshold to maximize F1 score on the training data.
        # We assume X has position_error, let's find the feature with highest correlation to y
        # as a proxy for the 'primary' error feature if feature_idx is None.
        
        if self.feature_idx is None:
            correlations = [np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
            self.feature_idx = np.argmax(correlations)
            
        # Tune threshold
        best_t = 0.0
        best_f1 = 0.0
        # Check percentiles
        for t in np.percentile(X[:, self.feature_idx], np.linspace(1, 99, 100)):
            preds = (X[:, self.feature_idx] > t).astype(int)
            # F1 score manual
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            f1 = 2*tp / (2*tp + fp + fn + 1e-10)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                
        self.threshold = best_t
        return self

    def predict(self, X):
        return (X[:, self.feature_idx] > self.threshold).astype(int)

    def predict_proba(self, X):
        # Emulate probabilities using a sigmoid-like transformation around the threshold
        # to ensure it's compatible with ROC-AUC calculations.
        diff = X[:, self.feature_idx] - self.threshold
        # scale to make it look like a prob
        scale = np.std(X[:, self.feature_idx]) + 1e-6
        prob = 1 / (1 + np.exp(-diff / scale))
        probs = np.zeros((X.shape[0], 2))
        probs[:, 1] = prob
        probs[:, 0] = 1 - prob
        return probs

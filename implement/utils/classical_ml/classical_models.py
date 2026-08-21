from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .threshold_baseline import ThresholdBaseline

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def get_point_classifiers(random_state=42, use_class_weights=False, use_smote=False):
    """
    Returns a dictionary of point-wise classifier models configured for Checkpoint-1.
    If use_class_weights is True, sets class_weight='balanced' for supporting models,
    and appends '(No Class Weights)' to keys of non-supporting models.
    SVM RBF Kernel is only included if RAPIDS is active, or if CUDA is available and SMOTE is disabled.
    """
    lr_kwargs = {"max_iter": 1000, "random_state": random_state}
    rf_kwargs = {"random_state": random_state}
    
    if use_class_weights:
        lr_kwargs["class_weight"] = "balanced"
        rf_kwargs["class_weight"] = "balanced"
        
    models = {}
    
    # Random Forest (Supports class_weight)
    models['Random Forest'] = RandomForestClassifier(**rf_kwargs)

    # Logistic Regression (Supports class_weight)
    models['Logistic Regression'] = LogisticRegression(**lr_kwargs)
    
    if HAS_XGBOOST:
        xgb_key = 'XGBoost (No Class Weights)' if use_class_weights else 'XGBoost'
        models[xgb_key] = XGBClassifier(random_state=random_state, eval_metric='logloss')
        
    # Baseline
    models['Threshold(PE)'] = ThresholdBaseline()
        
    return models

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve

def fit_and_score_models(models, X_train, y_train, X_test, y_test):
    """
    Fits models on training data, predicts on test data,
    and returns a dictionary of standard classification performance metrics.
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate probabilities or decision scores for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred  # fallback
            
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec_vals, prec_vals)
            
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'pr_auc': pr_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    return results

def train_and_evaluate_point_models(models, X_train, y_train, X_test, y_test):
    """
    Trains each model in the models dictionary on X_train/y_train,
    predicts on X_test, and returns a dictionary of performance metrics.
    """
    results = fit_and_score_models(models, X_train, y_train, X_test, y_test)
    for name, metrics in results.items():
        print(f"Training {name}...")
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall   : {metrics['recall']*100:.2f}%")
        print(f"  F1-score : {metrics['f1_score']*100:.2f}%")
        print(f"  ROC-AUC  : {metrics['roc_auc']*100:.2f}%")
        print(f"  PR-AUC   : {metrics['pr_auc']*100:.2f}%")
        print(f"  Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    return results

def plot_roc_curves(models, X_test, y_test, output_dir):
    """
    Plots Receiver Operating Characteristic (ROC) curves for all models and saves the plot.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'roc_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves plot to: {plot_path}")

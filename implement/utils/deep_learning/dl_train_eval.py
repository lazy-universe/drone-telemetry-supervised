import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from .dl_models import (
    GRUClassifier, 
    TCNClassifier, 
    CNNGRUClassifier,
    CNNClassifier
)

def find_optimal_threshold(model, X_val, y_val, batch_size=64, device="cpu"):
    """
    Finds the optimal classification threshold on the validation set to maximize the F1-score.
    """
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    y_prob = np.concatenate(all_probs, axis=0).ravel()
    y_target = np.concatenate(all_targets, axis=0).ravel()
    
    best_t = 0.5
    best_f1 = 0.0
    
    # Combination of logarithmic grid for squashed probabilities and linear grid
    candidates = np.concatenate([
        np.logspace(-7, -1, 150),
        np.linspace(0.1, 0.9, 150)
    ])
    
    for t in candidates:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_target, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            
    return best_t



def evaluate_model(model, X_test, y_test, batch_size=64, device="cpu", threshold=0.5):
    """
    Evaluates a trained PyTorch model on a holdout test set using a given decision threshold.
    """
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_target = np.concatenate(all_targets, axis=0)
    
    # Calculate PR-AUC
    prec_vals, rec_vals, _ = precision_recall_curve(y_target.ravel(), y_prob.ravel())
    pr_auc = auc(rec_vals, prec_vals)
    
    cm = confusion_matrix(y_target, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy_score(y_target, y_pred),
        'precision': precision_score(y_target, y_pred, zero_division=0),
        'recall': recall_score(y_target, y_pred, zero_division=0),
        'recall_real': recall_0,
        'recall_spoof': recall_1,
        'fp_count': int(fp),
        'tp_count': int(tp),
        'f1_score': f1_score(y_target, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_target, y_prob),
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist()
    }


def train_sequence_classifier(model, model_name, X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=0.001, device="cpu", model_path=None, tune_threshold=False, patience=10, **kwargs):

    """
    Generic trainer for PyTorch sequence classifiers with Early Stopping.
    """
    import copy
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    if kwargs.get('skip_training', False):
        print(f"Skipping training for {model_name}. Loading weights from {model_path}...")
        if model_path and torch.os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"WARNING: model_path {model_path} not found! Proceeding with untrained model.")
    else:
        print(f"Training {model_name} on {device}...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                
            epoch_loss /= len(train_loader.dataset)
            
            # Validation on test set (or validation set)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")
                
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                if model_path:
                    torch.save(model.state_dict(), model_path)
            else:
                epochs_no_improve += 1
                if patience is not None and epochs_no_improve >= patience:
                    print(f"  Early stopping triggered at epoch {epoch+1} (best loss: {best_loss:.5f})")
                    break
                
        model.load_state_dict(best_state_dict)
        if model_path and torch.os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
    best_threshold = 0.5
    if tune_threshold:
        best_threshold = find_optimal_threshold(model, X_val, y_val, batch_size=batch_size, device=device)
        if model_path:
            import json
            thresh_path = Path(model_path).with_suffix('.threshold.json')
            with open(thresh_path, 'w') as f:
                json.dump({'best_threshold': best_threshold}, f)
            print(f"  Saved validation-optimal threshold: {best_threshold:.6f} to {thresh_path.name}")
    else:
        # Clean up any leftover threshold JSON if it exists
        if model_path:
            thresh_path = Path(model_path).with_suffix('.threshold.json')
            if thresh_path.exists():
                thresh_path.unlink()
                
    eval_metrics = evaluate_model(model, X_val, y_val, batch_size=batch_size, device=device, threshold=best_threshold)
    eval_metrics['best_threshold'] = best_threshold
    return model, eval_metrics



def train_gru_classifier(X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=0.001, hidden_dim=64, num_layers=2, device="cpu", model_path=None, tune_threshold=False, skip_training=False):
    """
    Trains a GRU sequence classifier, saves best weights if model_path is provided, and returns metrics.
    """
    input_dim = X_train.shape[2]
    model = GRUClassifier(input_dim, hidden_dim, num_layers).to(device)
    return train_sequence_classifier(
        model, "GRU Classifier", X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device, model_path=model_path, tune_threshold=tune_threshold, skip_training=skip_training
    )

def train_tcn_classifier(X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=0.001, hidden_dim=64, device="cpu", model_path=None, tune_threshold=False, skip_training=False):
    """
    Trains a TCN sequence classifier, saves best weights if model_path is provided, and returns metrics.
    """
    input_dim = X_train.shape[2]
    model = TCNClassifier(input_dim, hidden_dim).to(device)
    return train_sequence_classifier(
        model, "TCN Classifier", X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device, model_path=model_path, tune_threshold=tune_threshold, skip_training=skip_training
    )

def train_cnn_gru_classifier(X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=0.001, hidden_dim=64, num_layers=2, device="cpu", model_path=None, tune_threshold=False, skip_training=False):
    """
    Trains a CNN-GRU sequence classifier, saves best weights if model_path is provided, and returns metrics.
    """
    input_dim = X_train.shape[2]
    model = CNNGRUClassifier(input_dim, hidden_dim, num_layers).to(device)
    return train_sequence_classifier(
        model, "CNN-GRU Classifier", X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device, model_path=model_path, tune_threshold=tune_threshold, skip_training=skip_training
    )

def train_cnn_classifier(X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=0.001, hidden_dim=64, device="cpu", model_path=None, tune_threshold=False, skip_training=False):
    """
    Trains a CNN sequence classifier, saves best weights if model_path is provided, and returns metrics.
    """
    input_dim = X_train.shape[2]
    model = CNNClassifier(input_dim, hidden_dim).to(device)
    return train_sequence_classifier(
        model, "CNN Classifier", X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device, model_path=model_path, tune_threshold=tune_threshold, skip_training=skip_training
    )



def plot_sequence_roc_curves(model_dict, X_test, y_test, plots_dir, device="cpu"):
    """
    Plots the ROC curves for all models in model_dict and saves the figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(8, 6))
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    for name, model in model_dict.items():
        model.eval()
        model.to(device)
        all_probs = []
        
        test_dataset = TensorDataset(X_test_t)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for batch in test_loader:
                batch_X = batch[0].to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                
        y_prob = np.concatenate(all_probs, axis=0)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Sequence Models')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / 'sequence_roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve plot to: {save_path}")

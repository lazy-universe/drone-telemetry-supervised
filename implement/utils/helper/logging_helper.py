import numpy as np
import pandas as pd

def log_dataset_statistics(X_train, y_train, X_test=None, y_test=None, X_val=None, y_val=None):
    """
    Backward-compatible wrapper to route to log_pointwise_dataset_statistics.
    """
    test_X = X_test if X_test is not None else X_val
    test_y = y_test if y_test is not None else y_val
    if test_X is None or test_y is None:
        print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
        return
    log_pointwise_dataset_statistics(X_train, y_train, test_X, test_y)

def log_pointwise_dataset_statistics(X_train, y_train, X_test, y_test):
    """
    Logs dataset statistics for point-wise (non-sequence) models.
    Supports train/test splits.
    """
    print("\n" + "="*60)
    print("=== POINTWISE DATASET STATISTICS (Train/Test) ===")
    print("="*60)
    
    def to_numpy_labels(y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            return y.to_numpy()
        return np.asarray(y)

    y_train_np = to_numpy_labels(y_train)
    y_test_np = to_numpy_labels(y_test)
    
    # Real is class 0 (DJI), Fake is class 1 (ESP32/Spoofed)
    train_real = int(np.sum(y_train_np == 0))
    train_fake = int(np.sum(y_train_np == 1))
    
    test_real = int(np.sum(y_test_np == 0))
    test_fake = int(np.sum(y_test_np == 1))
    
    total_real = train_real + test_real
    total_fake = train_fake + test_fake
    total_samples = total_real + total_fake
    
    print(f"Total Dataset Size: {total_samples} samples")
    if total_samples > 0:
        print(f"  - Total Real (DJI): {total_real} ({total_real/total_samples*100:.2f}%)")
        print(f"  - Total Fake (ESP32/Attack): {total_fake} ({total_fake/total_samples*100:.2f}%)")
    
    # Dimensions
    print(f"\nDimensions:")
    print(f"  - X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"  - X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")
        
    # Split Ratios
    n_train = len(y_train_np)
    n_test = len(y_test_np)
    n_total = n_train + n_test
    
    ratio_train = (n_train / n_total) * 100 if n_total > 0 else 0
    ratio_test = (n_test / n_total) * 100 if n_total > 0 else 0
    
    print(f"\nSplit Ratios (Train / Test):")
    print(f"  - {ratio_train:.1f}% / {ratio_test:.1f}%")
        
    # Breakdown of samples per split
    print(f"\nSamples Breakdown:")
    print(f"  - Train Split : {n_train} samples (Real: {train_real}, Fake: {train_fake})")
    print(f"  - Test Split  : {n_test} samples (Real: {test_real}, Fake: {test_fake})")
    print("="*60 + "\n")


def log_sequence_dataset_statistics(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Logs dataset statistics for sequence models (GRU, TCN, CNN-LSTM, Transformer).
    Supports train/val/test splits.
    """
    print("\n" + "="*60)
    print("=== SEQUENCE DATASET STATISTICS (Train/Val/Test) ===")
    print("="*60)
    
    def to_numpy_labels(y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            return y.to_numpy()
        return np.asarray(y)

    y_train_np = to_numpy_labels(y_train)
    y_val_np = to_numpy_labels(y_val)
    y_test_np = to_numpy_labels(y_test)
    
    # Real is class 0 (DJI), Fake is class 1 (ESP32/Spoofed)
    train_real = int(np.sum(y_train_np == 0))
    train_fake = int(np.sum(y_train_np == 1))
    
    val_real = int(np.sum(y_val_np == 0))
    val_fake = int(np.sum(y_val_np == 1))
    
    test_real = int(np.sum(y_test_np == 0))
    test_fake = int(np.sum(y_test_np == 1))
    
    total_real = train_real + val_real + test_real
    total_fake = train_fake + val_fake + test_fake
    total_samples = total_real + total_fake
    
    print(f"Total Dataset Size: {total_samples} samples")
    if total_samples > 0:
        print(f"  - Total Real (DJI): {total_real} ({total_real/total_samples*100:.2f}%)")
        print(f"  - Total Fake (ESP32/Attack): {total_fake} ({total_fake/total_samples*100:.2f}%)")
    
    # Dimensions
    print(f"\nDimensions:")
    print(f"  - X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"  - X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
    print(f"  - X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")
        
    # Split Ratios
    n_train = len(y_train_np)
    n_val = len(y_val_np)
    n_test = len(y_test_np)
    n_total = n_train + n_val + n_test
    
    ratio_train = (n_train / n_total) * 100 if n_total > 0 else 0
    ratio_val = (n_val / n_total) * 100 if n_total > 0 else 0
    ratio_test = (n_test / n_total) * 100 if n_total > 0 else 0
    
    print(f"\nSplit Ratios (Train / Val / Test):")
    print(f"  - {ratio_train:.1f}% / {ratio_val:.1f}% / {ratio_test:.1f}%")
        
    # Breakdown of samples per split
    print(f"\nSamples Breakdown:")
    print(f"  - Train Split : {n_train} samples (Real: {train_real}, Fake: {train_fake})")
    print(f"  - Val Split   : {n_val} samples (Real: {val_real}, Fake: {val_fake})")
    print(f"  - Test Split  : {n_test} samples (Real: {test_real}, Fake: {test_fake})")
    print("="*60 + "\n")

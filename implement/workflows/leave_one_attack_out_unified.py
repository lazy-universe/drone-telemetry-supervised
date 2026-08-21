import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc
)

from implement.utils.helper import (
    get_output_dir,
    get_or_preprocess_genuine_dji_flights,
    get_or_preprocess_hardware_spoofer,
    SUPERVISED_FEATURES,
    WINDOW_LEN
)
from implement.utils.dataset_processing.dataset_helper import impute_and_scale_data

# Imports for Point Models
from implement.utils.classical_ml.classical_models import get_point_classifiers

# Imports for Sequence Models
from implement.utils.deep_learning import (
    train_gru_classifier,
    train_tcn_classifier,
    train_cnn_gru_classifier,
    train_cnn_classifier,
    evaluate_model
)
from implement.utils.deep_learning.sequence_helper import generate_sequences_from_df


def set_global_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_leave_one_attack_out_workflow(
    model_type: str = 'point',
    epochs: int = 15,
    batch_size: int = 64,
    tune_threshold: bool = False,
    balanced: bool = False,
    use_smote: bool = False,
    random_state: int = 42,
    device: str = 'cuda',
    output_suffix: str = ""
):
    """
    Unified workflow for Leave-One-Attack-Out validation.
    Supports both classical point-based ('point') and sequence deep learning ('sequence') models.
    """
    set_global_seeds(random_state)
    output_dir = get_output_dir()
    subfolder = f"point_leave_one_attack_out{output_suffix}" if model_type == 'point' else f"sequence_leave_one_attack_out{output_suffix}"
    evaluation_dir = output_dir / subfolder
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print(f"=== WORKFLOW: LOAO TELEMETRY CLASSIFICATION ({model_type.upper()}) ===")
    print(f"=== Config: Random State={random_state} | SMOTE={use_smote} ===")
    if model_type == 'sequence':
        print(f"=== DL Config: Epochs={epochs} | Batch={batch_size} | Device={device} ===")
    print(f"=== Output Directory: {evaluation_dir} ===")
    print("=====================================================================")

    # 1. Load datasets
    print("\n[Step 1/4] Loading and preprocessing datasets...")
    # Point models don't filter length < 100, sequence models must to build windows of 20 samples
    filter_100 = (model_type == 'sequence')
    dji_df = get_or_preprocess_genuine_dji_flights(filter_length_100=filter_100, features=SUPERVISED_FEATURES)
    esp32_df = get_or_preprocess_hardware_spoofer()

    # Get attack groupings
    all_flights = esp32_df['flight_id'].unique()
    sim_flights = [f for f in all_flights if f != 'esp32_flight']
    categories = ['easy', 'medium', 'hard', 'baseline', 'geometry']
    attack_sources = {}
    for cat in categories:
        attack_sources[cat] = [f for f in sim_flights if f.startswith(f"sim_{cat}")]
    attack_sources['real_esp32'] = ['esp32_flight']

    loao_results = []

    # 2. Execute LOAO iterations
    for left_out_name, left_out_flights in attack_sources.items():
        print(f"\n--- LOAO Iteration: Leaving Out '{left_out_name.upper()}' ---")
        print(f"  Left-out flights: {left_out_flights}")

        # Split Esp32/Simulated data
        esp32_train_raw = esp32_df[~esp32_df['flight_id'].isin(left_out_flights)]
        esp32_test_raw = esp32_df[esp32_df['flight_id'].isin(left_out_flights)]

        if model_type == 'point':
            # Flight-wise DJI split (80% train, 20% test)
            unique_dji_flights = dji_df['flight_id'].unique()
            np.random.seed(random_state)
            shuffled_dji_flights = np.random.permutation(unique_dji_flights)
            split_idx = int(len(shuffled_dji_flights) * 0.8)
            dji_train_flights = shuffled_dji_flights[:split_idx]
            dji_test_flights = shuffled_dji_flights[split_idx:]

            dji_train = dji_df[dji_df['flight_id'].isin(dji_train_flights)].drop(columns=['flight_id'], errors='ignore')
            dji_test = dji_df[dji_df['flight_id'].isin(dji_test_flights)].drop(columns=['flight_id'], errors='ignore')

            train_df = pd.concat([dji_train, esp32_train_raw.drop(columns=['flight_id'], errors='ignore')], ignore_index=True)
            test_df = pd.concat([dji_test, esp32_test_raw.drop(columns=['flight_id'], errors='ignore')], ignore_index=True)

            # Shuffle
            train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            X_train_raw, y_train = train_df[SUPERVISED_FEATURES], train_df['nature'].to_numpy().astype(int)
            X_test_raw, y_test = test_df[SUPERVISED_FEATURES], test_df['nature'].to_numpy().astype(int)

            X_train_scaled, X_test_scaled, _, _ = impute_and_scale_data(X_train_raw, X_test_raw)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

            # Train and evaluate classical point models
            models = get_point_classifiers(use_class_weights=balanced, use_smote=use_smote, random_state=random_state)
            iteration_metrics = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
                prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
                
                iteration_metrics[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_prob),
                    'pr_auc': auc(rec_vals, prec_vals)
                }

        else:
            # Sequence split (70% train, 10% val, 20% test)
            unique_dji_flights = dji_df['flight_id'].unique()
            np.random.seed(random_state)
            shuffled_dji_flights = np.random.permutation(unique_dji_flights)

            split_idx_test = int(len(shuffled_dji_flights) * 0.2)
            dji_test_flights = shuffled_dji_flights[:split_idx_test]
            dji_temp_flights = shuffled_dji_flights[split_idx_test:]

            split_idx_val = int(len(dji_temp_flights) * 0.1)
            dji_val_flights = dji_temp_flights[:split_idx_val]
            dji_train_flights = dji_temp_flights[split_idx_val:]

            # Unique ESP32 trains
            unique_esp32_flights = esp32_train_raw['flight_id'].unique()
            shuffled_esp32 = np.random.permutation(unique_esp32_flights)
            split_esp32_val = int(len(shuffled_esp32) * 0.1)
            esp32_val_flights = shuffled_esp32[:split_esp32_val]
            esp32_train_flights = shuffled_esp32[split_esp32_val:]

            # Sequence generation
            def get_seqs(df, fids):
                windows = []
                for fid in fids:
                    w = generate_sequences_from_df(df[df['flight_id'] == fid], SUPERVISED_FEATURES, WINDOW_LEN)
                    if len(w) > 0:
                        windows.append(w)
                return np.concatenate(windows, axis=0) if windows else np.empty((0, WINDOW_LEN, len(SUPERVISED_FEATURES)))

            X_dji_train = get_seqs(dji_df, dji_train_flights)
            X_dji_val = get_seqs(dji_df, dji_val_flights)
            X_dji_test = get_seqs(dji_df, dji_test_flights)

            X_esp_train = get_seqs(esp32_train_raw, esp32_train_flights)
            X_esp_val = get_seqs(esp32_train_raw, esp32_val_flights)
            X_esp_test = get_seqs(esp32_test_raw, left_out_flights)

            # Combine splits
            X_train_unscaled = np.concatenate([X_dji_train, X_esp_train], axis=0)
            y_train = np.concatenate([np.zeros(len(X_dji_train)), np.ones(len(X_esp_train))])

            X_val_unscaled = np.concatenate([X_dji_val, X_esp_val], axis=0)
            y_val = np.concatenate([np.zeros(len(X_dji_val)), np.ones(len(X_esp_val))])

            X_test_unscaled = np.concatenate([X_dji_test, X_esp_test], axis=0)
            y_test = np.concatenate([np.zeros(len(X_dji_test)), np.ones(len(X_esp_test))])

            # Standardise
            n_tr, step, n_feat = X_train_unscaled.shape
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_unscaled.reshape(-1, n_feat)).reshape(n_tr, step, n_feat)
            X_val_scaled = scaler.transform(X_val_unscaled.reshape(-1, n_feat)).reshape(X_val_unscaled.shape[0], step, n_feat)
            X_test_scaled = scaler.transform(X_test_unscaled.reshape(-1, n_feat)).reshape(X_test_unscaled.shape[0], step, n_feat)

            # Shuffle train/val
            np.random.seed(random_state)
            tr_shuf = np.random.permutation(len(X_train_scaled))
            X_train_scaled, y_train = X_train_scaled[tr_shuf], y_train[tr_shuf]

            # Model suite
            dl_models = {
                'GRU': train_gru_classifier,
                'TCN': train_tcn_classifier,
                'CNN-GRU': train_cnn_gru_classifier,
                'CNN': train_cnn_classifier,
            }

            iteration_metrics = {}
            for name, train_fn in dl_models.items():
                model, _ = train_fn(X_train_scaled, y_train, X_val_scaled, y_val, epochs=epochs, batch_size=batch_size, device=device)
                m = evaluate_model(model, X_test_scaled, y_test, batch_size=batch_size, device=device)
                iteration_metrics[name] = m

        loao_results.append({
            'Left Out': left_out_name,
            'metrics': iteration_metrics
        })

    # Output formatting
    all_names = sorted(list(loao_results[0]['metrics'].keys()))
    col_w = 22
    print("\n" + "="*(32 + len(all_names)*col_w))
    print("=== LEAVE-ONE-ATTACK-OUT FINAL SUMMARY TABLE (F1-Scores) ===")
    print(f"{'Left-Out Attack Category':<30}" + "".join([f"{name:<{col_w}}" for name in all_names]))
    print("-"*(32 + len(all_names)*col_w))
    for row in loao_results:
        row_str = f"{row['Left Out'].upper():<30}"
        for name in all_names:
            f1 = f"{row['metrics'].get(name, {}).get('f1_score', 0.0)*100:.2f}%"
            row_str += f"{f1:<{col_w}}"
        print(row_str)
    print("="*(32 + len(all_names)*col_w) + "\n")

    joblib.dump(loao_results, evaluation_dir / "loao_results.joblib")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['point', 'sequence'], default='point')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_leave_one_attack_out_workflow(model_type=args.type, random_state=args.seed)

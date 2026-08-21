import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

from implement.utils.helper import (
    get_output_dir,
    get_or_preprocess_genuine_dji_flights,
    get_genuine_dji_flights_with_device_split,
    get_or_preprocess_hardware_spoofer,
    SUPERVISED_FEATURES,
    WINDOW_LEN,
    log_pointwise_dataset_statistics,
    log_sequence_dataset_statistics,
)
from implement.utils.dataset_processing.dataset_helper import combine_and_split_supervised_data
from implement.utils.classical_ml.classical_models import get_point_classifiers
from implement.utils.classical_ml.classical_train_eval import plot_roc_curves
from implement.utils.deep_learning import (
    prepare_sequence_dataset,
    train_gru_classifier,
    train_tcn_classifier,
    train_cnn_gru_classifier,
    train_cnn_classifier,
    evaluate_model,
)


CLASSICAL_MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "Threshold(PE)"]
SEQUENCE_MODELS = ["CNN", "GRU", "CNN-GRU", "TCN"]
SPLIT_MODES = ["flight", "device", "random"]


def _format_metric_row(metrics):
    return (
        f"{metrics['accuracy']*100:.2f}%",
        f"{metrics['precision']*100:.2f}%",
        f"{metrics['recall']*100:.2f}%",
        f"{metrics['recall_real']*100:.2f}%",
        f"{metrics['recall_spoof']*100:.2f}%",
        f"{metrics['fp_count']}",
        f"{metrics['tp_count']}",
        f"{metrics['f1_score']*100:.2f}%",
        f"{metrics['roc_auc']*100:.2f}%",
        f"{metrics['pr_auc']*100:.2f}%",
    )


def _classification_metrics(y_true, y_pred, y_prob):
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "recall_real": recall_0,
        "recall_spoof": recall_1,
        "fp_count": int(fp),
        "tp_count": int(tp),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": auc(rec_vals, prec_vals),
        "confusion_matrix": cm.tolist(),
    }


RAW_FEATURES = ['height', 'ground_speed', 'vertical_speed', 'course']

def _save_pointwise_artifacts(models, results, X_test, y_test, plots_dir, suppress, random_state, features):
    if suppress:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(models, X_test, y_test, plots_dir)

    try:
        import shap
        import matplotlib.pyplot as plt
        import warnings
        
        rng = np.random.default_rng(random_state)
        
        for shap_model_name in ["Random Forest", "XGBoost"]:
            if shap_model_name not in models:
                continue
            print(f"\n[SHAP Analysis] Generating SHAP explanations using '{shap_model_name}'...")
            sample_size = min(200, X_test.shape[0])
            indices = rng.choice(X_test.shape[0], sample_size, replace=False)
            X_sample = X_test[indices]
            explainer_model = models[shap_model_name]
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                explainer = shap.TreeExplainer(explainer_model)
                shap_values = explainer.shap_values(X_sample, check_additivity=False)
                
            if isinstance(shap_values, list):
                shap_val_to_plot = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_val_to_plot = shap_values[:, :, 1]
            else:
                shap_val_to_plot = shap_values
            plt.figure(figsize=(10, 6))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                shap.summary_plot(shap_val_to_plot, X_sample, feature_names=features, show=False)
                
            plt.title(f"SHAP Feature Importance: {shap_model_name}", fontsize=14, pad=15)
            plt.tight_layout()
            shap_plot_path = plots_dir / f"shap_summary_{shap_model_name.replace(' ', '_')}.png"
            plt.savefig(shap_plot_path, dpi=300)
            plt.close()
            print(f"✓ Saved {shap_model_name} SHAP summary plot to: {shap_plot_path}")
    except Exception as e:
        print(f"[SHAP Error] Could not generate SHAP plots: {e}")

    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        print("\n=== Generating Confusion Matrix Plots ===")
        for name, metrics in results.items():
            cm = np.array(metrics['confusion_matrix'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real DJI (0)', 'Spoofed (1)'])
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
            plt.title(f"Confusion Matrix: {name}", fontsize=14, pad=15)
            plt.tight_layout()
            cm_plot_path = plots_dir / f"confusion_matrix_{name.replace(' ', '_')}.png"
            plt.savefig(cm_plot_path, dpi=300)
            plt.close()
            print(f"✓ Saved confusion matrix plot for {name} to: {cm_plot_path}")
        print("=========================================\n")
    except Exception as e:
        print(f"[Confusion Matrix Plot Error] Could not generate confusion matrix plots: {e}")


def _run_point_models(dji_df, esp32_df, split_mode, random_state, features, tune=False, seq_val_data=None):
    X_train, y_train, X_test, y_test, _, _ = combine_and_split_supervised_data(
        dji_df,
        esp32_df,
        train_ratio=0.8,
        split_mode=split_mode,
        intermediate_dir=None,
        output_dir=None,
        features=features,
        random_state=random_state,
    )
    models = get_point_classifiers(random_state=random_state)
    results = {}
    trained_models = {}
    
    if tune and seq_val_data is not None:
        seq_X_val, seq_y_val = seq_val_data
        X_val = seq_X_val.reshape(-1, seq_X_val.shape[2])
        y_val = np.repeat(seq_y_val, seq_X_val.shape[1])
        X_train_fit, y_train_fit = X_train, y_train
    elif tune:
        from sklearn.model_selection import train_test_split
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.20, random_state=random_state, stratify=y_train
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    for name in CLASSICAL_MODELS:
        print(f"  Training pointwise model: {name}")
        model = models[name]
        model.fit(X_train_fit, y_train_fit)
        
        best_threshold = 0.5
        if tune:
            from sklearn.metrics import f1_score
            y_prob_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
            best_f1 = 0.0
            for thresh in np.arange(0.01, 1.0, 0.01):
                preds = (y_prob_val >= thresh).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_threshold = thresh
            print(f"  [Tuning] Best threshold for {name}: {best_threshold:.2f} (Val F1: {best_f1*100:.2f}%)")
            
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        y_pred = (y_prob >= best_threshold).astype(int)
        results[name] = _classification_metrics(y_test, y_pred, y_prob)
        trained_models[name] = model
    return results, trained_models, (X_train, y_train, X_test, y_test)


def _prepare_sequence_data(dji_df, esp32_df, split_mode, random_state, features):
    import implement.utils.helper.features as feat
    return prepare_sequence_dataset(
        dji_df,
        esp32_df,
        window_len=feat.WINDOW_LEN,
        random_state=random_state,
        features=features,
        split_mode=split_mode,
    )


def _run_sequence_models(seq_data, epochs, batch_size, tune_threshold, device, baseline_dir=None):
    X_train, y_train, X_val, y_val, X_test, y_test, *rest = seq_data
    model_specs = {
        "CNN": (train_cnn_classifier, {"hidden_dim": 64, "device": device}),
        "GRU": (train_gru_classifier, {"hidden_dim": 64, "num_layers": 2, "device": device}),
        "CNN-GRU": (train_cnn_gru_classifier, {"hidden_dim": 64, "num_layers": 2, "device": device}),
        "TCN": (train_tcn_classifier, {"hidden_dim": 64, "device": device}),
    }
    results = {}
    trained_models = {}
    thresholds = {}
    for name in SEQUENCE_MODELS:
        print(f"  Training sequence model: {name}")
        train_fn, kwargs = model_specs[name]
        
        if tune_threshold and baseline_dir is not None:
            clean_name = name.lower().replace("-", "_")
            model_path = baseline_dir / "models" / f"{clean_name}_classifier.pth"
            kwargs["skip_training"] = True
            kwargs["model_path"] = model_path
            
        model, val_metrics = train_fn(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            tune_threshold=tune_threshold,
            **kwargs,
        )
        threshold = val_metrics.get("best_threshold", 0.5) if tune_threshold else 0.5
        results[name] = evaluate_model(model, X_test, y_test, batch_size=batch_size, device=device, threshold=threshold)
        trained_models[name] = model
        thresholds[name] = threshold
    return results, trained_models, thresholds


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


def run_unified_supervised_workflow(epochs=15, batch_size=64, tune_threshold=False, suppress=False, random_state=42, split_mode="both", model_scope="all", feature_mode="engineered", tune=False, output_suffix="", cv=False):
    set_global_seeds(random_state)
    output_dir = get_output_dir() / feature_mode
    tune_prefix = "_tuned" if tune else ""
    evaluation_dir = output_dir / f"unified_supervised{tune_prefix}{output_suffix}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=====================================================================")
    print("=== WORKFLOW: UNIFIED SUPERVISED CLASSIFICATION ===")
    print(f"=== Epochs={epochs} | Batch={batch_size} | Feature Mode={feature_mode} | Device={device} ===")
    if cv:
        print("=== MODE: STRATIFIED FLIGHT CROSS-VALIDATION ===")
    print(f"=== Output Directory: {evaluation_dir} ===")
    print("=====================================================================")

    features = RAW_FEATURES if feature_mode == "raw" else SUPERVISED_FEATURES

    dji_point = get_or_preprocess_genuine_dji_flights(filter_length_100=False, features=features)
    dji_seq = get_or_preprocess_genuine_dji_flights(filter_length_100=True, features=features)
    dji_device_point = get_genuine_dji_flights_with_device_split(filter_length_100=False)
    dji_device_seq = get_genuine_dji_flights_with_device_split(filter_length_100=True)
    esp32_df = get_or_preprocess_hardware_spoofer()

    summary = {}
    split_modes = SPLIT_MODES if split_mode == "all" else [split_mode]
    run_point = model_scope in ("all", "point")
    run_sequence = model_scope in ("all", "sequence")

    if cv:
        # Perform Stratified Flight Cross-Validation (split_mode is strictly 'flight' inside cv)
        print("[Cross-Validation] Initializing 5-Fold Stratified Flight Cross-Validation...")
        
        # 1. Gather all flight IDs and group them by class prefix to maintain stratification
        # DJI flight IDs
        dji_flights = sorted(list(dji_point['flight_id'].unique()))
        
        # Simulated spoofed flight IDs grouped by prefix
        all_esp32_flights = esp32_df['flight_id'].unique()
        sim_flights = sorted([f for f in all_esp32_flights if f != 'esp32_flight'])
        sim_groups = {}
        for f in sim_flights:
            prefix = f.split('_flight_')[0] if '_flight_' in f else f.split('_')[0]
            sim_groups.setdefault(prefix, []).append(f)
            
        # Real esp32 flight
        esp32_real = esp32_df[esp32_df['flight_id'] == 'esp32_flight']
        
        # 2. Divide each group's unique flight IDs into 5 folds
        np.random.seed(random_state)
        n_folds = 5
        
        def split_into_k_folds(items, k):
            shuffled = list(items)
            np.random.shuffle(shuffled)
            folds = [[] for _ in range(k)]
            for i, item in enumerate(shuffled):
                folds[i % k].append(item)
            return folds
            
        dji_folds = split_into_k_folds(dji_flights, n_folds)
        sim_prefix_folds = {prefix: split_into_k_folds(flights, n_folds) for prefix, flights in sim_groups.items()}
        
        cv_metrics = {}
        ordered_models = []
        if run_point:
            ordered_models.extend(CLASSICAL_MODELS)
        if run_sequence:
            ordered_models.extend(SEQUENCE_MODELS)
            
        for model_name in ordered_models:
            cv_metrics[model_name] = {
                "accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": [], "pr_auc": []
            }
            
        for fold in range(n_folds):
            print(f"\n--- Running Fold {fold+1}/{n_folds} ---")
            
            # Select train and validation flight IDs for this fold
            val_dji_flights = dji_folds[fold]
            train_dji_flights = [f for f_list in dji_folds[:fold] + dji_folds[fold+1:] for f in f_list]
            
            val_sim_flights = []
            train_sim_flights = []
            for prefix, folds_list in sim_prefix_folds.items():
                val_sim_flights.extend(folds_list[fold])
                train_sim_flights.extend([f for f_list in folds_list[:fold] + folds_list[fold+1:] for f in f_list])
                
            # Filter DJI Dataframes
            dji_train_pt = dji_point[dji_point['flight_id'].isin(train_dji_flights)].drop(columns=['flight_id'], errors='ignore')
            dji_val_pt = dji_point[dji_point['flight_id'].isin(val_dji_flights)].drop(columns=['flight_id'], errors='ignore')
            
            dji_train_seq = dji_seq[dji_seq['flight_id'].isin(train_dji_flights)]
            dji_val_seq = dji_seq[dji_seq['flight_id'].isin(val_dji_flights)]
            
            # Filter Spoofed Dataframes
            esp32_train_pt = esp32_df[esp32_df['flight_id'].isin(train_sim_flights)].drop(columns=['flight_id'], errors='ignore')
            
            # Validation contains the real ESP32 flight (capped to 1500) and the selected simulated val flights
            if len(esp32_real) > 1500:
                # Cap the real esp32 samples to 1500 for validation fold
                esp32_real_val = esp32_real.sample(n=1500, random_state=random_state).drop(columns=['flight_id'], errors='ignore')
            else:
                esp32_real_val = esp32_real.drop(columns=['flight_id'], errors='ignore')
                
            esp32_sim_val_pt = esp32_df[esp32_df['flight_id'].isin(val_sim_flights)].drop(columns=['flight_id'], errors='ignore')
            esp32_val_pt = pd.concat([esp32_real_val, esp32_sim_val_pt], ignore_index=True)
            
            # Combine pointwise train and val datasets
            train_df_pt = pd.concat([dji_train_pt, esp32_train_pt], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
            val_df_pt = pd.concat([dji_val_pt, esp32_val_pt], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            X_train_pt = train_df_pt[features]
            y_train_pt = train_df_pt['nature'].to_numpy().astype(int)
            X_val_pt = val_df_pt[features]
            y_val_pt = val_df_pt['nature'].to_numpy().astype(int)
            
            # Scale and impute pointwise datasets
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            imputer = SimpleImputer(strategy="mean")
            scaler = StandardScaler()
            X_train_pt_scaled = scaler.fit_transform(X_train_pt)
            X_val_pt_scaled = scaler.transform(X_val_pt)
            
            if run_point:
                print(f"  [Pointwise Data] Train shape: {X_train_pt_scaled.shape} | Val shape: {X_val_pt_scaled.shape}")
                pt_models = get_point_classifiers(random_state=random_state)
                if tune:
                    # Carve out validation from training for pointwise tuning if tune is enabled
                    from sklearn.model_selection import train_test_split
                    X_fit, X_tune, y_fit, y_tune = train_test_split(
                        X_train_pt_scaled, y_train_pt, test_size=0.20, random_state=random_state, stratify=y_train_pt
                    )
                else:
                    X_fit, y_fit = X_train_pt_scaled, y_train_pt
                    
                for name in CLASSICAL_MODELS:
                    model = pt_models[name]
                    model.fit(X_fit, y_fit)
                    best_threshold = 0.5
                    if tune:
                        from sklearn.metrics import f1_score
                        y_prob_tune = model.predict_proba(X_tune)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_tune)
                        best_f1 = 0.0
                        for thresh in np.arange(0.01, 1.0, 0.01):
                            preds = (y_prob_tune >= thresh).astype(int)
                            score = f1_score(y_tune, preds, zero_division=0)
                            if score > best_f1:
                                best_f1 = score
                                best_threshold = thresh
                                
                    y_prob = model.predict_proba(X_val_pt_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val_pt_scaled)
                    y_pred = (y_prob >= best_threshold).astype(int)
                    fold_res = _classification_metrics(y_val_pt, y_pred, y_prob)
                    for metric in cv_metrics[name]:
                        cv_metrics[name][metric].append(fold_res[metric])
                        
            if run_sequence:
                # Prepare sequence datasets for this fold
                from implement.utils.deep_learning.sequence_helper import generate_sequences_from_df
                import implement.utils.helper.features as feat
                
                # Training sequences (non-overlapping step=WINDOW_LEN to avoid overlap leak, or continuous step=1)
                # Let's generate flight-by-flight train sequences
                train_dji_seqs = [generate_sequences_from_df(dji_seq[dji_seq['flight_id'] == f], features, feat.WINDOW_LEN, step=1) for f in train_dji_flights]
                train_sim_seqs = [generate_sequences_from_df(esp32_df[esp32_df['flight_id'] == f], features, feat.WINDOW_LEN, step=1) for f in train_sim_flights]
                
                X_dji_tr_seq = np.concatenate(train_dji_seqs, axis=0) if train_dji_seqs else np.empty((0, feat.WINDOW_LEN, len(features)))
                X_sim_tr_seq = np.concatenate(train_sim_seqs, axis=0) if train_sim_seqs else np.empty((0, feat.WINDOW_LEN, len(features)))
                
                # Val sequences
                val_dji_seqs = [generate_sequences_from_df(dji_seq[dji_seq['flight_id'] == f], features, feat.WINDOW_LEN, step=1) for f in val_dji_flights]
                val_sim_seqs = [generate_sequences_from_df(esp32_df[esp32_df['flight_id'] == f], features, feat.WINDOW_LEN, step=1) for f in val_sim_flights]
                
                # Real esp32 flight in validation
                esp32_real_seq_all = generate_sequences_from_df(esp32_df[esp32_df['flight_id'] == 'esp32_flight'], features, feat.WINDOW_LEN, step=feat.WINDOW_LEN)
                if len(esp32_real_seq_all) > 1500:
                    rng = np.random.default_rng(random_state)
                    indices = rng.choice(len(esp32_real_seq_all), size=1500, replace=False)
                    esp32_real_seq_val = esp32_real_seq_all[indices]
                else:
                    esp32_real_seq_val = esp32_real_seq_all
                    
                X_dji_val_seq = np.concatenate(val_dji_seqs, axis=0) if val_dji_seqs else np.empty((0, feat.WINDOW_LEN, len(features)))
                X_sim_val_seq = np.concatenate(val_sim_seqs, axis=0) if val_sim_seqs else np.empty((0, feat.WINDOW_LEN, len(features)))
                X_esp32_val_seq = np.concatenate([esp32_real_seq_val, X_sim_val_seq], axis=0) if len(esp32_real_seq_val) > 0 else X_sim_val_seq
                
                X_train_seq_unscaled = np.concatenate([X_dji_tr_seq, X_sim_tr_seq], axis=0)
                y_train_seq = np.concatenate([np.zeros(len(X_dji_tr_seq)), np.ones(len(X_sim_tr_seq))])
                
                X_val_seq_unscaled = np.concatenate([X_dji_val_seq, X_esp32_val_seq], axis=0)
                y_val_seq = np.concatenate([np.zeros(len(X_dji_val_seq)), np.ones(len(X_esp32_val_seq))])
                
                # Scale sequences using training parameters
                n_train, timesteps, n_features = X_train_seq_unscaled.shape
                n_val = X_val_seq_unscaled.shape[0]
                
                X_train_2d = X_train_seq_unscaled.reshape(-1, n_features)
                X_val_2d = X_val_seq_unscaled.reshape(-1, n_features)
                
                seq_scaler = StandardScaler()
                X_train_scaled_2d = seq_scaler.fit_transform(X_train_2d)
                X_val_scaled_2d = seq_scaler.transform(X_val_2d)
                
                X_train_seq = X_train_scaled_2d.reshape(n_train, timesteps, n_features)
                X_val_seq = X_val_scaled_2d.reshape(n_val, timesteps, n_features)
                
                X_train_seq = np.nan_to_num(X_train_seq, nan=0.0)
                X_val_seq = np.nan_to_num(X_val_seq, nan=0.0)
                
                # Shuffle
                shuffle_tr = np.random.permutation(len(X_train_seq))
                X_train_seq, y_train_seq = X_train_seq[shuffle_tr], y_train_seq[shuffle_tr]
                shuffle_val = np.random.permutation(len(X_val_seq))
                X_val_seq, y_val_seq = X_val_seq[shuffle_val], y_val_seq[shuffle_val]
                
                # Use standard training epochs for cross-validation consistency
                cv_epochs = epochs
                print(f"  [Notice] Running cross-validation for sequence models. Using epochs: {cv_epochs} for fold.")
                print(f"  [Sequence Data] Train shape: {X_train_seq.shape} | Val shape: {X_val_seq.shape}")
                
                # Train and evaluate deep learning models
                seq_data = (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_val_seq, y_val_seq, seq_scaler)
                # For CV, we don't skip training because we need to tune folds independently. 
                # The tune logic for CV handles its own validation.
                fold_seq_results, _, _ = _run_sequence_models(seq_data, cv_epochs, batch_size, tune_threshold or tune, device, baseline_dir=None)
                for name in SEQUENCE_MODELS:
                    fold_res = fold_seq_results[name]
                    for metric in cv_metrics[name]:
                        cv_metrics[name][metric].append(fold_res[metric])

        # Calculate final summary
        print(f"\n=======================================================")
        print("=== CROSS-VALIDATION SUMMARY STATISTICS ===")
        print(f"=======================================================")
        print(f"{'Model':<22}{'Accuracy (Mean ± Std)':<28}{'F1-Score (Mean ± Std)':<28}{'ROC-AUC (Mean ± Std)':<28}")
        print('-' * 90)
        
        cv_final_summary = {}
        for model_name in ordered_models:
            accs = np.array(cv_metrics[model_name]["accuracy"]) * 100
            f1s = np.array(cv_metrics[model_name]["f1_score"]) * 100
            aucs = np.array(cv_metrics[model_name]["roc_auc"]) * 100
            
            acc_str = f"{accs.mean():.2f}% ± {accs.std():.2f}%"
            f1_str = f"{f1s.mean():.2f}% ± {f1s.std():.2f}%"
            auc_str = f"{aucs.mean():.2f}% ± {aucs.std():.2f}%"
            print(f"{model_name:<22}{acc_str:<28}{f1_str:<28}{auc_str:<28}")
            
            cv_final_summary[model_name] = {
                metric: {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "raw": [float(v) for v in vals]
                }
                for metric, vals in cv_metrics[model_name].items()
            }
            
        result_path = evaluation_dir / "cv_results.joblib"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cv_final_summary, result_path)
        print(f"\nSaved cross-validation results to: {result_path}")
        print("=== UNIFIED SUPERVISED CLASSIFICATION CV COMPLETE ===")
        return cv_final_summary

    # Baseline directory for loading models if tuning
    baseline_output_dir = output_dir / f"unified_supervised{output_suffix}" if tune else None

    # --- Baseline Non-CV Flow ---
    for current_split_mode in split_modes:
        print(f"\n=== Running {current_split_mode.upper()} split ===")
        point_dji = dji_device_point if current_split_mode == "device" else dji_point
        seq_dji = dji_device_seq if current_split_mode == "device" else dji_seq
        split_summary = {}
        
        seq_data = None
        if run_sequence or tune:
            seq_data = _prepare_sequence_data(seq_dji, esp32_df, current_split_mode, random_state, features)
            seq_X_train, seq_y_train, seq_X_val, seq_y_val, seq_X_test, seq_y_test, seq_scaler, seq_flight_ids_test = seq_data

        if run_point:
            point_X_train, point_y_train, point_X_test, point_y_test, scaler, imputer = combine_and_split_supervised_data(
                point_dji,
                esp32_df,
                train_ratio=0.8,
                split_mode=current_split_mode,
                intermediate_dir=None,
                output_dir=None,
                features=features,
                random_state=random_state,
            )
            log_pointwise_dataset_statistics(point_X_train, point_y_train, point_X_test, point_y_test)
            
            seq_val_data = (seq_X_val, seq_y_val) if tune and seq_data else None
            point_results, point_models, _ = _run_point_models(point_dji, esp32_df, current_split_mode, random_state, features, tune=tune, seq_val_data=seq_val_data)
            split_summary.update(point_results)
            
            # Save pointwise model artifacts for downstream analysis presets
            point_dir = output_dir / f"point_baseline_{current_split_mode}{output_suffix}"
            models_dir = point_dir / "models"
            dataset_dir = point_dir / "dataset"
            models_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            if scaler is not None:
                joblib.dump(scaler, models_dir / "scaler.joblib")
            if imputer is not None:
                joblib.dump(imputer, models_dir / "imputer.joblib")
            for name, model in point_models.items():
                clean_name = name.replace(" (No Class Weights)", "").replace(" ", "_")
                joblib.dump(model, models_dir / f"{clean_name}_Classifier.joblib")
            
            test_df = pd.DataFrame(point_X_test, columns=features)
            test_df["nature"] = point_y_test
            test_df.to_csv(dataset_dir / "test_dataset.csv", index=False)
        else:
            point_X_test = point_y_test = None
            point_models = {}
            point_results = {}

        if run_sequence:
            # seq_data is already prepared above
            log_sequence_dataset_statistics(seq_X_train, seq_y_train, seq_X_val, seq_y_val, seq_X_test, seq_y_test)
            
            baseline_dir = output_dir / f"sequence_baseline_{current_split_mode}{output_suffix}" if tune else None
            
            seq_results, seq_models, seq_thresholds = _run_sequence_models(seq_data, epochs, batch_size, tune_threshold or tune, device, baseline_dir=baseline_dir)
            split_summary.update(seq_results)
            
            # Save sequence model artifacts for downstream analysis presets
            prefix = "sequence_tuned" if (tune_threshold or tune) else "sequence_baseline"
            seq_dir = output_dir / f"{prefix}_{current_split_mode}{output_suffix}"
            models_dir = seq_dir / "models"
            dataset_dir = seq_dir / "dataset"
            models_dir.mkdir(parents=True, exist_ok=True)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(dataset_dir / "X_test.npy", seq_X_test)
            np.save(dataset_dir / "y_test.npy", seq_y_test)
            np.save(dataset_dir / "flight_ids_test.npy", seq_flight_ids_test)
            
            import json
            for name, model in seq_models.items():
                clean_name = name.lower().replace("-", "_")
                torch.save(model.state_dict(), models_dir / f"{clean_name}_classifier.pth")
                with open(models_dir / f"{clean_name}_classifier.threshold.json", "w") as f:
                    json.dump({"best_threshold": seq_thresholds[name]}, f)
        else:
            seq_X_test = seq_y_test = None
            seq_models = {}
            seq_results = {}

        summary[current_split_mode] = split_summary

        if not suppress:
            if run_point:
                point_plots_dir = evaluation_dir / f"{current_split_mode}_point" / "plots"
                _save_pointwise_artifacts(point_models, point_results, point_X_test, point_y_test, point_plots_dir, suppress, random_state, features)
            if run_sequence:
                seq_plots_dir = evaluation_dir / f"{current_split_mode}_sequence" / "plots"
                from implement.utils.deep_learning.dl_train_eval import plot_sequence_roc_curves
                plot_sequence_roc_curves(seq_models, seq_X_test, seq_y_test, seq_plots_dir, device=device)

        print(f"\n=== FINAL SUMMARY: {current_split_mode.upper()} SPLIT ===")
        print(f"{'Model':<22}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'Rec(Real)':<10}{'Rec(Spf)':<10}{'FP':<6}{'TP':<6}{'F1':<10}{'ROC-AUC':<10}{'PR-AUC':<10}")
        print('-' * 114)
        ordered_models = []
        if run_point:
            ordered_models.extend(CLASSICAL_MODELS)
        if run_sequence:
            ordered_models.extend(SEQUENCE_MODELS)
        for model_name in ordered_models:
            metrics = summary[current_split_mode][model_name]
            acc, prec, rec, r_real, r_spf, fp, tp, f1s, roc_auc, pr_auc = _format_metric_row(metrics)
            print(f"{model_name:<22}{acc:<10}{prec:<10}{rec:<10}{r_real:<10}{r_spf:<10}{fp:<6}{tp:<6}{f1s:<10}{roc_auc:<10}{pr_auc:<10}")

        print("\n=== Confusion Matrices ===")
        for model_name in ordered_models:
            metrics = summary[current_split_mode][model_name]
            print(f"{model_name}:")
            print(np.array(metrics['confusion_matrix']))
            print()
    result_path = evaluation_dir / "unified_results.joblib"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(summary, result_path)
    print("\n=== UNIFIED SUPERVISED CLASSIFICATION COMPLETE ===")
    return summary


if __name__ == "__main__":
    run_unified_supervised_workflow()

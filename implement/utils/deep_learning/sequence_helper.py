import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from implement.utils.dataset_processing.dataset_helper import INTERSECTING_FEATURES
from implement.utils.helper import WINDOW_LEN

def generate_sequences_from_numpy(data, window_len=WINDOW_LEN, step=1):
    """
    Generates window sequences of shape (num_windows, window_len, num_features)
    from a 2D numpy array using a specified step size.
    """
    n_rows = len(data)
    if n_rows < window_len:
        return np.empty((0, window_len, data.shape[1]))
    
    indices = np.arange(0, n_rows - window_len + 1, step)
    return np.array([data[i:i+window_len] for i in indices])

def generate_sequences_from_df(df, features, window_len=WINDOW_LEN, step=1):
    """
    Generates window sequences of shape (num_windows, window_len, num_features)
    from a continuous dataframe.
    """
    df_clean = df[features].ffill().bfill().fillna(0.0)
    data = df_clean.to_numpy()
    return generate_sequences_from_numpy(data, window_len, step)

def prepare_sequence_dataset(dji_df, esp32_df, window_len=WINDOW_LEN, random_state=42, features=None, split_mode='device'):
    """
    Prepares sequential train, validation, and test datasets.
    """
    if features is None:
        features = INTERSECTING_FEATURES
        
    dji_df = dji_df.copy()
    esp32_df = esp32_df.copy()

    def collect_windows_by_flight(df, flight_ids, step=1):
        windows = []
        window_flight_ids = []
        for flight_id in flight_ids:
            flight_data = df[df['flight_id'] == flight_id]
            flight_windows = generate_sequences_from_df(flight_data, features, window_len, step=step)
            if len(flight_windows) > 0:
                windows.append(flight_windows)
                window_flight_ids.extend([flight_id] * len(flight_windows))
        return windows, np.array(window_flight_ids)

    def concatenate_windows(window_batches):
        return np.concatenate(window_batches, axis=0) if window_batches else np.empty((0, window_len, len(features)))
    
    if split_mode == 'random':
        print("[Split Mode] Executing Point-wise Random split with non-overlapping windows (80/20)...")
        # DJI windows (non-overlapping)
        dji_flights = dji_df['flight_id'].unique()
        dji_windows_list = [generate_sequences_from_df(dji_df[dji_df['flight_id'] == f], features, window_len, step=window_len) for f in dji_flights]
        X_dji = concatenate_windows(dji_windows_list)
        
        # Split DJI windows randomly 80/20
        X_dji_temp, X_dji_test = train_test_split(X_dji, test_size=0.20, random_state=random_state)
        dji_test_fids = np.array(['random_dji'] * len(X_dji_test))
        # Split temp into Train (80% of temp) and Val (20% of temp)
        X_dji_train, X_dji_val = train_test_split(X_dji_temp, test_size=0.20, random_state=random_state)
        
        # Spoofed windows (non-overlapping, including real esp32_flight mixed in)
        esp32_flights = esp32_df['flight_id'].unique()
        esp32_windows_list = [generate_sequences_from_df(esp32_df[esp32_df['flight_id'] == f], features, window_len, step=window_len) for f in esp32_flights]
        X_esp32_all = concatenate_windows(esp32_windows_list)
        
        # Split spoofed windows 80/20
        X_esp32_temp, X_esp32_test = train_test_split(X_esp32_all, test_size=0.20, random_state=random_state)
        esp32_test_fids = np.array(['random_esp32'] * len(X_esp32_test))
        # Split temp into Train (80% of temp) and Val (20% of temp)
        X_esp32_train, X_esp32_val = train_test_split(X_esp32_temp, test_size=0.20, random_state=random_state)
            
    elif split_mode == 'device':
        if 'drone_model' not in dji_df.columns:
            raise ValueError(
                "split_mode='device' requires dji_df to have a 'drone_model' column."
            )
        
        # Split unique drone models dynamically 80/20
        unique_models = dji_df['drone_model'].dropna().unique()
        train_val_models, test_models = train_test_split(
            unique_models, test_size=0.20, random_state=random_state
        )
        
        # Train/Val models further split 90/10 to carve out val
        train_models, val_models = train_test_split(
            train_val_models, test_size=0.10, random_state=random_state
        )
        
        train_flights = dji_df.loc[dji_df['drone_model'].isin(train_models), 'flight_id'].dropna().unique()
        val_flights = dji_df.loc[dji_df['drone_model'].isin(val_models), 'flight_id'].dropna().unique()
        test_flights = dji_df.loc[dji_df['drone_model'].isin(test_models), 'flight_id'].dropna().unique()
        
        # DJI Train/Val/Test sequences
        dji_train_windows, _ = collect_windows_by_flight(dji_df, train_flights)
        dji_val_windows, _ = collect_windows_by_flight(dji_df, val_flights)
        dji_test_windows, dji_test_fids = collect_windows_by_flight(dji_df, test_flights)

        X_dji_train = concatenate_windows(dji_train_windows)
        X_dji_val = concatenate_windows(dji_val_windows)
        X_dji_test = concatenate_windows(dji_test_windows)
        
        # Spoofed Split
        all_flights = esp32_df['flight_id'].unique()
        sim_flights = [f for f in all_flights if f != 'esp32_flight']
        
        sim_groups = {}
        for f in sim_flights:
            prefix = f.split('_flight_')[0] if '_flight_' in f else f.split('_')[0]
            sim_groups.setdefault(prefix, []).append(f)
            
        test_sim_flights = []
        train_pool_flights = []
        for prefix, flights in sim_groups.items():
            if len(flights) > 1:
                tr_f, te_f = train_test_split(
                    sorted(flights), test_size=0.20, random_state=random_state
                )
                train_pool_flights.extend(tr_f)
                test_sim_flights.extend(te_f)
            else:
                train_pool_flights.extend(flights)
            
        train_flights_sim, val_flights_sim = train_test_split(
            train_pool_flights, test_size=0.20, random_state=random_state
        )
        
        esp32_train_df = esp32_df[esp32_df['flight_id'].isin(train_flights_sim)].drop(columns=['flight_id'], errors='ignore')
        esp32_val_df = esp32_df[esp32_df['flight_id'].isin(val_flights_sim)].drop(columns=['flight_id'], errors='ignore')
        
        esp32_sim_test_windows, esp32_sim_fids = collect_windows_by_flight(esp32_df, test_sim_flights)
        esp32_real_windows, esp32_real_fids = collect_windows_by_flight(esp32_df, ['esp32_flight'], step=window_len)
        
        X_esp32_train = generate_sequences_from_df(esp32_train_df, features, window_len)
        X_esp32_val = generate_sequences_from_df(esp32_val_df, features, window_len)
        X_esp32_sim_test = concatenate_windows(esp32_sim_test_windows)
        X_esp32_real_test = concatenate_windows(esp32_real_windows)

        
        if len(X_esp32_real_test) > 1500:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(X_esp32_real_test), size=1500, replace=False)
            X_esp32_real_test = X_esp32_real_test[indices]
            esp32_real_fids = esp32_real_fids[indices]
            
        if len(X_esp32_sim_test) > 0:
            X_esp32_test = np.concatenate([X_esp32_real_test, X_esp32_sim_test], axis=0) if len(X_esp32_real_test) > 0 else X_esp32_sim_test
            esp32_test_fids = np.concatenate([esp32_real_fids, esp32_sim_fids], axis=0) if len(esp32_real_fids) > 0 else esp32_sim_fids
        else:
            X_esp32_test = X_esp32_real_test
            esp32_test_fids = esp32_real_fids
            
    else:
        # Flight Split
        unique_flights = dji_df['flight_id'].unique()
        temp_flights, test_flights = train_test_split(
            unique_flights, test_size=0.20, random_state=random_state
        )
        train_flights, val_flights = train_test_split(
            temp_flights, test_size=0.20, random_state=random_state
        )

        dji_train_windows, _ = collect_windows_by_flight(dji_df, train_flights)
        dji_val_windows, _ = collect_windows_by_flight(dji_df, val_flights)
        dji_test_windows, dji_test_fids = collect_windows_by_flight(dji_df, test_flights)

        X_dji_train = concatenate_windows(dji_train_windows)
        X_dji_val = concatenate_windows(dji_val_windows)
        X_dji_test = concatenate_windows(dji_test_windows)
        
        all_flights = esp32_df['flight_id'].unique()
        sim_flights = [f for f in all_flights if f != 'esp32_flight']
        
        sim_groups = {}
        for f in sim_flights:
            prefix = f.split('_flight_')[0] if '_flight_' in f else f.split('_')[0]
            sim_groups.setdefault(prefix, []).append(f)
            
        test_sim_flights = []
        train_pool_flights = []
        for prefix, flights in sim_groups.items():
            if len(flights) > 1:
                tr_f, te_f = train_test_split(
                    sorted(flights), test_size=0.20, random_state=random_state
                )
                train_pool_flights.extend(tr_f)
                test_sim_flights.extend(te_f)
            else:
                train_pool_flights.extend(flights)
            
        train_flights_sim, val_flights_sim = train_test_split(
            train_pool_flights, test_size=0.20, random_state=random_state
        )
        
        esp32_train_df = esp32_df[esp32_df['flight_id'].isin(train_flights_sim)].drop(columns=['flight_id'], errors='ignore')
        esp32_val_df = esp32_df[esp32_df['flight_id'].isin(val_flights_sim)].drop(columns=['flight_id'], errors='ignore')
        
        esp32_sim_test_windows, esp32_sim_fids = collect_windows_by_flight(esp32_df, test_sim_flights)
        esp32_real_windows, esp32_real_fids = collect_windows_by_flight(esp32_df, ['esp32_flight'], step=window_len)
        
        X_esp32_train = generate_sequences_from_df(esp32_train_df, features, window_len)
        X_esp32_val = generate_sequences_from_df(esp32_val_df, features, window_len)
        X_esp32_sim_test = concatenate_windows(esp32_sim_test_windows)
        X_esp32_real_test = concatenate_windows(esp32_real_windows)

        
        if len(X_esp32_real_test) > 1500:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(X_esp32_real_test), size=1500, replace=False)
            X_esp32_real_test = X_esp32_real_test[indices]
            esp32_real_fids = esp32_real_fids[indices]
            
        if len(X_esp32_sim_test) > 0:
            X_esp32_test = np.concatenate([X_esp32_real_test, X_esp32_sim_test], axis=0) if len(X_esp32_real_test) > 0 else X_esp32_sim_test
            esp32_test_fids = np.concatenate([esp32_real_fids, esp32_sim_fids], axis=0) if len(esp32_real_fids) > 0 else esp32_sim_fids
        else:
            X_esp32_test = X_esp32_real_test
            esp32_test_fids = esp32_real_fids
    
    print(f"Generated Raw Sequences:")
    print(f"  DJI split mode: {split_mode}")
    print(f"  Real DJI Train: {X_dji_train.shape} | Val: {X_dji_val.shape} | Test: {X_dji_test.shape}")
    print(f"  Fake ESP32 Train: {X_esp32_train.shape} | Val: {X_esp32_val.shape} | Test: {X_esp32_test.shape}")
    
    X_train_unscaled = np.concatenate([X_dji_train, X_esp32_train], axis=0)
    y_train = np.concatenate([np.zeros(len(X_dji_train)), np.ones(len(X_esp32_train))])

    X_val_unscaled = np.concatenate([X_dji_val, X_esp32_val], axis=0)
    y_val = np.concatenate([np.zeros(len(X_dji_val)), np.ones(len(X_esp32_val))])
    
    X_test_unscaled = np.concatenate([X_dji_test, X_esp32_test], axis=0)
    y_test = np.concatenate([np.zeros(len(X_dji_test)), np.ones(len(X_esp32_test))])
    flight_ids_test = np.concatenate([dji_test_fids, esp32_test_fids], axis=0)
        
    # Fit and Apply Scaler
    n_train, timesteps, n_features = X_train_unscaled.shape
    n_val = X_val_unscaled.shape[0]
    n_test = X_test_unscaled.shape[0]
    
    X_train_2d = X_train_unscaled.reshape(-1, n_features)
    X_val_2d = X_val_unscaled.reshape(-1, n_features)
    X_test_2d = X_test_unscaled.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_val_scaled_2d = scaler.transform(X_val_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)
    
    X_train = X_train_scaled_2d.reshape(n_train, timesteps, n_features)
    X_val = X_val_scaled_2d.reshape(n_val, timesteps, n_features)
    X_test = X_test_scaled_2d.reshape(n_test, timesteps, n_features)
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Shuffle splits
    np.random.seed(random_state)
    shuffle_train_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_train_idx]
    y_train = y_train[shuffle_train_idx]
    
    shuffle_val_idx = np.random.permutation(len(X_val))
    X_val = X_val[shuffle_val_idx]
    y_val = y_val[shuffle_val_idx]
    
    shuffle_test_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_test_idx]
    y_test = y_test[shuffle_test_idx]
    flight_ids_test = flight_ids_test[shuffle_test_idx]
    
    print(f"Final Sequence Datasets:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape} (Pos ratio: {np.mean(y_train)*100:.2f}%)")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape} (Pos ratio: {np.mean(y_val)*100:.2f}%)")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape} (Pos ratio: {np.mean(y_test)*100:.2f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, flight_ids_test

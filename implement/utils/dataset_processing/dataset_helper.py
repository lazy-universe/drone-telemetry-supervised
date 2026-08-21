import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from implement.utils.helper import INTERSECTING_FEATURES

def impute_and_scale_data(X_train, X_test, imputer=None, scaler=None):
    """
    Imputes missing values and scales features. Fits new models if not provided.
    Returns scaled features, fitted scaler, and fitted imputer.
    """
    if imputer is None:
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = imputer.fit_transform(X_train)
    else:
        X_train_imputed = imputer.transform(X_train)
        
    X_test_imputed = imputer.transform(X_test)
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
    else:
        X_train_scaled = scaler.transform(X_train_imputed)
        
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, scaler, imputer

def combine_and_split_supervised_data(dji_df, esp32_df, train_ratio=0.8, split_mode='random', intermediate_dir=None, output_dir=None, random_state=42, features=None):
    """
    Merges DJI and ESP32 datasets, splits into train/test using either point-wise random split or flight-wise partition split,
    removes duplicates, and fits standardizer and imputer.
    """
    if features is None:
        features = INTERSECTING_FEATURES

    dji_clean = dji_df.copy()
    esp32_clean = esp32_df.copy()
    
    if split_mode == 'flight':
        print(f"[Split Mode] Executing Flight-wise partition split ({int(train_ratio*100)}/{int((1-train_ratio)*100)})...")
        # 1. Split DJI flight IDs flight-wise
        if 'flight_id' in dji_clean.columns:
            unique_dji_flights = dji_clean['flight_id'].unique()
            dji_train_flights, dji_test_flights = train_test_split(
                unique_dji_flights, test_size=(1.0 - train_ratio), random_state=random_state
            )
            dji_train = dji_clean[dji_clean['flight_id'].isin(dji_train_flights)].drop(columns=['flight_id'], errors='ignore')
            dji_test = dji_clean[dji_clean['flight_id'].isin(dji_test_flights)].drop(columns=['flight_id'], errors='ignore')
        else:
            dji_train, dji_test = train_test_split(dji_clean, test_size=(1.0 - train_ratio), random_state=random_state)
            
        # 2. Split Spoofed flight-wise dynamically: Train strictly on simulated flights (excluding test files), 
        # Test strictly on real ESP32 AND selected simulated flights dynamically split by class prefix
        if 'flight_id' in esp32_clean.columns:
            all_flights = esp32_clean['flight_id'].unique()
            sim_flights = [f for f in all_flights if f != 'esp32_flight']
            
            # Group simulated flights dynamically by prefix
            sim_groups = {}
            for f in sim_flights:
                prefix = f.split('_flight_')[0] if '_flight_' in f else f.split('_')[0]
                sim_groups.setdefault(prefix, []).append(f)
                
            test_sim_flights = []
            train_sim_flights = []
            for prefix, flights in sim_groups.items():
                if len(flights) > 1:
                    tr_f, te_f = train_test_split(
                        sorted(flights), test_size=(1.0 - train_ratio), random_state=random_state
                    )
                    train_sim_flights.extend(tr_f)
                    test_sim_flights.extend(te_f)
                else:
                    train_sim_flights.extend(flights)
            
            esp32_train = esp32_clean[esp32_clean['flight_id'].isin(train_sim_flights)].drop(columns=['flight_id'], errors='ignore')
            
            # Test contains the real ESP32 flight (limited to 1500 samples) AND the selected simulated test flights
            esp32_real = esp32_clean[esp32_clean['flight_id'] == 'esp32_flight'].drop(columns=['flight_id'], errors='ignore')
            if len(esp32_real) > 1500:
                esp32_real_test = esp32_real.sample(n=1500, random_state=random_state)
            else:
                esp32_real_test = esp32_real
                
            esp32_sim_test = esp32_clean[esp32_clean['flight_id'].isin(test_sim_flights)].drop(columns=['flight_id'], errors='ignore')
            esp32_test = pd.concat([esp32_real_test, esp32_sim_test], ignore_index=True)
            print(f"  Simulated flights selected for Testing: {test_sim_flights}")
        else:
            esp32_df_clean = esp32_clean.drop(columns=['flight_id'], errors='ignore')
            esp32_train, esp32_test = train_test_split(
                esp32_df_clean, test_size=(1.0 - train_ratio), random_state=random_state
            )
    elif split_mode == 'device':
        print(f"[Split Mode] Executing Device-wise split dynamically grouped by drone models ({int(train_ratio*100)}/{int((1-train_ratio)*100)})...")
        if 'drone_model' not in dji_clean.columns:
            raise ValueError(
                "split_mode='device' requires dji_df to have a 'drone_model' column."
            )
        
        # Symmetrically split drone models: 80% train/val models, 20% test models
        unique_models = dji_clean['drone_model'].unique()
        dji_train_models, dji_test_models = train_test_split(
            unique_models, test_size=(1.0 - train_ratio), random_state=random_state
        )
        print(f"  Training drone models: {list(dji_train_models)}")
        print(f"  Testing drone models: {list(dji_test_models)}")
        
        dji_train = (
            dji_clean[dji_clean['drone_model'].isin(dji_train_models)]
            .drop(columns=['drone_model', 'flight_id'], errors='ignore')
        )
        dji_test = (
            dji_clean[dji_clean['drone_model'].isin(dji_test_models)]
            .drop(columns=['drone_model', 'flight_id'], errors='ignore')
        )
        # Spoofed: same convention as flight-wise — train on simulated (except test ones), test on real ESP32 (capped to 1500) + selected simulated flights
        if 'flight_id' in esp32_clean.columns:
            all_flights = esp32_clean['flight_id'].unique()
            sim_flights = [f for f in all_flights if f != 'esp32_flight']
            
            # Group simulated flights dynamically by prefix
            sim_groups = {}
            for f in sim_flights:
                prefix = f.split('_flight_')[0] if '_flight_' in f else f.split('_')[0]
                sim_groups.setdefault(prefix, []).append(f)
                
            test_sim_flights = []
            train_sim_flights = []
            for prefix, flights in sim_groups.items():
                if len(flights) > 1:
                    tr_f, te_f = train_test_split(
                        sorted(flights), test_size=(1.0 - train_ratio), random_state=random_state
                    )
                    train_sim_flights.extend(tr_f)
                    test_sim_flights.extend(te_f)
                else:
                    train_sim_flights.extend(flights)
                
            esp32_train = esp32_clean[esp32_clean['flight_id'].isin(train_sim_flights)].drop(columns=['flight_id'], errors='ignore')
            
            esp32_real = esp32_clean[esp32_clean['flight_id'] == 'esp32_flight'].drop(columns=['flight_id'], errors='ignore')
            if len(esp32_real) > 1500:
                esp32_real_test = esp32_real.sample(n=1500, random_state=random_state)
            else:
                esp32_real_test = esp32_real
                
            esp32_sim_test = esp32_clean[esp32_clean['flight_id'].isin(test_sim_flights)].drop(columns=['flight_id'], errors='ignore')
            esp32_test = pd.concat([esp32_real_test, esp32_sim_test], ignore_index=True)
        else:
            esp32_train, esp32_test = train_test_split(
                esp32_clean.drop(columns=['flight_id'], errors='ignore'),
                test_size=(1.0 - train_ratio), random_state=random_state
            )
        print(f"  DJI  — train: {len(dji_train)} rows | test: {len(dji_test)} rows")
        print(f"  ESP32 — train: {len(esp32_train)} rows | test: {len(esp32_test)} rows")
    else:
        print("[Split Mode] Executing Point-wise random split on mixed datasets...")
        dji_no_id = dji_clean.drop(columns=['flight_id'], errors='ignore')
        esp32_no_id = esp32_clean.drop(columns=['flight_id'], errors='ignore')
        
        combined_df = pd.concat([dji_no_id, esp32_no_id], ignore_index=True)
        
        train_df_split, test_df_split = train_test_split(
            combined_df, test_size=(1.0 - train_ratio), random_state=random_state, stratify=combined_df['nature']
        )
        
        dji_train = train_df_split[train_df_split['nature'] == 0]
        esp32_train = train_df_split[train_df_split['nature'] == 1]
        dji_test = test_df_split[test_df_split['nature'] == 0]
        esp32_test = test_df_split[test_df_split['nature'] == 1]
        
    train_df = pd.concat([dji_train, esp32_train], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([dji_test, esp32_test], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_df['nature'] = train_df['nature'].astype(int)
    test_df['nature'] = test_df['nature'].astype(int)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / 'train_dataset.csv', index=False)
        test_df.to_csv(output_dir / 'test_dataset.csv', index=False)
        print(f"Saved split datasets to: {output_dir}")
        
    X_train = train_df[features]
    y_train = train_df['nature'].to_numpy()
    
    X_test = test_df[features]
    y_test = test_df['nature'].to_numpy()
    
    X_train_scaled, X_test_scaled, scaler, imputer = impute_and_scale_data(X_train, X_test)
    print(f"Dataset split completed ({split_mode}): Train={X_train_scaled.shape} | Test={X_test_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, imputer

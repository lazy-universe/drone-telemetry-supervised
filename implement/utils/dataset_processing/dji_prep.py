import io
import os
import re
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


# Conversion factors
FT_TO_METERS = 0.3048
MPH_TO_MS = 0.44704

from implement.utils.helper import SUPERVISED_FEATURES, engineer_features_for_df
from implement.utils.helper.features import compute_geographic_bearing


# Forensic raw flight log filename pattern: e.g. DJIFlightRecord_2017-08-03_[15-59-24].csv
FORENSIC_PATTERN = re.compile(r'.*_\d{4}-\d{2}-\d{2}_\[\d{2}-\d{2}-\d{2}\].*\.csv$', re.IGNORECASE)

def wrap_angle_difference(diff):
    """
    Wraps angle differences to the range [-180, 180] degrees to handle angular boundaries.
    """
    return (diff + 180) % 360 - 180


def clean_and_load_csv(file_path):
    """
    Cleans UTF-8 BOM, sep= metadata, and parses a CSV file cleanly.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if not lines:
        return None

    if lines[0].startswith('\ufeff'):
        lines[0] = lines[0].replace('\ufeff', '')

    start_idx = 0
    if lines[0].strip().startswith('sep='):
        start_idx = 1

    header_line = lines[start_idx]
    delim = '\t' if '\t' in header_line else ','

    clean_csv_data = ''.join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(clean_csv_data), sep=delim)
    df.columns = [col.strip() for col in df.columns]
    
    # Drop duplicate columns to prevent DataFrame-indexing errors
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Drop index column if present
    first_col = df.columns[0]
    if first_col.lower() in ['#', 'no', 'no.', 'index']:
        df = df.drop(columns=[first_col])

    return df


def extract_raw_dji_logs(dji_raw_dir, output_raw_dir):
    """
    Recursively scans dji_raw_dir for CSV files matching the forensic log filename pattern,
    maps them to their drone models, determines train/test prefix, and copies them to output_raw_dir
    with names like f"{prefix}_flight_{idx:02d}_{model_suffix}.csv".
    """
    dji_raw_dir = Path(dji_raw_dir)
    output_raw_dir = Path(output_raw_dir)
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if files are already extracted
    if any(output_raw_dir.glob("*.csv")):
        print(f"Raw directory '{output_raw_dir}' is not empty. Skipping extraction.")
        return output_raw_dir

    # Find droner raw dir to map filenames to models
    droner_raw_dir = dji_raw_dir / 'DroNER Dataset for Drone Named Entity Recognition' / 'droner' / 'raw'
    if not droner_raw_dir.exists():
        droner_raw_dir = dji_raw_dir.parent / 'DroNER Dataset for Drone Named Entity Recognition' / 'droner' / 'raw'

    filename_to_model = {}
    if droner_raw_dir.exists():
        for root, _, files in os.walk(droner_raw_dir):
            for file_name in files:
                if file_name.endswith('.csv'):
                    parts = Path(root).parts
                    if 'raw' in parts:
                        idx = parts.index('raw')
                        if idx + 1 < len(parts):
                            model_name = parts[idx + 1]
                            filename_to_model[file_name] = model_name

    def std_name(name):
        return re.sub(r'[^A-Z0-9]', '', str(name).upper())

    unique_models = sorted(list(set(filename_to_model.values())))
    if not unique_models:
        unique_models = ['DJI_PHANTOM_4_PRO_V2', 'DJI_MAVIC_AIR', 'DJI_MATRICE_600']
    
    import random
    rng = random.Random(42)
    test_models = rng.sample(unique_models, k=max(1, int(len(unique_models) * 0.3)))
    test_models_std = {std_name(m) for m in test_models}

    matched_files = []
    for root, _, files in os.walk(dji_raw_dir):
        for file_name in files:
            if not file_name.lower().endswith('.csv'):
                continue
            if file_name.startswith('extracted_') or 'extracted_' in root or file_name.endswith('.csv.csv'):
                continue
            # Avoid matching already extracted/renamed files in raw_dataset
            if 'raw_dataset' in Path(root).parts:
                continue
            if FORENSIC_PATTERN.match(file_name):
                matched_files.append((file_name, Path(root) / file_name))

    print(f"Found {len(matched_files)} matching forensic DJI raw logs. Copying and renaming to raw_dataset folder...")
    
    prefix_counts = {"train": 0, "test": 0}
    for file_name, full_src_path in sorted(matched_files):
        drone_model = filename_to_model.get(file_name, 'Unknown')
        if std_name(drone_model) in test_models_std:
            prefix = 'test'
        else:
            prefix = 'train'
            
        prefix_counts[prefix] += 1
        model_suffix = drone_model.lower()
        dest_name = f"{prefix}_flight_{prefix_counts[prefix]:02d}_{model_suffix}.csv"
        dest_path = output_raw_dir / dest_name
        shutil.copy2(full_src_path, dest_path)
        
    print(f"Copied and renamed {len(matched_files)} raw CSV files into {output_raw_dir}.")
    return output_raw_dir


def preprocess_dji_points(raw_dir, output_master_file=None):
    """
    Loads all raw DJI flight CSVs recursively, converts units to metric, derives flight path features,
    and returns a combined point-wise DataFrame labeled with nature = 1.
    If output_master_file is provided, saves the processed master CSV to disk.
    Deduplicates flight records files in trimmed_dir using MD5 hashing once the trimmed dataset is configured.
    """
    from implement.utils.helper import (
        get_dji_engineered_dir,
        get_dji_trimmed_dir
    )
    engineered_dir = get_dji_engineered_dir()
    trimmed_dir = get_dji_trimmed_dir()
    
    if output_master_file:
        output_master_file = Path(output_master_file)
        output_master_file.parent.mkdir(parents=True, exist_ok=True)
    
    engineered_dir.mkdir(parents=True, exist_ok=True)
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path(raw_dir)
    genuine_dji_flights_dir = raw_dir.parent
    
    # 1. Check/Fetch DroNER dataset (cache logic)
    droner_raw_dir = genuine_dji_flights_dir / 'DroNER Dataset for Drone Named Entity Recognition' / 'droner' / 'raw'
    if not droner_raw_dir.exists() or not any(droner_raw_dir.rglob("*.csv")):
        print("DroNER dataset not found in cache. Fetching...")
        from implement.initial_setup import download_and_extract_genuine_dji_flights
        download_and_extract_genuine_dji_flights(genuine_dji_flights_dir)
        
    # 2. Extract and rename raw DJI flight logs if not already done
    if not raw_dir.exists() or not any(raw_dir.glob("*.csv")):
        print(f"Raw directory {raw_dir} is empty. Extracting and renaming from source...")
        extract_raw_dji_logs(genuine_dji_flights_dir, raw_dir)

    # Check if trimmed DJI CSVs are already present in trimmed_dir
    trimmed_files = sorted(list(trimmed_dir.glob("*.csv")))
    
    if not trimmed_files:
        print(f"Trimmed directory {trimmed_dir} is empty. Migrating raw -> trimmed...")
        matched_files = sorted(list(raw_dir.glob("*.csv")))
        print(f"Processing {len(matched_files)} raw DJI flight logs for point features...")
        
        for file_path in matched_files:
            df = clean_and_load_csv(file_path)
            if df is None or len(df) < 100:
                continue

            def get_col(*names):
                for name in names:
                    if name in df.columns:
                        return name
                return None

            fly_time_col = get_col('OSD.flyTime [s]', 'OSD.flyTime')
            lat_col = get_col('OSD.latitude', 'OSD.latitude_relative')
            lon_col = get_col('OSD.longitude', 'OSD.longitude_relative')
            altitude_col = get_col('OSD.altitude [ft]', 'OSD.altitude [m]', 'OSD.altitude')
            height_col = get_col('OSD.height [ft]', 'OSD.height [m]', 'OSD.height')
            hspeed_col = get_col('OSD.hSpeed [MPH]', 'OSD.hSpeed [m/s]', 'OSD.hSpeed')
            vspeed_col = get_col('OSD.vSpeed [MPH]', 'OSD.vSpeed [m/s]', 'OSD.vSpeed', 'OSD.zSpeed [MPH]', 'OSD.zSpeed [m/s]', 'OSD.zSpeed')
            xspeed_col = get_col('OSD.xSpeed [MPH]', 'OSD.xSpeed [m/s]', 'OSD.xSpeed')
            yspeed_col = get_col('OSD.ySpeed [MPH]', 'OSD.ySpeed [m/s]', 'OSD.ySpeed')

            if fly_time_col is None or lat_col is None or lon_col is None:
                continue

            df_trimmed = pd.DataFrame()
            fly_time = pd.to_numeric(df[fly_time_col], errors='coerce')
            t_start = fly_time.iloc[0] if len(fly_time) > 0 else 0.0
            df_trimmed['timestamp'] = fly_time - t_start
            df_trimmed['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
            df_trimmed['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')

            if altitude_col is not None:
                df_trimmed['altitude'] = pd.to_numeric(df[altitude_col], errors='coerce')
                if '[ft]' in altitude_col:
                    df_trimmed['altitude'] = df_trimmed['altitude'] * FT_TO_METERS
            else:
                df_trimmed['altitude'] = np.nan

            if height_col is not None:
                df_trimmed['height'] = pd.to_numeric(df[height_col], errors='coerce')
                if '[ft]' in height_col:
                    df_trimmed['height'] = df_trimmed['height'] * FT_TO_METERS
            else:
                df_trimmed['height'] = np.nan

            if hspeed_col is not None:
                df_trimmed['ground_speed'] = pd.to_numeric(df[hspeed_col], errors='coerce')
                if '[MPH]' in hspeed_col:
                    df_trimmed['ground_speed'] = df_trimmed['ground_speed'] * MPH_TO_MS
            else:
                df_trimmed['ground_speed'] = np.nan

            if vspeed_col is not None:
                df_trimmed['vertical_speed'] = pd.to_numeric(df[vspeed_col], errors='coerce')
                if '[MPH]' in vspeed_col:
                    df_trimmed['vertical_speed'] = df_trimmed['vertical_speed'] * MPH_TO_MS
            else:
                df_trimmed['vertical_speed'] = np.nan

            x_speed = None
            y_speed = None
            if xspeed_col is not None and yspeed_col is not None:
                x_speed = pd.to_numeric(df[xspeed_col], errors='coerce')
                y_speed = pd.to_numeric(df[yspeed_col], errors='coerce')
                if '[MPH]' in xspeed_col:
                    x_speed = x_speed * MPH_TO_MS
                if '[MPH]' in yspeed_col:
                    y_speed = y_speed * MPH_TO_MS

            if x_speed is not None and y_speed is not None:
                vector_course = (np.degrees(np.arctan2(y_speed, x_speed)) + 360) % 360
            else:
                vector_course = pd.Series(np.nan, index=df.index, dtype=float)

            next_lat = df_trimmed['latitude'].shift(-1)
            next_lon = df_trimmed['longitude'].shift(-1)
            bearing_course = compute_geographic_bearing(df_trimmed['latitude'].values, df_trimmed['longitude'].values, next_lat.values, next_lon.values)
            course = pd.Series(vector_course, index=df.index, dtype=float)
            course = course.where(course.notna(), bearing_course)
            df_trimmed['course'] = course

            df_trimmed = df_trimmed[['timestamp', 'latitude', 'longitude', 'altitude', 'height', 'ground_speed', 'vertical_speed', 'course']]
            df_trimmed = df_trimmed.replace([np.inf, -np.inf], np.nan)

            # Save trimmed dataset
            df_trimmed.to_csv(trimmed_dir / file_path.name, index=False)

        print(f"Trimmed dataset configured in {trimmed_dir}.")
    else:
        print(f"✓ Found {len(trimmed_files)} trimmed DJI logs in {trimmed_dir}.")

    # Deduplicate flight records files in trimmed_dir using MD5 hashing and filter out files with < 100 rows
    # This is done when the trimmed dataset is already configured on disk.
    import hashlib
    trimmed_files = sorted(list(trimmed_dir.glob("*.csv")))
    
    hash_map = {}  # md5_hash -> first_filepath
    duplicates = []
    
    for file_path in trimmed_files:
        try:
            df_check = pd.read_csv(file_path, low_memory=False)
            if len(df_check) < 100:
                print(f"Adding flight record file with < 100 samples to deletion queue: {file_path.name} (length: {len(df_check)})")
                duplicates.append((file_path, None))
                continue
        except Exception as e:
            print(f"Error reading file {file_path.name} for row count check: {e}")
            duplicates.append((file_path, None))
            continue
            
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        file_hash = hasher.hexdigest()
        
        if file_hash in hash_map:
            duplicates.append((file_path, hash_map[file_hash]))
        else:
            hash_map[file_hash] = file_path
            
    for dup_path, orig_path in duplicates:
        if orig_path:
            print(f"Removing duplicate flight record file: {dup_path.name} (duplicate of {orig_path.name})")
        else:
            print(f"Removing short or corrupt flight record file: {dup_path.name}")
        try:
            dup_path.unlink()
        except FileNotFoundError:
            pass

    # Now gather unique files and process engineering features
    unique_trimmed_files = sorted(list(trimmed_dir.glob("*.csv")))
    print(f"Total unique trimmed flight records: {len(unique_trimmed_files)}")

    # Create and populate consistent_dataset
    from implement.utils.helper import get_consistent_dataset_dir
    consistent_dir = get_consistent_dataset_dir()
    consistent_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating consistent dataset files in: {consistent_dir}...")
    
    for file_path in unique_trimmed_files:
        df_const = pd.read_csv(file_path, low_memory=False)
        dest_name = file_path.name

        # Standard schema is already stored in trimmed_dir; copy it forward for consistency.
        df_const.to_csv(consistent_dir / dest_name, index=False)

    # Sync engineered_dir to match trimmed_dir by deleting orphaned engineered files
    unique_names = {f.name for f in unique_trimmed_files}
    for eng_file in engineered_dir.glob("*.csv"):
        if eng_file.name not in unique_names:
            try:
                eng_file.unlink()
            except FileNotFoundError:
                pass
                
    all_points = []
    for file_path in unique_trimmed_files:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Compute engineered features directly from the standard schema.
        df_eng = engineer_features_for_df(
            df=df,
            lat_col='latitude',
            lon_col='longitude',
            alt_col='altitude',
            speed_col='ground_speed',
            heading_col='course',
            time_col='timestamp'
        )
        df_eng['height'] = df_eng['altitude']

        # Save engineered dataset individual file
        df_eng.to_csv(engineered_dir / file_path.name, index=False)
        
        # Filter to target columns and assign a unique flight_id (filename)
        df_eng = df_eng[SUPERVISED_FEATURES + ['height']].copy()
        df_eng['flight_id'] = file_path.name
        
        all_points.append(df_eng)
        
    if not all_points:
        empty_df = pd.DataFrame(columns=SUPERVISED_FEATURES + ['height', 'nature', 'flight_id'])
        if output_master_file:
            empty_df.to_csv(output_master_file, index=False)
        return empty_df
        
    combined_dji = pd.concat(all_points, ignore_index=True)
    combined_dji['nature'] = 0  # DJI label (Normal)
    
    # Keep all rows without global deduplication on features
    combined_dji = combined_dji.reset_index(drop=True)
    
    if output_master_file:
        output_master_file = Path(output_master_file)
        output_master_file.parent.mkdir(parents=True, exist_ok=True)
        # Drop flight_id when saving master CSV to match previous schema
        combined_dji.drop(columns=['flight_id'], errors='ignore').to_csv(output_master_file, index=False)
        print(f"Saved DJI master processed points to: {output_master_file}")
        
    return combined_dji


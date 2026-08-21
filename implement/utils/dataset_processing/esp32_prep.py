import numpy as np
import pandas as pd
from pathlib import Path

from implement.utils.helper.features import compute_geographic_bearing

def wrap_angle_difference(diff):
    """
    Wraps angle differences to the range [-180, 180] degrees to handle angular boundaries.
    """
    return (diff + 180) % 360 - 180


def preprocess_esp32_points(esp32_file, output_file=None):
    """
    Loads raw ESP32 bluetooth telemetry, filters Location messages, and converts the data
    directly into the standard Remote ID-style schema:
    timestamp, latitude, longitude, altitude, ground_speed, vertical_speed, course.
    """
    esp32_file = Path(esp32_file)
    if not esp32_file.exists():
        raise FileNotFoundError(f"ESP32 file not found: {esp32_file}")

    df = pd.read_csv(esp32_file)

    df_loc = df[df['Message Type'] == 'Location'].copy()
    if len(df_loc) < 100:
        raise ValueError("Insufficient Location packets found in ESP32 source file.")

    df_loc['Direction'] = pd.to_numeric(df_loc['Direction'].astype(str).str.rstrip('°'), errors='coerce')
    df_loc['Speed Horizontal'] = pd.to_numeric(df_loc['Speed Horizontal'].astype(str).str.replace(' m/s', '', regex=False), errors='coerce')
    if 'Speed Vertical' in df_loc.columns:
        df_loc['Speed Vertical'] = pd.to_numeric(df_loc['Speed Vertical'].astype(str).str.replace(' m/s', '', regex=False), errors='coerce')
    if 'Height' in df_loc.columns:
        df_loc['Height'] = pd.to_numeric(df_loc['Height'].astype(str).str.replace(' m', '', regex=False), errors='coerce')
    if 'Altitude Geodetic' in df_loc.columns:
        df_loc['Altitude Geodetic'] = pd.to_numeric(df_loc['Altitude Geodetic'].astype(str).str.replace(' m', '', regex=False), errors='coerce')

    df_loc['Timestamp'] = pd.to_datetime(df_loc['Timestamp'])
    time_sec = (df_loc['Timestamp'] - df_loc['Timestamp'].iloc[0]).dt.total_seconds()

    df_clean = pd.DataFrame()
    df_clean['timestamp'] = time_sec
    df_clean['latitude'] = pd.to_numeric(df_loc['Latitide'], errors='coerce')
    df_clean['longitude'] = pd.to_numeric(df_loc['Logitude'], errors='coerce')
    if 'Altitude Geodetic' in df_loc.columns:
        df_clean['altitude'] = df_loc['Altitude Geodetic']
    else:
        df_clean['altitude'] = np.nan
    if 'Height' in df_loc.columns:
        df_clean['height'] = df_loc['Height']
    else:
        df_clean['height'] = np.nan
    df_clean['ground_speed'] = df_loc['Speed Horizontal']

    if 'Speed Vertical' in df_loc.columns:
        df_clean['vertical_speed'] = df_loc['Speed Vertical']
    else:
        df_clean['vertical_speed'] = np.nan

    direction = df_loc['Direction'] if 'Direction' in df_loc.columns else pd.Series(np.nan, index=df_loc.index, dtype=float)
    direction = direction.where(direction.notna(), np.nan)
    next_lat = df_clean['latitude'].shift(-1)
    next_lon = df_clean['longitude'].shift(-1)
    bearing_course = compute_geographic_bearing(df_clean['latitude'].values, df_clean['longitude'].values, next_lat.values, next_lon.values)
    course = pd.Series(direction % 360, index=df_loc.index, dtype=float)
    course = course.where(course.notna(), bearing_course)
    df_clean['course'] = course

    df_clean = df_clean[['timestamp', 'latitude', 'longitude', 'altitude', 'height', 'ground_speed', 'vertical_speed', 'course']]
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_file, index=False)
        print(f"Saved ESP32 standard dataset to: {output_file}")

    return df_clean

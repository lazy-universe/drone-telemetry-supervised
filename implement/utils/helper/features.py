import numpy as np
import pandas as pd

WINDOW_LEN = 20  # Window length for sequence slicing (20 samples @ 2Hz = 10.0 seconds of log history)
# WINDOW_LEN = 10  # Window length for sequence slicing (10 samples @ 2Hz = 5.0 seconds of log history)
# WINDOW_LEN = 30  # Window length for sequence slicing (30 samples @ 2Hz = 15.0 seconds of log history)

INTERSECTING_FEATURES = [
    'prediction_error',
    'ground_speed',
    'vertical_acceleration',
    'vertical_speed',
    'acceleration',
    'turn_rate',
    'path_curvature',
    'heading_speed_consistency',
    'motion_smoothness'
    # 'jerk',      // no additional value provided as well might increase the noise due ot higher order derivative
    # 'altitude',
    # 'altitude',
    # 'speed_variance',
    # 'bearing_entropy',
    # 'heading_variance',
    # 'vertical_speed_variance',
]

SUPERVISED_FEATURES = INTERSECTING_FEATURES.copy()
UNSUPERVISED_FEATURES = INTERSECTING_FEATURES.copy()


def compute_geographic_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing from (lat1, lon1) to (lat2, lon2).
    Returns degrees in [0, 360) and preserves NaN where the bearing is undefined.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    bearing = np.full(lat1.shape, np.nan, dtype=float)
    valid = np.isfinite(lat1) & np.isfinite(lon1) & np.isfinite(lat2) & np.isfinite(lon2)
    if not np.any(valid):
        return bearing

    delta_lon = np.radians(lon2[valid] - lon1[valid])
    lat1_rad = np.radians(lat1[valid])
    lat2_rad = np.radians(lat2[valid])
    y = np.sin(delta_lon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon)
    bearing[valid] = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return bearing


def compute_relative_height(altitude_abs):
    """
    Convert an absolute altitude series into a relative height series.
    The first finite altitude becomes zero.
    """
    altitude_abs = np.asarray(altitude_abs, dtype=float)
    height = np.full(altitude_abs.shape, np.nan, dtype=float)
    valid = np.isfinite(altitude_abs)
    if not np.any(valid):
        return height

    first_idx = np.flatnonzero(valid)[0]
    height = altitude_abs - altitude_abs[first_idx]
    return height


def wrap_angle_difference(diff):
    """
    Wraps angle differences to the range [-180, 180] degrees to handle angular boundaries.
    """
    return (diff + 180) % 360 - 180

def compute_entropy(window_values):
    """
    Computes Shannon entropy of binned heading angles inside a sliding window.
    """
    wrapped = window_values % 360
    bins = (wrapped // 45).astype(int)
    counts = np.bincount(bins, minlength=8)
    probs = counts / len(window_values)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def resample_df(df, lat_col, lon_col, alt_col, speed_col, heading_col, time_col, step_size=0.5):
    """
    Resamples a dataframe to a strict regular grid (default 2 Hz / 0.5s) using linear interpolation
    for coordinates & speed, and angular (sin/cos) interpolation for heading.
    """
    time = df[time_col].values
    if len(time) < 2:
        return df.copy()
        
    start_time = time[0]
    end_time = time[-1]
    
    # Generate strict regular time grid
    new_time = np.arange(start_time, end_time, step_size)
    if len(new_time) < 2:
        return df.copy()
        
    resampled_df = pd.DataFrame()
    resampled_df[time_col] = new_time
    
    # Linear interpolation for lat, lon, alt, speed
    for col in [lat_col, lon_col, alt_col, speed_col]:
        if col in df.columns:
            resampled_df[col] = np.interp(new_time, time, df[col].values)
            
    # Angular interpolation for heading to handle 0/360 wrap-around
    if heading_col in df.columns:
        heading = df[heading_col].values
        theta_rad = np.radians(heading)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        new_cos = np.interp(new_time, time, cos_theta)
        new_sin = np.interp(new_time, time, sin_theta)
        new_heading = np.degrees(np.arctan2(new_sin, new_cos)) % 360
        resampled_df[heading_col] = new_heading
        
    return resampled_df

def engineer_features_for_df(df, lat_col, lon_col, alt_col, speed_col, heading_col, time_col, step_size=0.5):
    """
    Computes all 15 engineered features specified in FEATURES.md.
    Automatically resamples the input dataframe to a regular 2 Hz (0.5s) grid first.
    All sensor headings are assumed to be in Aviation convention (degrees clockwise from North)
    and are converted to Cartesian convention (East = 0, counter-clockwise) for physics calculations.
    """
    # 1. Resample to strict 2 Hz (0.5s) grid first to align sampling frequencies
    df_resampled = resample_df(df, lat_col, lon_col, alt_col, speed_col, heading_col, time_col, step_size=step_size)
    
    # Extract absolute variables from the resampled dataframe
    altitude_abs = df_resampled[alt_col].values
    ground_speed_abs = df_resampled[speed_col].values
    heading = df_resampled[heading_col].values
    time = df_resampled[time_col].values
    latitude_abs = df_resampled[lat_col].values
    longitude_abs = df_resampled[lon_col].values
    # 1. Keep altitude absolute and derive height relative to the first valid sample.
    altitude = altitude_abs
    height = compute_relative_height(altitude_abs)
    ground_speed = np.clip(ground_speed_abs, 0.0, 100.0)  # Keep absolute ground speed capped at 100 m/s
    latitude = latitude_abs - latitude_abs[0]
    longitude = longitude_abs - longitude_abs[0]
    
    # Since we resampled to step_size, time_diff is strictly step_size (1.0s)
    time_diff = np.diff(time, prepend=time[0] - step_size)
    time_diff[time_diff == 0] = step_size
    time_diff[np.isnan(time_diff)] = step_size
    
    # 2. Vertical Speed (Keep absolute vertical speed)
    # vz_i = (h_i - h_{i-1}) / dt
    altitude_diff = np.diff(height, prepend=height[0])
    vertical_speed = np.clip(altitude_diff / time_diff, -100.0, 100.0)  # Keep absolute vertical speed capped at +/-100 m/s
    
    # 3. Acceleration
    # a_i = (v_i - v_{i-1}) / dt
    speed_diff = np.diff(ground_speed, prepend=ground_speed[0])
    acceleration = np.clip(speed_diff / time_diff, -15.0, 15.0)
    
    # 4. Vertical Acceleration
    # az_i = (vz_i - vz_{i-1}) / dt
    vspeed_diff = np.diff(vertical_speed, prepend=vertical_speed[0])
    vertical_acceleration = np.clip(vspeed_diff / time_diff, -10.0, 10.0)
    
    # 5. Jerk
    # j_i = (a_i - a_{i-1}) / dt
    acc_diff = np.diff(acceleration, prepend=acceleration[0])
    jerk = np.clip(acc_diff / time_diff, -30.0, 30.0)
    
    # 6. Turn Rate
    # turn_rate = wrap(theta_i - theta_{i-1}) / dt
    heading_diff = np.diff(heading, prepend=heading[0])
    heading_diff_wrapped = wrap_angle_difference(heading_diff)
    turn_rate = np.clip(heading_diff_wrapped / time_diff, -360.0, 360.0)  # Clip turn rate to +/- 360 deg/s
    
    # 7. Path Curvature (requires absolute speed for correct physics)
    # path_curvature = |turn_rate| / ground_speed
    path_curvature = np.where(ground_speed < 0.1, 0.0, np.abs(turn_rate) / np.maximum(ground_speed, 1e-5))
    path_curvature = np.clip(path_curvature, 0.0, 100.0)  # Clip path curvature to 100 rad/m
    
    # Convert series to pandas for rolling calculations
    df_temp = pd.DataFrame({
        'ground_speed': ground_speed,
        'vertical_speed': vertical_speed,
        'heading': heading,
        'jerk': jerk
    })
    
    # Compute rolling window size representing 3 seconds of telemetry history
    window_len = int(round(3.0 / step_size))
    
    # 8. Speed Variance
    speed_variance = df_temp['ground_speed'].rolling(window=window_len, min_periods=1).var(ddof=0).values
    
    # 9. Vertical Speed Variance
    vertical_speed_variance = df_temp['vertical_speed'].rolling(window=window_len, min_periods=1).var(ddof=0).values
    
    # 10. Heading Variance
    # Convert compass heading (Aviation) to Cartesian radians for angular variance calculations
    theta_rad = np.radians((90.0 - heading) % 360)
    df_temp['cos_theta'] = np.cos(theta_rad)
    df_temp['sin_theta'] = np.sin(theta_rad)
    mean_cos = df_temp['cos_theta'].rolling(window=window_len, min_periods=1).mean()
    mean_sin = df_temp['sin_theta'].rolling(window=window_len, min_periods=1).mean()
    heading_variance = (1.0 - np.sqrt(mean_cos**2 + mean_sin**2)).values
    
    # 11. Bearing Entropy
    bearing_entropy = df_temp['heading'].rolling(window=window_len, min_periods=1).apply(compute_entropy, raw=True).values
    
    # 12. Prediction Error (uses absolute coordinates & absolute speed for physical correctness)
    lat_diff = np.diff(latitude_abs, prepend=latitude_abs[0])
    lon_diff = np.diff(longitude_abs, prepend=longitude_abs[0])
    ref_lat = latitude_abs[0]
    cos_ref_lat = np.cos(np.radians(ref_lat))
    dy_obs = lat_diff * 111139.0
    dx_obs = lon_diff * 111139.0 * cos_ref_lat
    
    # Prediction uses Cartesian heading radians
    theta_rad_prev = np.roll(theta_rad, 1)
    theta_rad_prev[0] = theta_rad[0]
    speed_prev = np.roll(ground_speed_abs, 1)
    speed_prev[0] = ground_speed_abs[0]
    
    x_pred_diff = speed_prev * np.cos(theta_rad_prev) * time_diff
    y_pred_diff = speed_prev * np.sin(theta_rad_prev) * time_diff
    
    # First step error is 0.0
    x_pred_diff[0] = 0.0
    y_pred_diff[0] = 0.0
    dy_obs[0] = 0.0
    dx_obs[0] = 0.0
    
    err_x = dx_obs - x_pred_diff
    err_y = dy_obs - y_pred_diff
    prediction_error = np.clip(np.sqrt(err_x**2 + err_y**2), 0.0, 25.0)

    
    # 13. Heading Speed Consistency
    # theta_track is calculated in Cartesian degrees (relative to East)
    theta_track = np.degrees(np.arctan2(dy_obs, dx_obs))
    # Convert theta_track to Aviation degrees (relative to North) to compare with sensor heading
    theta_track_aviation = (90.0 - theta_track) % 360
    
    diff = np.abs(theta_track_aviation - heading) % 360
    heading_speed_consistency = np.minimum(diff, 360.0 - diff)
    # First step consistency is 0.0
    heading_speed_consistency[0] = 0.0
    
    # 14. Motion Smoothness
    motion_smoothness = df_temp['jerk'].abs().rolling(window=window_len, min_periods=1).mean().values
    
    # Construct final dataframe
    df_out = pd.DataFrame()
    df_out['timestamp'] = time
    df_out['latitude'] = latitude_abs
    df_out['longitude'] = longitude_abs
    df_out['course'] = heading
    df_out['ground_speed'] = ground_speed
    df_out['vertical_speed'] = vertical_speed
    df_out['altitude'] = altitude
    df_out['height'] = height
    df_out['acceleration'] = acceleration
    df_out['vertical_acceleration'] = vertical_acceleration
    df_out['jerk'] = jerk
    df_out['turn_rate'] = turn_rate
    df_out['path_curvature'] = path_curvature
    df_out['speed_variance'] = speed_variance
    df_out['vertical_speed_variance'] = vertical_speed_variance
    df_out['heading_variance'] = heading_variance
    df_out['bearing_entropy'] = bearing_entropy
    df_out['prediction_error'] = prediction_error
    df_out['heading_speed_consistency'] = heading_speed_consistency
    df_out['motion_smoothness'] = motion_smoothness
    
    # Preserve NaNs so missing telemetry stays explicit.
    df_out = df_out.replace([np.inf, -np.inf], np.nan)

    return df_out

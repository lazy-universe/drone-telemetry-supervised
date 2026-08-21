from pathlib import Path
import sys

# Detect if running in Google Colab environment
is_colab = 'google.colab' in sys.modules

# Base Directory definition
BASE_DIR = ( Path('/content/drone-telemetry-supervised') if is_colab else Path(__file__).resolve().parents[3] )
REPO_ROOT = BASE_DIR

# Staging directories
DATASET_DIR = BASE_DIR / 'implement' / 'dataset'
OUTPUT_DIR = BASE_DIR / 'implement' / 'output'

# Transient and Ephermal datasets
TRANSIENT_DIR = REPO_ROOT / 'dataset'
EPHERMAL_DIR = DATASET_DIR

def get_base_dir() -> Path:
    return BASE_DIR

def get_dataset_dir() -> Path:
    return DATASET_DIR

def get_output_dir() -> Path:
    return OUTPUT_DIR

def get_transient_dir() -> Path:
    """Returns the original/raw dataset directory (formerly og_dataset)."""
    return TRANSIENT_DIR

def get_pipeline_mode() -> str:
    """Determines which pipeline is active: supervised, unsupervised, or exploration."""
    import os
    if 'PIPELINE_MODE' in os.environ:
        return os.environ['PIPELINE_MODE']
        
    import sys
    argv0 = sys.argv[0] if sys.argv else ""
    argv0_lower = argv0.lower()
    
    # Check if we are running in supervised presets
    if any(k in argv0_lower for k in ["run_random_split", "run_flight_split", "run_sequence_classification", "run_vae_classification", "run_multi_defense"]):
        return "supervised"
        
    # Check if we are running in unsupervised presets
    if any(k in argv0_lower for k in ["run_isolation_forest", "run_autoencoders"]):
        return "unsupervised"
        
    # Check if we are in exploration.ipynb or debug_accuracy.py
    if any(k in argv0_lower for k in ["exploration", "debug_accuracy"]):
        return "exploration"
        
    # Inspect stack frames for notebook filenames or scripts
    import inspect
    for frame_info in inspect.stack():
        filename = frame_info.filename.lower()
        if "supervised" in filename:
            return "supervised"
        if "unsupervised" in filename:
            return "unsupervised"
        if "exploration" in filename:
            return "exploration"
            
    # Default fallback
    return "exploration"

def get_ephermal_dir() -> Path:
    """Returns the intermediate/temp dataset directory (formerly intermediate_dataset)."""
    mode = get_pipeline_mode()
    path = DATASET_DIR / f'ephermal_dataset_{mode}'
    path.mkdir(parents=True, exist_ok=True)
    return path

# Specific file and subdirectory paths
def get_esp32_raw_file() -> Path:
    return TRANSIENT_DIR / 'hardware_spoofer' / 'esp32_source_telemetry.csv'

def get_esp32_final_file() -> Path:
    return get_ephermal_dir() / 'hardware_spoofer' / 'esp32_engineered_features.csv'

def get_genuine_dji_flights_dir() -> Path:
    return TRANSIENT_DIR / 'genuine_dji_flights'

def get_dji_raw_dir() -> Path:
    return TRANSIENT_DIR / 'genuine_dji_flights' / 'raw_dataset'

def get_dji_master_file() -> Path:
    return get_ephermal_dir() / 'genuine_dji_flights' / 'all_processed_dji_records.csv'

def get_dji_engineered_dir() -> Path:
    return get_ephermal_dir() / 'genuine_dji_flights' / 'engineered_dataset'

def get_combined_dataset_dir() -> Path:
    return get_ephermal_dir() / 'combined_dataset'

def get_dji_trimmed_dir() -> Path:
    return TRANSIENT_DIR / 'genuine_dji_flights' / 'trimmed_dataset'

def get_consistent_dataset_dir() -> Path:
    return TRANSIENT_DIR / 'genuine_dji_flights' / 'consistent_dataset'

def get_curated_flights_dir() -> Path:
    return TRANSIENT_DIR / 'curated_flights'

def get_simulated_engineered_dir() -> Path:
    return get_ephermal_dir() / 'curated_flights' / 'engineered_dataset'



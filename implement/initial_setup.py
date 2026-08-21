import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Source URL for Mendeley DJI Drone Telemetry Forensic Dataset
MENDELEY_DATASET_URL = 'https://data.mendeley.com/public-api/zip/fwcjyc754h/download/1'

def download_and_extract_genuine_dji_flights(dest_folder):
    """Downloads the raw Mendeley DJI flight log zip file and extracts it into the target folder."""
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    zip_path = dest_folder / 'downloaded_dataset.zip'

    print(f"Downloading dataset from {MENDELEY_DATASET_URL} using curl...")
    
    # Run curl to follow redirects and save output
    cmd = ['curl', '-L', MENDELEY_DATASET_URL, '-o', str(zip_path)]
    subprocess.run(cmd, check=True)
    
    print(f"Dataset downloaded to {zip_path}. Unzipping files...")
    
    # Extract the downloaded zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
        
    print(f"Dataset unzipped successfully into {dest_folder}.")
    
    # Remove temporary zip file
    if zip_path.exists():
        os.remove(zip_path)
        print("Removed temporary zip file.")

    return dest_folder

def get_workspace_paths(custom_dataset_dir=None, custom_output_dir=None):
    """Retrieves the configured directory paths, returning default values directly."""
    from implement.utils.helper import get_base_dir, get_dataset_dir, get_output_dir, get_transient_dir, get_ephermal_dir
    
    dataset_dir = get_dataset_dir()
    output_dir = get_output_dir()
    transient_dir = get_transient_dir()
    ephermal_dir = get_ephermal_dir()
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    transient_dir.mkdir(parents=True, exist_ok=True)
    ephermal_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': get_base_dir(),
        'dataset': dataset_dir,
        'output': output_dir,
        'transient_dataset': transient_dir,
        'ephermal_dataset': ephermal_dir
    }

def get_run_subfolder_name(pca: bool = False, tune: bool = False, balanced: bool = False, validate: bool = False, cv: int = 5, smote: bool = False):
    """
    Generates an argument-dependent subfolder name for organizing dataset and evaluation outputs.
    Returns None if no active flags are present.
    """
    parts = []
    if pca:
        parts.append("pca")
    if tune:
        parts.append("tune")
    if balanced:
        parts.append("balanced")
    if validate:
        parts.append(f"validate_{cv}")
    if smote:
        parts.append("smote")
    return "_".join(parts) if parts else None

if __name__ == '__main__':
    from implement.utils.helper import get_genuine_dji_flights_dir, get_dataset_dir, get_output_dir
    
    genuine_dji_flights_dir = get_genuine_dji_flights_dir()
    download_and_extract_genuine_dji_flights(genuine_dji_flights_dir)
    
    print("\n" + "="*60)
    print("=== INITIAL SETUP COMPLETED SUCCESSFULLY ===")
    print(f"  Dataset directory: {get_dataset_dir()}")
    print(f"  Output directory:  {get_output_dir()}")
    print("="*60 + "\n")


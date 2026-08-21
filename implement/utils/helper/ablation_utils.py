"""
Centralised ablation override helpers.
All preset scripts import from here instead of duplicating the block.
"""
import shutil
from pathlib import Path

DEFAULT_SEED = 42  # Canonical headline seed — change here and nowhere else

def apply_ablation_overrides(window_size=None, exclude_features=None,
                              project_root: Path = None):
    """
    Mutates the feature module globals for the current process.
    Call BEFORE importing any workflow that reads SUPERVISED_FEATURES / WINDOW_LEN.
    """
    import implement.utils.helper.features as feat

    # Back up originals once per process
    if not hasattr(feat, "_ORIGINAL_SUPERVISED_FEATURES"):
        feat._ORIGINAL_SUPERVISED_FEATURES   = list(feat.SUPERVISED_FEATURES)
        feat._ORIGINAL_INTERSECTING_FEATURES = list(feat.INTERSECTING_FEATURES)
        feat._ORIGINAL_UNSUPERVISED_FEATURES = list(feat.UNSUPERVISED_FEATURES)
        feat._ORIGINAL_WINDOW_LEN = feat.WINDOW_LEN

    # Always restore to originals first
    feat.SUPERVISED_FEATURES[:]   = list(feat._ORIGINAL_SUPERVISED_FEATURES)
    feat.INTERSECTING_FEATURES[:] = list(feat._ORIGINAL_INTERSECTING_FEATURES)
    feat.UNSUPERVISED_FEATURES[:] = list(feat._ORIGINAL_UNSUPERVISED_FEATURES)
    feat.WINDOW_LEN = feat._ORIGINAL_WINDOW_LEN

    if window_size:
        print(f"[Ablation] Overriding WINDOW_LEN → {window_size}")
        feat.WINDOW_LEN = window_size

    if exclude_features:
        excluded = [f.strip() for f in exclude_features.split(",") if f.strip()]
        print(f"[Ablation] Excluding features: {excluded}")
        feat.SUPERVISED_FEATURES[:]   = [f for f in feat.SUPERVISED_FEATURES   if f not in excluded]
        feat.INTERSECTING_FEATURES[:] = [f for f in feat.INTERSECTING_FEATURES if f not in excluded]
        feat.UNSUPERVISED_FEATURES[:] = [f for f in feat.UNSUPERVISED_FEATURES if f not in excluded]

    if project_root:
        ephemeral = project_root / "implement" / "dataset" / "ephermal_dataset_supervised"
        if ephemeral.exists():
            print(f"[Startup] Clearing ephemeral cache to prevent contamination: {ephemeral}")
            shutil.rmtree(ephemeral)


def get_ablation_suffix(window_size=None, exclude_features=None):
    """Returns a compact string suffix for output directory naming."""
    INITIALS = {
        'prediction_error': 'pe', 'ground_speed': 'gs',
        'vertical_acceleration': 'va', 'vertical_speed': 'vs',
        'acceleration': 'acc', 'turn_rate': 'tr',
        'path_curvature': 'pc', 'jerk': 'j',
        'heading_speed_consistency': 'hsc', 'motion_smoothness': 'ms',
    }
    suffix = ""
    if window_size:
        suffix += f"_w{window_size}"
    if exclude_features:
        excluded = [f.strip() for f in exclude_features.split(",") if f.strip()]
        initials = [INITIALS.get(f, "".join(w[0] for w in f.split("_"))) for f in excluded]
        suffix += "_ex_" + "_".join(initials)
    return suffix

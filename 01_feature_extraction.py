"""
Spatial Enhancement Heterogeneity and Peritumoral Microenvironment Analysis
============================================================================
Version 4.0 - Methodological Revision

Changes from v3.0:
- P0-D: Structured error logging; pipeline cannot complete without error summary
- P0-D: validate_dataset() fail-fast before any processing
- P0-B: Unified geometry layer — spatial ops in SimpleITK, NumPy only for stats
- P0-B: spacing=(1,1,1) hardcode removed everywhere
- P0-E: Delta features removed (no registration available)
- Capsule search relocated to external shell (P1-B, partial)
- Local heterogeneity converted to 3D (P1-C)
- Gradient anisotropy corrected for physical spacing (P1-D)
- Multifocality uses authority-phase rule C1>C2>C3>P (P1-F)
"""

import os
import sys
import json
import traceback
import logging
import platform
import importlib
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats
from skimage.measure import label as sk_label
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    BASE_DIR = "./data"  # set this to your local data directory
    CT_DIR   = f"{BASE_DIR}/ct_files"
    MASK_DIR = f"{BASE_DIR}/mask_files"
    CLINICAL_PATH = f"{BASE_DIR}/patient_data.csv"
    OUTPUT_DIR    = f"{BASE_DIR}/spatial_analysis_v4"

    PHASES = ['P', 'C1', 'C2', 'C3']
    PHASE_NAMES = {'P': 'Pre-contrast', 'C1': 'Arterial',
                   'C2': 'Portal Venous', 'C3': 'Delayed'}

    # Capsule detection only in these phases
    CAPSULE_PHASES = ['C2', 'C3']

    CLASSES = ['HCC', 'ICC', 'CHCC']
    CLASS_COLORS = {'HCC': '#e74c3c', 'ICC': '#3498db', 'CHCC': '#9b59b6'}

    # Spatial parameters
    PERITUMORAL_DISTANCES = [5, 10]   # mm
    N_RADIAL_BINS  = 5
    N_ANGULAR_SECTORS = 8

    # Parallel processing
    # 'threading' avoids subprocess-spawning issues with SimpleITK loggers.
    # Increase N_JOBS for more CPU cores; decrease if memory is constrained.
    N_JOBS  = 4
    BACKEND = 'threading'

    # Necrosis: hybrid rule (P1-A)
    # HU band calibrated for clip-space [-200, 200]; range [10, 40] fully preserved.
    # Hyperenhancement cap at 200 HU is a known clip artefact — documented in Methods.
    # Enhancement threshold: voxels with contrast uptake < 15 HU vs pre-contrast
    # are considered non-enhancing (necrotic). Literature ref: Fowler et al. 2018.
    NECROSIS_HU_MIN      = 10
    NECROSIS_HU_MAX      = 40
    NECROSIS_ENH_MAX_HU  = 15   # max enhancement vs pre-contrast for necrosis label

    # Capsule
    CAPSULE_SHELL_MM        = 3.0   # external shell width (mm)
    CAPSULE_HU_DIFF         = 15    # min HU elevation of shell vs core
    CAPSULE_ENH_RATIO       = 1.15  # min shell/core enhancement ratio
    # Minimum fraction of shell voxels that must exceed the HU threshold
    # for the ring to be considered morphologically continuous.
    # Stored here (not hardcoded) so it can be reported in Methods.
    CAPSULE_CONTINUITY_MIN  = 0.30

    # Feature selection (used in external FeatureSelector, kept here for reference)
    CORRELATION_THRESHOLD  = 0.90
    MIN_VARIANCE_THRESHOLD = 0.01

    # Multifocality authority-phase order
    AUTHORITY_PHASE_ORDER = ['C1', 'C2', 'C3', 'P']

    RANDOM_STATE = 42

    # P3-A: Scanner harmonization
    # Column name in patient_data.csv that identifies the acquisition device.
    # Set to None to skip harmonization.
    SCANNER_COL = 'scanner'          # e.g. 'Canon' / 'Philips' per dataset paper
    APPLY_SCANNER_HARMONIZATION = False   # toggle; requires neuroCombat if True



# ============================================================
# P0-D: STRUCTURED ERROR LOGGING
# ============================================================
@dataclass
class ErrorRecord:
    patient_id:    str
    phase:         str
    feature_block: str
    error_type:    str
    message:       str
    traceback_short: str


class AnalysisLogger:
    """
    Central error + event logger for the pipeline.
    Pipeline must call summarize() at the end; raises if errors exist.
    """

    def __init__(self, output_dir: str):
        self._errors: list[ErrorRecord] = []
        self._output_dir = output_dir

        log_path = os.path.join(output_dir, f"run_log_{datetime.now():%Y%m%d_%H%M%S}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                      logging.StreamHandler()],
            force=True,
        )
        self._logger = logging.getLogger("pipeline")

    def info(self, msg: str, *args):
        self._logger.info(msg, *args)

    def record_error(self, patient_id: str, phase: str,
                     feature_block: str, exc: Exception):
        tb = traceback.format_exc().strip().split('\n')
        tb_short = ' | '.join(tb[-3:]) if len(tb) >= 3 else ' | '.join(tb)
        rec = ErrorRecord(
            patient_id=patient_id,
            phase=phase,
            feature_block=feature_block,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback_short=tb_short,
        )
        self._errors.append(rec)
        self._logger.warning(
            f"ERROR patient={patient_id} phase={phase} "
            f"block={feature_block} [{type(exc).__name__}]: {exc}"
        )

    def summarize(self, raise_on_errors: bool = False):
        """Write error CSV. If raise_on_errors and errors exist, raise RuntimeError."""
        if not self._errors:
            self._logger.info("Run completed — no errors recorded.")
            return

        df = pd.DataFrame([vars(r) for r in self._errors])
        csv_path = os.path.join(self._output_dir, "error_summary.csv")
        df.to_csv(csv_path, index=False)

        summary = df.groupby(['error_type', 'feature_block']).size().reset_index(name='count')
        self._logger.warning(
            f"\n{'='*60}\nERROR SUMMARY ({len(self._errors)} total)\n{summary.to_string()}"
            f"\nFull log: {csv_path}\n{'='*60}"
        )

        if raise_on_errors:
            raise RuntimeError(
                f"Pipeline completed with {len(self._errors)} errors. "
                f"See {csv_path} for details."
            )


# ============================================================
# P0-D: DATASET VALIDATION (fail-fast before any processing)
# ============================================================
# ============================================================
# P3-B: REPRODUCIBILITY — run config snapshot
# ============================================================
def save_run_config(config: Config, output_dir: str):
    """
    Save a JSON snapshot of Config + library versions + Python/platform info.
    Written at the start of every run so results are reproducible.
    """
    def _ver(mod_name: str) -> str:
        try:
            return importlib.import_module(mod_name).__version__
        except Exception:
            return 'unknown'

    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot = {
        'run_id':   run_id,
        'timestamp': datetime.now().isoformat(),
        'python':   sys.version,
        'platform': platform.platform(),
        'libraries': {
            'numpy':     _ver('numpy'),
            'pandas':    _ver('pandas'),
            'SimpleITK': _ver('SimpleITK'),
            'scipy':     _ver('scipy'),
            'scikit-image': _ver('skimage'),
            'joblib':    _ver('joblib'),
            'matplotlib': _ver('matplotlib'),
        },
        'config': {k: v for k, v in vars(config.__class__).items()
                   if not k.startswith('_') and not callable(v)},
    }

    out_path = os.path.join(output_dir, f"run_config_{run_id}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=str)

    return run_id, out_path


# ============================================================
# P3-A: SCANNER HARMONIZATION (feature-level)
# ============================================================
def apply_scanner_harmonization(combined_df: pd.DataFrame,
                                  scanner_col: str,
                                  logger: AnalysisLogger,
                                  output_dir: str) -> pd.DataFrame:
    """
    Feature-level scanner harmonization using z-score per scanner batch
    (a lightweight alternative to ComBat when neuroCombat is unavailable).

    For each numeric feature, within each scanner group the values are
    centred and scaled to unit variance, then rescaled to the overall
    feature mean/std so the global distribution is preserved.

    If neuroCombat is installed and preferred, swap the inner block with:
        from neuroCombat import neuroCombat
        data = neuroCombat(dat, covars, batch_col=scanner_col, ...)

    The original (pre-harmonization) features are saved separately so the
    effect can be compared (Methods requirement).
    """
    if scanner_col not in combined_df.columns:
        logger.info(
            "Scanner harmonization skipped: column '%s' not found in data.",
            scanner_col
        )
        return combined_df

    scanners = combined_df[scanner_col].unique()
    logger.info(
        "Scanner harmonization: %d scanners found: %s",
        len(scanners), list(scanners)
    )

    exclude_cols = {'patient_id', 'cancer_type', scanner_col}
    numeric_cols = [
        c for c in combined_df.select_dtypes(include=[float, int]).columns
        if c not in exclude_cols
    ]

    # Save pre-harmonization snapshot
    pre_path = os.path.join(output_dir, "combined_spatial_v4_pre_harmonization.csv")
    combined_df.to_csv(pre_path, index=False)
    logger.info("Pre-harmonization features saved: %s", pre_path)

    df_harm = combined_df.copy()
    global_means = df_harm[numeric_cols].mean()
    global_stds  = df_harm[numeric_cols].std().replace(0, 1)

    for scanner in scanners:
        idx = df_harm[scanner_col] == scanner
        subset = df_harm.loc[idx, numeric_cols]

        batch_means = subset.mean()
        batch_stds  = subset.std().replace(0, 1)

        # Remove batch effect: standardise within scanner, rescale to global
        df_harm.loc[idx, numeric_cols] = (
            (subset - batch_means) / batch_stds
        ) * global_stds + global_means

    n_scanners = combined_df[scanner_col].value_counts().to_dict()
    logger.info("Harmonization complete. Scanner counts: %s", n_scanners)

    return df_harm


# ============================================================
# P0-D: DATASET VALIDATION (fail-fast before any processing)
# ============================================================
def validate_dataset(config: Config, logger: AnalysisLogger) -> tuple[pd.DataFrame, dict]:
    """
    Load CSV, verify every referenced file exists and is a readable NIfTI.
    Also propagates scanner label (P3-A) if the column exists in CSV.
    Returns filtered DataFrame with only patients that have all 4 CT phases.
    Patients with missing/unreadable files are logged and excluded.
    """
    logger.info("=== DATASET VALIDATION ===")
    df = pd.read_csv(config.CLINICAL_PATH)
    df['patient_id'] = df['patient_id'].astype(str).str.strip()

    # Keep only cancer classes
    df = df[df['cancer_type'].isin(config.CLASSES)].copy()
    logger.info("Rows in CSV (cancer classes, pre-dedup): %d", len(df))

    # Build liver mask lookup from phase-level rows BEFORE collapsing to patient level.
    # CSV liver_mask_path is relative to BASE_DIR; convert to absolute here.
    # This lookup is passed to PeritumoralAnalyzer so TLR always uses the CSV-declared
    # path rather than guessing by filename pattern (which uses a different convention).
    liver_mask_lookup: dict = {}
    if 'liver_mask_path' in df.columns and 'phase' in df.columns:
        for _, row in df.iterrows():
            lm_rel = row.get('liver_mask_path', '')
            ph     = row.get('phase', '')
            pid    = row['patient_id']
            if pd.notna(lm_rel) and str(lm_rel).strip():
                abs_path = os.path.join(config.BASE_DIR, str(lm_rel).strip())
                if os.path.exists(abs_path):
                    liver_mask_lookup[(pid, ph)] = abs_path
        logger.info("Liver mask lookup built: %d phase entries for TLR", len(liver_mask_lookup))

    # Collapse to one row per patient.
    # Sort by phase authority order so .first() always picks the same phase row
    # regardless of CSV row order (C1 → C2 → C3 → P, matching AUTHORITY_PHASE_ORDER).
    if 'phase' in df.columns:
        _phase_rank = {ph: i for i, ph in enumerate(config.AUTHORITY_PHASE_ORDER)}
        df['_phase_rank'] = df['phase'].map(_phase_rank).fillna(99).astype(int)
        df = (
            df.sort_values(['patient_id', '_phase_rank'])
              .groupby('patient_id', as_index=False)
              .first()
              .drop(columns=['_phase_rank'])
        )
    else:
        df = (
            df.sort_values('patient_id')
              .groupby('patient_id', as_index=False)
              .first()
        )
    logger.info("Unique patients after dedup: %d", len(df))

    # P3-A: log scanner distribution if column exists
    if config.SCANNER_COL and config.SCANNER_COL in df.columns:
        logger.info("Scanner distribution:\n%s",
                    df[config.SCANNER_COL].value_counts().to_string())
    elif config.SCANNER_COL:
        logger.info(
            "Scanner column '%s' not found in CSV — "
            "harmonization will be skipped.", config.SCANNER_COL
        )

    valid_rows = []
    invalid_count = 0
    total_patients = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10 == 0:
            print(f"[Validate] {i}/{total_patients} patients checked...", flush=True)
        pid = row['patient_id']
        phase_files = _find_phase_files(pid, config)

        missing_phases = [p for p in config.PHASES if p not in phase_files]
        if missing_phases:
            logger.record_error(
                pid, str(missing_phases), 'validate_dataset',
                FileNotFoundError(f"Missing phases: {missing_phases}")
            )
            invalid_count += 1
            continue

        # Integrity check 1: header readable + Integrity check 2: CT-mask geometry
        readable = True
        phase_meta: dict = {}   # phase -> (ct_size, ct_spacing)

        for phase, paths in phase_files.items():
            ct_reader   = sitk.ImageFileReader()
            mask_reader = sitk.ImageFileReader()
            try:
                ct_reader.SetFileName(paths['ct'])
                ct_reader.ReadImageInformation()
                mask_reader.SetFileName(paths['mask'])
                mask_reader.ReadImageInformation()
            except Exception as exc:
                logger.record_error(pid, phase, 'validate_dataset:header', exc)
                readable = False
                break

            ct_size    = ct_reader.GetSize()
            mask_size  = mask_reader.GetSize()
            ct_spacing = ct_reader.GetSpacing()

            # Size mismatch is non-fatal (load_sitk_pair will resample),
            # but log it so systematic mismatches are visible.
            if ct_size != mask_size:
                logger.info(
                    "Size mismatch pid=%s phase=%s ct=%s mask=%s — "
                    "will resample at load time.",
                    pid, phase, ct_size, mask_size
                )

            phase_meta[phase] = (ct_size, ct_spacing)

        # Cross-phase spacing consistency: warn if Z-spacing varies > 20%
        # across phases for the same patient (indicates different acquisitions).
        if readable and len(phase_meta) > 1:
            z_spacings = [sp[1][2] for sp in phase_meta.values()]
            z_mean = float(np.mean(z_spacings))
            if z_mean > 0:
                z_cv = float(np.std(z_spacings) / z_mean)
                if z_cv > 0.20:
                    logger.info(
                        "Cross-phase Z-spacing variation pid=%s cv=%.2f %s — "
                        "check acquisition consistency.",
                        pid, z_cv, z_spacings
                    )

        if readable:
            valid_rows.append(row)
        else:
            invalid_count += 1

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    logger.info(
        f"Validation complete: {len(valid_df)} valid, {invalid_count} excluded. "
        f"See error log for details."
    )
    return valid_df, liver_mask_lookup


def _find_phase_files(patient_id: str, config: Config) -> dict:
    """
    Return {phase: {ct: path, mask: path}} for phases that have
    BOTH a phase-specific CT AND a phase-specific mask.
    No fallback masks — missing mask = missing phase.
    """
    files = {}
    for phase in config.PHASES:
        ct_candidates = [
            os.path.join(config.CT_DIR, f"{patient_id}_ct_{phase}.nii.gz"),
            os.path.join(config.CT_DIR, f"{patient_id}_{phase}.nii.gz"),
            os.path.join(config.CT_DIR, f"{patient_id}_ct_{phase}.nii"),
        ]
        mask_candidates = [
            os.path.join(config.MASK_DIR, f"{patient_id}_mask_{phase}.nii.gz"),
            os.path.join(config.MASK_DIR, f"{patient_id}_mask_{phase}.nii"),
        ]
        # NO generic fallback mask — every phase needs its own mask

        ct_path   = next((p for p in ct_candidates   if os.path.exists(p)), None)
        mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

        if ct_path and mask_path:
            files[phase] = {'ct': ct_path, 'mask': mask_path}

    return files


# ============================================================
# P0-B: UNIFIED GEOMETRY LAYER
# ============================================================
# Rule: every spatial operation (distance map, gradient, morphology,
#       resample) is performed on sitk.Image objects.
#       NumPy arrays are used ONLY for summary statistics (mean, std, etc.)
#       and are obtained via sitk.GetArrayFromImage() as late as possible.
#
# Spacing convention:
#   sitk uses (x, y, z) order.
#   sitk.GetArrayFromImage() returns array in (z, y, x) order.
#   All helper functions that accept a sitk.Image work in sitk space.
#   When a helper must accept an ndarray (for stats), it also receives
#   the sitk.Image to extract spacing — never a raw tuple.

def get_spacing_zyx(sitk_image: sitk.Image) -> tuple[float, float, float]:
    """Return spacing in (z, y, x) order matching numpy array axes."""
    sx, sy, sz = sitk_image.GetSpacing()
    return (sz, sy, sx)


def sitk_distance_map_inside(mask_sitk: sitk.Image) -> sitk.Image:
    """
    Signed distance map inside the mask (positive = inside, in mm).
    Uses Maurer distance map for physical-space correctness.
    """
    mask_uint = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)
    dist = sitk.SignedMaurerDistanceMap(
        mask_uint,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return dist


def sitk_distance_map_outside(mask_sitk: sitk.Image) -> sitk.Image:
    """
    Distance map outside the mask (positive = outside, in mm).
    """
    mask_uint = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)
    inverted = sitk.Cast(mask_uint == 0, sitk.sitkUInt8)
    dist = sitk.SignedMaurerDistanceMap(
        inverted,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return dist


def sitk_gradient_magnitude_physical(ct_sitk: sitk.Image) -> sitk.Image:
    """
    Gradient magnitude corrected for physical voxel spacing (P1-D).
    Each axis derivative is divided by that axis's spacing before
    combining into magnitude — equivalent to a true physical derivative.
    Uses GradientMagnitudeRecursiveGaussian with very small sigma
    (0.5 mm) to stay close to a finite-difference derivative while
    keeping numerical stability.
    """
    ct_float = sitk.Cast(ct_sitk, sitk.sitkFloat64)
    grad_mag = sitk.GradientMagnitudeRecursiveGaussian(ct_float, sigma=0.5)
    return grad_mag


def load_sitk_pair(ct_path: str, mask_path: str) -> tuple[sitk.Image, sitk.Image]:
    """
    Load CT and mask as SimpleITK images.
    Resamples mask to CT space (nearest-neighbor) if sizes differ.
    Returns (ct_sitk, mask_sitk).
    """
    ct_sitk   = sitk.ReadImage(ct_path,   sitk.sitkFloat64)
    mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    if ct_sitk.GetSize() != mask_sitk.GetSize():
        mask_sitk = sitk.Resample(
            mask_sitk, ct_sitk,
            sitk.Transform(), sitk.sitkNearestNeighbor, 0,
            mask_sitk.GetPixelID()
        )
    return ct_sitk, mask_sitk


def sitk_peritumoral_mask(mask_sitk: sitk.Image,
                           distance_mm: float) -> sitk.Image:
    """
    Binary mask of the peritumoral shell: voxels outside the tumor
    within [0, distance_mm] mm.
    """
    dist_out = sitk_distance_map_outside(mask_sitk)
    peri = (dist_out > 0) & (dist_out <= distance_mm)
    return sitk.Cast(peri, sitk.sitkUInt8)


def sitk_external_shell(mask_sitk: sitk.Image,
                         inner_mm: float,
                         outer_mm: float) -> sitk.Image:
    """External annular shell: [inner_mm, outer_mm] outside tumor surface."""
    dist_out = sitk_distance_map_outside(mask_sitk)
    shell = (dist_out >= inner_mm) & (dist_out <= outer_mm)
    return sitk.Cast(shell, sitk.sitkUInt8)


def sitk_radial_zone_map(mask_sitk: sitk.Image,
                          n_bins: int = 5) -> Optional[sitk.Image]:
    """
    Radial zone map based on distance from tumor boundary inward (P0-B fix).
    Zone 0 = outermost (near boundary), zone n_bins-1 = deepest core.
    Returns sitk.Image of int32 zone labels (-1 outside tumor).
    """
    dist_inside = sitk_distance_map_inside(mask_sitk)
    dist_arr    = sitk.GetArrayFromImage(dist_inside)
    mask_arr    = sitk.GetArrayFromImage(mask_sitk)

    inside = dist_arr > 0
    if inside.sum() == 0:
        return None

    max_dist = dist_arr[inside].max()
    if max_dist < 1e-3:
        return None

    normalized = np.where(inside, dist_arr / max_dist, 0.0)
    bins = np.linspace(0, 1, n_bins + 1)
    zone_arr = np.full_like(mask_arr, -1, dtype=np.int32)
    zone_arr[inside] = np.digitize(normalized[inside], bins) - 1
    zone_arr[inside] = np.clip(zone_arr[inside], 0, n_bins - 1)

    zone_sitk = sitk.GetImageFromArray(zone_arr)
    zone_sitk.CopyInformation(mask_sitk)
    return zone_sitk


# ============================================================
# UTILITY: SUMMARY STATISTICS (NumPy only)
# ============================================================
def compute_entropy(values: np.ndarray, bins: int = 32) -> float:
    if len(values) == 0:
        return np.nan
    hist, _ = np.histogram(values, bins=bins)
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0


def compute_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def robust_stats(values: np.ndarray) -> dict:
    """Core robust summary: median, IQR, MAD, entropy, skewness, kurtosis."""
    if len(values) == 0:
        return {}
    return {
        'median':   float(np.median(values)),
        'iqr':      float(np.percentile(values, 75) - np.percentile(values, 25)),
        'mad':      compute_mad(values),
        'entropy':  compute_entropy(values),
        'skewness': float(stats.skew(values)),
        'kurtosis': float(stats.kurtosis(values)),
        'p10':      float(np.percentile(values, 10)),
        'p90':      float(np.percentile(values, 90)),
    }


# ============================================================
# INTRATUMORAL HETEROGENEITY ANALYZER (v4.0)
# ============================================================
class IntratumoralHeterogeneityAnalyzer:

    def __init__(self, config: Config, logger: AnalysisLogger,
                 valid_df: pd.DataFrame):
        self.config   = config
        self.logger   = logger
        self.patients = valid_df

    def _compute_radial_profile(self, ct_sitk: sitk.Image,
                                 mask_sitk: sitk.Image) -> Optional[dict]:
        zone_sitk = sitk_radial_zone_map(mask_sitk, self.config.N_RADIAL_BINS)
        if zone_sitk is None:
            return None

        ct_arr   = sitk.GetArrayFromImage(ct_sitk)
        zone_arr = sitk.GetArrayFromImage(zone_sitk)

        profile = {}
        zone_medians = []

        for z in range(self.config.N_RADIAL_BINS):
            vox = ct_arr[zone_arr == z]
            if len(vox) == 0:
                zone_medians.append(np.nan)
                continue
            profile[f'zone_{z}_median'] = float(np.median(vox))
            profile[f'zone_{z}_mad']    = compute_mad(vox)
            zone_medians.append(profile[f'zone_{z}_median'])

        valid = [(i, m) for i, m in enumerate(zone_medians) if not np.isnan(m)]
        if len(valid) >= 2:
            x_vals = [v[0] for v in valid]
            y_vals = [v[1] for v in valid]
            profile['radial_gradient'] = float(y_vals[-1] - y_vals[0])
            slope, _, r, _, _ = stats.linregress(x_vals, y_vals)
            profile['radial_slope']     = float(slope)
            profile['radial_r_squared'] = float(r ** 2)

        return profile

    def _compute_ring_enhancement(self, ct_sitk: sitk.Image,
                                   mask_sitk: sitk.Image) -> Optional[dict]:
        dist_inside = sitk_distance_map_inside(mask_sitk)
        dist_arr    = sitk.GetArrayFromImage(dist_inside)
        max_dist    = dist_arr.max()

        if max_dist < 3:
            return None

        ct_arr   = sitk.GetArrayFromImage(ct_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        core_mask = (mask_arr > 0) & (dist_arr > max_dist * 0.5)
        rim_mask  = (mask_arr > 0) & (dist_arr <= max_dist * 0.2)

        if core_mask.sum() == 0 or rim_mask.sum() == 0:
            return None

        core_vals = ct_arr[core_mask]
        rim_vals  = ct_arr[rim_mask]

        core_med = float(np.median(core_vals))
        rim_med  = float(np.median(rim_vals))

        return {
            'core_median':     core_med,
            'rim_median':      rim_med,
            'rim_core_diff':   float(rim_med - core_med),
            'rim_core_ratio':  float(rim_med / (core_med + 1e-5)),
            'ring_enhancement': 1 if rim_med > core_med * 1.1 else 0,
            'core_mad':        compute_mad(core_vals),
            'rim_mad':         compute_mad(rim_vals),
            'core_entropy':    compute_entropy(core_vals),
            'rim_entropy':     compute_entropy(rim_vals),
        }

    def _compute_angular_heterogeneity_3d(self, ct_sitk: sitk.Image,
                                          mask_sitk: sitk.Image) -> Optional[dict]:
        """
        True 3D angular heterogeneity using spherical coordinates in physical space.
        Physical voxel spacing is applied before angle computation so results are
        invariant to anisotropic sampling. Azimuthal (φ) and polar (θ) angles are
        binned independently; only sectors with ≥3 voxels contribute.
        """
        mask_arr   = sitk.GetArrayFromImage(mask_sitk)
        ct_arr     = sitk.GetArrayFromImage(ct_sitk)
        sz, sy, sx = get_spacing_zyx(ct_sitk)

        coords = np.argwhere(mask_arr > 0)
        if len(coords) == 0:
            return None

        # Convert voxel indices to physical mm positions
        coords_mm = coords * np.array([sz, sy, sx])
        center_mm = coords_mm.mean(axis=0)
        rel_mm    = coords_mm - center_mm

        r     = np.linalg.norm(rel_mm, axis=1)
        r_max = r.max()
        if r_max < 1e-3:
            return None

        # Exclude voxels within 10% of mass centre (angles numerically unstable)
        far = r > 0.1 * r_max
        if far.sum() < 10:
            return None

        rel_far    = rel_mm[far]
        coords_far = coords[far]
        r_far      = r[far]

        # Spherical coordinates (array axis order z,y,x → z=axial)
        phi   = np.arctan2(rel_far[:, 1], rel_far[:, 2])               # azimuthal -π…π
        theta = np.arccos(np.clip(rel_far[:, 0] / r_far, -1.0, 1.0))  # polar      0…π

        n_azi      = self.config.N_ANGULAR_SECTORS
        n_pol      = 2   # superior / inferior hemispheres
        phi_bins   = np.linspace(-np.pi, np.pi, n_azi + 1)
        theta_bins = np.array([0.0, np.pi / 2.0, np.pi])

        azi_idx = np.clip(np.digitize(phi,   phi_bins)   - 1, 0, n_azi - 1)
        pol_idx = np.clip(np.digitize(theta, theta_bins) - 1, 0, n_pol - 1)

        sector_medians = []
        for a in range(n_azi):
            for p in range(n_pol):
                sel = coords_far[(azi_idx == a) & (pol_idx == p)]
                if len(sel) < 3:
                    continue
                vals = ct_arr[sel[:, 0], sel[:, 1], sel[:, 2]]
                sector_medians.append(float(np.median(vals)))

        if len(sector_medians) < 4:
            return None

        sm = np.array(sector_medians)
        return {
            'angular_median_cv': float(np.std(sm) / (np.mean(sm) + 1e-5)),
            'angular_range':     float(sm.max() - sm.min()),
            'angular_entropy':   float(stats.entropy(sm - sm.min() + 1)),
            'angular_mad':       compute_mad(sm),
            'n_valid_sectors':   len(sm),
        }

    def _compute_local_heterogeneity_3d(self, ct_sitk: sitk.Image,
                                         mask_sitk: sitk.Image,
                                         kernel_mm: float = 5.0) -> Optional[dict]:
        """
        P1-C: true 3D local heterogeneity.
        Kernel size derived from physical spacing, not slice index.
        Uses sitk.Statistics on a local neighbourhood via recursive Gaussian.
        Approximation: local std ≈ sqrt(E[x²] - E[x]²) computed with
        sitk.SmoothingRecursiveGaussian on x and x².
        """
        ct_float  = sitk.Cast(ct_sitk, sitk.sitkFloat64)
        mask_arr  = sitk.GetArrayFromImage(mask_sitk)

        if (mask_arr > 0).sum() == 0:
            return None

        # sigma in mm (converts to voxels internally via image spacing)
        sigma = kernel_mm / 2.355  # FWHM to sigma

        ct_smooth    = sitk.SmoothingRecursiveGaussian(ct_float, sigma)
        ct_sq        = sitk.Pow(ct_float, 2)
        ct_sq_smooth = sitk.SmoothingRecursiveGaussian(ct_sq, sigma)

        mean_arr  = sitk.GetArrayFromImage(ct_smooth)
        sq_arr    = sitk.GetArrayFromImage(ct_sq_smooth)

        local_var = np.maximum(sq_arr - mean_arr ** 2, 0)
        local_std = np.sqrt(local_var)

        tumor_std = local_std[mask_arr > 0]
        tumor_std = tumor_std[~np.isnan(tumor_std)]

        if len(tumor_std) == 0:
            return None

        result = {f'local_hetero_{k}': v
                  for k, v in robust_stats(tumor_std).items()}
        result['high_hetero_fraction'] = float(
            np.sum(tumor_std > np.percentile(tumor_std, 75)) / len(tumor_std)
        )
        return result

    def _compute_necrosis_features(self, ct_sitk: sitk.Image,
                                    mask_sitk: sitk.Image,
                                    phase: str,
                                    precontrast_sitk: Optional[sitk.Image] = None,
                                    ) -> Optional[dict]:
        """
        P1-A: Hybrid necrosis rule.

        A voxel is necrotic when ALL of these hold:
          1. HU in [NECROSIS_HU_MIN, NECROSIS_HU_MAX]  (absolute band, clip-space)
          2. Enhancement vs pre-contrast < NECROSIS_ENH_MAX_HU
             (low contrast uptake = no viable tissue)
          3. Inside the tumour mask

        When pre-contrast is unavailable, rule 2 is skipped and
        'necrosis_qc_method' is set to 'hu_band_only' so the caller
        can flag those cases.

        Pre-contrast is resampled to the target phase space (linear
        interpolation + intensity clamp) before subtraction so physical
        voxel correspondence is guaranteed.

        Data are clipped to [-200, 200] HU. The necrosis band [10, 40]
        is fully preserved in this range. Enhancement values above 200
        are capped — documented in Methods as a known acquisition
        artefact of the dataset.
        """
        if phase == 'P':
            return None

        ct_arr   = sitk.GetArrayFromImage(ct_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        tumor_vox = ct_arr[mask_arr > 0]
        if len(tumor_vox) == 0:
            return None

        # ── absolute HU band (rule 1) ────────────────────────
        hu_band_mask = (
            (ct_arr >= self.config.NECROSIS_HU_MIN) &
            (ct_arr <= self.config.NECROSIS_HU_MAX) &
            (mask_arr > 0)
        )

        # ── pre-contrast enhancement check (rule 2) ──────────
        qc_method = 'hu_band_only'
        if precontrast_sitk is not None:
            try:
                # Resample pre-contrast to target phase space (P1-A)
                pre_resampled = sitk.Resample(
                    sitk.Cast(precontrast_sitk, sitk.sitkFloat64),
                    ct_sitk,
                    sitk.Transform(),
                    sitk.sitkLinear,
                    0.0,
                    sitk.sitkFloat64,
                )
                # Clamp resampled values to dataset clip range
                pre_arr = sitk.GetArrayFromImage(pre_resampled)
                pre_arr = np.clip(pre_arr, -200.0, 200.0)

                enhancement = ct_arr - pre_arr   # positive = enhancement
                low_enh_mask = enhancement < self.config.NECROSIS_ENH_MAX_HU

                necrosis_mask = hu_band_mask & low_enh_mask
                qc_method = 'hybrid_hu_band_plus_enhancement'
            except Exception:
                # Pre-contrast resample failed — fall back gracefully
                necrosis_mask = hu_band_mask
                qc_method = 'hu_band_only_resample_failed'
        else:
            necrosis_mask = hu_band_mask

        tumor_n    = int((mask_arr > 0).sum())
        necrosis_n = int(necrosis_mask.sum())
        frac       = necrosis_n / tumor_n if tumor_n > 0 else 0.0

        feats = {
            'necrosis_fraction':  float(frac),
            'necrosis_qc_method': qc_method,
        }

        # Centrality using physical distance map (P0-B: no hardcode spacing)
        if necrosis_n > 0:
            dist_inside = sitk_distance_map_inside(mask_sitk)
            dist_arr    = sitk.GetArrayFromImage(dist_inside)
            max_dist    = dist_arr[mask_arr > 0].max()
            if max_dist > 0:
                necrosis_dists = dist_arr[necrosis_mask]
                feats['necrosis_centrality'] = float(necrosis_dists.mean() / max_dist)

        return feats

    def _compute_enhancement_stats(self, ct_sitk: sitk.Image,
                                    mask_sitk: sitk.Image) -> Optional[dict]:
        ct_arr   = sitk.GetArrayFromImage(ct_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)
        vals     = ct_arr[mask_arr > 0]
        if len(vals) == 0:
            return None
        # P2-A: single robust summary set; no redundant mean+median+trimmed_mean
        return {f'enh_{k}': v for k, v in robust_stats(vals).items()}

    def analyze_patient(self, patient_id: str, cancer_type: str) -> Optional[dict]:
        files = _find_phase_files(patient_id, self.config)
        if len(files) < 4:
            return None

        results = {'patient_id': patient_id, 'cancer_type': cancer_type}

        # P1-F: authority-phase multifocality
        for auth_phase in self.config.AUTHORITY_PHASE_ORDER:
            if auth_phase in files:
                try:
                    _, mask_auth = load_sitk_pair(
                        files[auth_phase]['ct'], files[auth_phase]['mask']
                    )
                    mask_arr = sitk.GetArrayFromImage(mask_auth)
                    labeled, n = sk_label(mask_arr > 0, return_num=True)
                    results['is_multifocal']       = int(n > 1)
                    results['n_lesions']            = int(n)
                    results['multifocal_auth_phase'] = auth_phase
                    if n > 1:
                        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
                        results['largest_lesion_fraction'] = float(max(sizes) / sum(sizes))
                    else:
                        results['largest_lesion_fraction'] = 1.0
                except Exception as exc:
                    self.logger.record_error(patient_id, auth_phase,
                                             'multifocality', exc)
                break

        # P1-A: load pre-contrast once; reused by necrosis for all contrast phases
        precontrast_sitk: Optional[sitk.Image] = None
        if 'P' in files:
            try:
                precontrast_sitk, _ = load_sitk_pair(
                    files['P']['ct'], files['P']['mask']
                )
            except Exception as exc:
                self.logger.record_error(patient_id, 'P', 'load_precontrast', exc)

        for phase in self.config.PHASES:
            if phase not in files:
                continue

            try:
                ct_sitk, mask_sitk = load_sitk_pair(
                    files[phase]['ct'], files[phase]['mask']
                )
            except Exception as exc:
                self.logger.record_error(patient_id, phase, 'load', exc)
                continue

            # Capture loop variables for lambdas (avoids late-binding closure bug)
            _ct, _mask, _phase = ct_sitk, mask_sitk, phase
            _pre = precontrast_sitk

            blocks = {
                'radial_profile':    lambda c=_ct, m=_mask: self._compute_radial_profile(c, m),
                'ring_enhancement':  lambda c=_ct, m=_mask: self._compute_ring_enhancement(c, m),
                'angular_hetero':    lambda c=_ct, m=_mask: self._compute_angular_heterogeneity_3d(c, m),
                'local_hetero_3d':   lambda c=_ct, m=_mask: self._compute_local_heterogeneity_3d(c, m),
                'necrosis':          lambda c=_ct, m=_mask, p=_phase, pre=_pre: (
                                         self._compute_necrosis_features(c, m, p, pre)
                                     ),
                'enhancement_stats': lambda c=_ct, m=_mask: self._compute_enhancement_stats(c, m),
            }

            for block_name, fn in blocks.items():
                try:
                    feats = fn()
                    if feats:
                        for k, v in feats.items():
                            results[f'{_phase}_{k}'] = v
                except Exception as exc:
                    self.logger.record_error(patient_id, phase, block_name, exc)

        return results if len(results) > 4 else None

    def run_analysis(self) -> pd.DataFrame:
        self.logger.info("=== INTRATUMORAL HETEROGENEITY ANALYSIS (v4.0) ===")
        self.logger.info(f"Patients: {len(self.patients)}, Workers: {self.config.N_JOBS}")

        checkpoint_path = os.path.join(self.config.OUTPUT_DIR,
                                        "checkpoint_intratumoral.csv")
        processed, existing = _load_checkpoint(checkpoint_path)

        patient_list = [
            (r['patient_id'], r['cancer_type'])
            for _, r in self.patients.iterrows()
            if str(r['patient_id']) not in processed
        ]
        self.logger.info(f"Remaining: {len(patient_list)}")

        if not patient_list:
            self.results_df = pd.DataFrame(existing)
            return self.results_df

        all_new = []
        batch_size = 20
        for start in range(0, len(patient_list), batch_size):
            batch = patient_list[start:start + batch_size]
            batch_num = start // batch_size + 1
            end = min(start + batch_size, len(patient_list))
            print(f"[Intratumoral] Batch {batch_num}: patients {start+1}–{end}", flush=True)
            self.logger.info("Batch %d: patients %d–%d", batch_num, start+1, end)

            batch_results = Parallel(n_jobs=self.config.N_JOBS,
                                     backend=self.config.BACKEND,
                                     verbose=0)(
                delayed(self.analyze_patient)(pid, ctype)
                for pid, ctype in batch
            )
            all_new.extend(r for r in batch_results if r is not None)
            _save_checkpoint(checkpoint_path, existing + all_new)
            total = len(existing) + len(all_new)
            print(f"[Intratumoral] Checkpoint saved: {total} patients done", flush=True)
            self.logger.info("Checkpoint saved: %d total", total)

        all_results = existing + all_new
        self.results_df = pd.DataFrame(all_results)
        self.results_df.to_csv(
            os.path.join(self.config.OUTPUT_DIR, "intratumoral_heterogeneity_v4.csv"),
            index=False
        )
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        self.logger.info(f"Intratumoral analysis done: {len(self.results_df)} patients")
        return self.results_df


# ============================================================
# PERITUMORAL MICROENVIRONMENT ANALYZER (v4.0)
# ============================================================
class PeritumoralAnalyzer:

    def __init__(self, config: Config, logger: AnalysisLogger,
                 valid_df: pd.DataFrame, liver_mask_lookup: dict):
        self.config            = config
        self.logger            = logger
        self.patients          = valid_df
        self.liver_mask_lookup = liver_mask_lookup

    def _compute_peritumoral_features(self, ct_sitk: sitk.Image,
                                       mask_sitk: sitk.Image,
                                       distance_mm: float) -> Optional[dict]:
        peri_sitk = sitk_peritumoral_mask(mask_sitk, distance_mm)
        peri_arr  = sitk.GetArrayFromImage(peri_sitk)
        ct_arr    = sitk.GetArrayFromImage(ct_sitk)
        mask_arr  = sitk.GetArrayFromImage(mask_sitk)

        # Restrict to plausible liver parenchyma HU
        liver_range = (ct_arr > -50) & (ct_arr < 200)
        peri_arr    = peri_arr & liver_range

        peri_vals  = ct_arr[peri_arr > 0]
        tumor_vals = ct_arr[mask_arr > 0]
        if len(peri_vals) == 0 or len(tumor_vals) == 0:
            return None

        tumor_med = float(np.median(tumor_vals))
        peri_med  = float(np.median(peri_vals))

        feats = {f'peri_{k}': v for k, v in robust_stats(peri_vals).items()}
        feats['tumor_peri_diff']     = float(tumor_med - peri_med)
        feats['tumor_peri_ratio']    = float(tumor_med / (peri_med + 1e-5))
        feats['tumor_peri_contrast'] = float(
            (tumor_med - peri_med) / (abs(tumor_med) + abs(peri_med) + 1e-5)
        )
        sx, sy, sz = ct_sitk.GetSpacing()
        voxel_vol  = sx * sy * sz
        feats['peri_volume_mm3'] = float(peri_arr.sum() * voxel_vol)
        return feats

    def _compute_boundary_features(self, ct_sitk: sitk.Image,
                                    mask_sitk: sitk.Image,
                                    phase: str) -> Optional[dict]:
        """
        P1-D: gradient magnitude computed in physical space via sitk.
        P1-B: capsule searched in external shell (0–CAPSULE_SHELL_MM mm).
        """
        # 3D gradient magnitude (physically corrected, P1-D)
        grad_sitk = sitk_gradient_magnitude_physical(ct_sitk)
        mask_arr  = sitk.GetArrayFromImage(mask_sitk)

        # Physical 1mm boundary band each side of mask surface (spacing-invariant).
        # Distance-map approach replaces voxel-radius dilate/erode so boundary
        # thickness is consistent across patients regardless of slice spacing.
        dist_in_sitk  = sitk_distance_map_inside(mask_sitk)
        dist_out_sitk = sitk_distance_map_outside(mask_sitk)
        in_band       = sitk.Cast((dist_in_sitk  > 0) & (dist_in_sitk  <= 1.0), sitk.sitkUInt8)
        out_band      = sitk.Cast((dist_out_sitk > 0) & (dist_out_sitk <= 1.0), sitk.sitkUInt8)
        boundary_sitk = sitk.Cast(in_band | out_band, sitk.sitkUInt8)
        boundary_arr  = sitk.GetArrayFromImage(boundary_sitk)

        if boundary_arr.sum() == 0:
            return None

        grad_arr = sitk.GetArrayFromImage(grad_sitk)
        boundary_grads = grad_arr[boundary_arr > 0]

        grad_med  = float(np.median(boundary_grads))
        grad_mad  = compute_mad(boundary_grads)
        irreg     = float(compute_mad(boundary_grads) / (grad_med + 1e-5))

        feats = {
            'boundary_grad_median':    grad_med,
            'boundary_grad_mad':       grad_mad,
            'boundary_irregularity_3d': irreg,
        }

        # Inner vs outer boundary intensity
        inner = (mask_arr > 0) & (boundary_arr > 0)
        outer = (mask_arr == 0) & (boundary_arr > 0)
        ct_arr = sitk.GetArrayFromImage(ct_sitk)
        if inner.sum() > 0 and outer.sum() > 0:
            feats['boundary_gradient'] = float(
                np.median(ct_arr[inner]) - np.median(ct_arr[outer])
            )

        # P1-B: Capsule in EXTERNAL shell (0–CAPSULE_SHELL_MM mm)
        if phase in self.config.CAPSULE_PHASES:
            shell_sitk = sitk_external_shell(
                mask_sitk, 0.0, self.config.CAPSULE_SHELL_MM
            )
            shell_arr = sitk.GetArrayFromImage(shell_sitk)

            # Internal reference: deep core (>5 mm inside)
            dist_inside = dist_in_sitk   # reuse map from boundary computation
            dist_arr    = sitk.GetArrayFromImage(dist_inside)
            core_mask   = (dist_arr > 5) & (mask_arr > 0)

            if shell_arr.sum() > 0 and core_mask.sum() > 0:
                shell_vals = ct_arr[shell_arr > 0]
                core_vals  = ct_arr[core_mask]
                shell_med  = float(np.median(shell_vals))
                core_med   = float(np.median(core_vals))
                hu_diff    = shell_med - core_med
                enh_ratio  = shell_med / (core_med + 1e-5)

                # Ring continuity: fraction of shell voxels above threshold
                continuity = float(
                    np.sum(shell_vals > core_med + self.config.CAPSULE_HU_DIFF)
                    / len(shell_vals)
                )
                has_capsule = int(
                    hu_diff    > self.config.CAPSULE_HU_DIFF and
                    enh_ratio  > self.config.CAPSULE_ENH_RATIO and
                    continuity > self.config.CAPSULE_CONTINUITY_MIN
                )
                feats['capsule_hu_diff']   = float(hu_diff)
                feats['capsule_enh_ratio'] = float(enh_ratio)
                feats['capsule_continuity'] = continuity
                feats['has_capsule']        = has_capsule
        else:
            feats['has_capsule'] = np.nan

        return feats

    def _compute_invasion_features(self, ct_sitk: sitk.Image,
                                    mask_sitk: sitk.Image) -> Optional[dict]:
        """
        Perilesional attenuation decay profile using continuous equal-width 3mm bands
        (0–15mm, 5 bands). No gaps or overlaps between bands; band midpoints are used
        as x-coordinates for slope regression so spacing is uniform.
        """
        ct_arr = sitk.GetArrayFromImage(ct_sitk)
        feats  = {}

        # Continuous non-overlapping 3mm bands covering 0–15mm
        bands     = [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0), (12.0, 15.0)]
        midpoints = [1.5, 4.5, 7.5, 10.5, 13.5]
        ring_meds = []

        for (inner, outer), mid in zip(bands, midpoints):
            shell     = sitk_external_shell(mask_sitk, inner, outer)
            shell_arr = sitk.GetArrayFromImage(shell)
            liver_ok  = (ct_arr > -50) & (ct_arr < 200)
            vals = ct_arr[(shell_arr > 0) & liver_ok]
            if len(vals) == 0:
                ring_meds.append(np.nan)
                continue
            label = f'ring_{int(inner)}_{int(outer)}mm'
            feats[f'{label}_median'] = float(np.median(vals))
            feats[f'{label}_mad']    = compute_mad(vals)
            ring_meds.append(feats[f'{label}_median'])

        valid = [(mid, med) for mid, med in zip(midpoints, ring_meds) if not np.isnan(med)]
        if len(valid) >= 2:
            x = [v[0] for v in valid]
            y = [v[1] for v in valid]
            slope, _, r, p, _ = stats.linregress(x, y)
            feats['invasion_slope'] = float(slope)
            feats['invasion_r']     = float(r)
            feats['invasion_p']     = float(p)

        return feats if feats else None

    def _compute_tlr(self, ct_sitk: sitk.Image,
                     mask_sitk: sitk.Image,
                     liver_mask_sitk: Optional[sitk.Image]) -> Optional[dict]:
        """
        P1-E: TLR uses liver mask if available; otherwise falls back to
        peritumoral ring with no HU hard-filter (steatosis-safe).
        """
        ct_arr    = sitk.GetArrayFromImage(ct_sitk)
        mask_arr  = sitk.GetArrayFromImage(mask_sitk)
        tumor_vals = ct_arr[mask_arr > 0]
        if len(tumor_vals) == 0:
            return None

        if liver_mask_sitk is not None:
            liver_arr  = sitk.GetArrayFromImage(liver_mask_sitk)
            dist_out   = sitk_distance_map_outside(mask_sitk)
            dist_arr   = sitk.GetArrayFromImage(dist_out)
            # Normal parenchyma: liver mask, ≥30 mm from tumor, not inside tumor
            ref_mask = (liver_arr > 0) & (dist_arr >= 30) & (mask_arr == 0)
            ref_vals = ct_arr[ref_mask]
        else:
            # Fallback: annular ring 20–30 mm, no HU filter
            shell = sitk_external_shell(mask_sitk, 20.0, 30.0)
            shell_arr = sitk.GetArrayFromImage(shell)
            ref_vals  = ct_arr[shell_arr > 0]

        if len(ref_vals) < 50:
            return None

        tumor_med = float(np.median(tumor_vals))
        ref_med   = float(np.median(ref_vals))

        return {
            'tumor_liver_ratio':        float(tumor_med / (ref_med + 1e-5)),
            'tumor_liver_diff':         float(tumor_med - ref_med),
            'liver_ref_median':         ref_med,
            'liver_ref_mad':            compute_mad(ref_vals),
            'tlr_ref_source':           'liver_mask' if liver_mask_sitk is not None else 'ring_fallback',
            'tlr_liver_mask_available': int(liver_mask_sitk is not None),
        }

    def analyze_patient(self, patient_id: str,
                         cancer_type: str) -> Optional[dict]:
        files = _find_phase_files(patient_id, self.config)
        if len(files) < 4:
            return None

        results = {'patient_id': patient_id, 'cancer_type': cancer_type}
        # P0-E: NO delta features computed

        for phase in self.config.PHASES:
            if phase not in files:
                continue

            try:
                ct_sitk, mask_sitk = load_sitk_pair(
                    files[phase]['ct'], files[phase]['mask']
                )
            except Exception as exc:
                self.logger.record_error(patient_id, phase, 'load', exc)
                continue

            # Liver mask: CSV lookup has priority over filename-pattern search
            csv_liver_path  = self.liver_mask_lookup.get((patient_id, phase))
            liver_mask_sitk = _try_load_liver_mask(
                patient_id, phase, self.config, csv_path=csv_liver_path
            )

            blocks = {
                'peri_5mm':   lambda: self._compute_peritumoral_features(ct_sitk, mask_sitk, 5.0),
                'peri_10mm':  lambda: self._compute_peritumoral_features(ct_sitk, mask_sitk, 10.0),
                'boundary':   lambda: self._compute_boundary_features(ct_sitk, mask_sitk, phase),
                'invasion':   lambda: self._compute_invasion_features(ct_sitk, mask_sitk),
                'tlr':        lambda: self._compute_tlr(ct_sitk, mask_sitk, liver_mask_sitk),
            }

            for block_name, fn in blocks.items():
                try:
                    feats = fn()
                    if feats:
                        prefix = f'{phase}_{block_name}_'
                        for k, v in feats.items():
                            results[prefix + k] = v
                except Exception as exc:
                    self.logger.record_error(patient_id, phase, block_name, exc)

        return results if len(results) > 4 else None

    def run_analysis(self) -> pd.DataFrame:
        self.logger.info("=== PERITUMORAL MICROENVIRONMENT ANALYSIS (v4.0) ===")
        self.logger.info(f"Patients: {len(self.patients)}, Workers: {self.config.N_JOBS}")

        checkpoint_path = os.path.join(self.config.OUTPUT_DIR,
                                        "checkpoint_peritumoral.csv")
        processed, existing = _load_checkpoint(checkpoint_path)

        patient_list = [
            (r['patient_id'], r['cancer_type'])
            for _, r in self.patients.iterrows()
            if str(r['patient_id']) not in processed
        ]
        self.logger.info(f"Remaining: {len(patient_list)}")

        if not patient_list:
            self.results_df = pd.DataFrame(existing)
            return self.results_df

        all_new = []
        batch_size = 20
        for start in range(0, len(patient_list), batch_size):
            batch = patient_list[start:start + batch_size]
            batch_num = start // batch_size + 1
            end = min(start + batch_size, len(patient_list))
            print(f"[Peritumoral] Batch {batch_num}: patients {start+1}–{end}", flush=True)
            self.logger.info("Batch %d: patients %d–%d", batch_num, start+1, end)

            batch_results = Parallel(n_jobs=self.config.N_JOBS,
                                     backend=self.config.BACKEND,
                                     verbose=0)(
                delayed(self.analyze_patient)(pid, ctype)
                for pid, ctype in batch
            )
            all_new.extend(r for r in batch_results if r is not None)
            _save_checkpoint(checkpoint_path, existing + all_new)
            total = len(existing) + len(all_new)
            print(f"[Peritumoral] Checkpoint saved: {total} patients done", flush=True)
            self.logger.info("Checkpoint saved: %d total", total)

        all_results = existing + all_new
        self.results_df = pd.DataFrame(all_results)
        self.results_df.to_csv(
            os.path.join(self.config.OUTPUT_DIR, "peritumoral_analysis_v4.csv"),
            index=False
        )
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        self.logger.info(f"Peritumoral analysis done: {len(self.results_df)} patients")
        return self.results_df


# ============================================================
# STATISTICAL ANALYSIS (v4.0) — P1.5: FDR
# ============================================================
class SpatialStatisticalAnalyzer:

    def __init__(self, intra_df: pd.DataFrame, peri_df: pd.DataFrame,
                 config: Config, logger: AnalysisLogger):
        self.intra_df = intra_df
        self.peri_df  = peri_df
        self.config   = config
        self.logger   = logger

        self.combined_df = pd.merge(
            intra_df, peri_df,
            on=['patient_id', 'cancer_type'],
            how='outer', suffixes=('', '_peri')
        )

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """Return FDR-adjusted q-values (Benjamini-Hochberg)."""
        n = len(p_values)
        if n == 0:
            return np.array([])
        order    = np.argsort(p_values)
        ranked_p = p_values[order]
        q        = np.minimum(1.0, ranked_p * n / (np.arange(1, n + 1)))
        # Make cumulative min from right (enforce monotonicity)
        for i in range(n - 2, -1, -1):
            q[i] = min(q[i], q[i + 1])
        q_out = np.empty(n)
        q_out[order] = q
        return q_out

    def statistical_comparison(self) -> pd.DataFrame:
        self.logger.info("=== STATISTICAL COMPARISON (v4.0) ===")

        numeric_cols = self.combined_df.select_dtypes(include=[np.number]).columns
        exclude = {'patient_id'}
        feature_cols = [c for c in numeric_cols if c not in exclude
                        and c != 'cancer_type']

        rows = []
        for feat in feature_cols:
            groups = {
                ct: self.combined_df[self.combined_df['cancer_type'] == ct][feat].dropna().values
                for ct in self.config.CLASSES
            }
            groups = {k: v for k, v in groups.items() if len(v) > 2}
            if len(groups) < 2:
                continue

            group_arrs = list(groups.values())
            all_vals   = np.concatenate(group_arrs)
            if np.std(all_vals) < 1e-10:
                continue

            try:
                h, p = stats.kruskal(*group_arrs)
            except ValueError:
                continue

            n_total   = sum(len(g) for g in group_arrs)
            eta_sq    = max(0.0, (h - len(groups) + 1) / (n_total - len(groups)))

            row = {'feature': feat, 'KW_H': float(h),
                   'KW_p': float(p), 'eta_squared': float(eta_sq)}

            for ct in self.config.CLASSES:
                if ct in groups:
                    row[f'{ct}_median'] = float(np.median(groups[ct]))
                    row[f'{ct}_iqr']    = float(np.percentile(groups[ct], 75)
                                                 - np.percentile(groups[ct], 25))
                    row[f'{ct}_n']      = len(groups[ct])

            # Pairwise with Bonferroni (3 comparisons)
            pairs = [('HCC', 'ICC'), ('HCC', 'CHCC'), ('ICC', 'CHCC')]
            for c1, c2 in pairs:
                if c1 in groups and c2 in groups:
                    _, pv = stats.mannwhitneyu(groups[c1], groups[c2],
                                               alternative='two-sided')
                    row[f'p_{c1}_{c2}']      = float(pv)
                    row[f'p_{c1}_{c2}_bonf'] = float(min(1.0, pv * 3))

            rows.append(row)

        results_df = pd.DataFrame(rows)

        # P1.5: Benjamini-Hochberg FDR on all KW p-values
        if not results_df.empty:
            q_vals = self._benjamini_hochberg(results_df['KW_p'].values)
            results_df['KW_q_fdr'] = q_vals
            results_df = results_df.sort_values('KW_p')

        csv_path = os.path.join(self.config.OUTPUT_DIR, "spatial_statistics_v4.csv")
        results_df.to_csv(csv_path, index=False)

        sig = results_df[results_df.get('KW_q_fdr', results_df['KW_p']) < 0.05]
        self.logger.info(
            f"Statistical comparison: {len(results_df)} features tested, "
            f"{len(sig)} significant after FDR (q<0.05)"
        )

        self.stats_df = results_df
        return results_df

    def plot_key_features(self, top_n: int = 12):
        if not hasattr(self, 'stats_df') or self.stats_df.empty:
            return

        top_features = (
            self.stats_df
            .dropna(subset=['KW_q_fdr'])
            .nsmallest(top_n, 'KW_q_fdr')['feature']
            .tolist()
        )

        n_cols = 4
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 5, n_rows * 4))
        axes = np.array(axes).flatten()

        for ax, feat in zip(axes, top_features):
            data = [
                self.combined_df[self.combined_df['cancer_type'] == ct][feat].dropna()
                for ct in self.config.CLASSES
            ]
            bp = ax.boxplot(data, labels=self.config.CLASSES, patch_artist=True)
            for patch, ct in zip(bp['boxes'], self.config.CLASSES):
                patch.set_facecolor(self.config.CLASS_COLORS[ct])
                patch.set_alpha(0.7)
            row = self.stats_df[self.stats_df['feature'] == feat].iloc[0]
            ax.set_title(f"{feat}\np={row['KW_p']:.3f}, q={row['KW_q_fdr']:.3f}",
                         fontsize=8)

        for ax in axes[len(top_features):]:
            ax.set_visible(False)

        plt.tight_layout()
        out = os.path.join(self.config.OUTPUT_DIR, "top_features_comparison_v4.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Feature plot saved: {out}")


# ============================================================
# HELPERS: checkpoint, liver mask loader
# ============================================================
def _load_checkpoint(path: str) -> tuple[set, list]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        return set(df['patient_id'].astype(str).tolist()), df.to_dict('records')
    return set(), []


def _save_checkpoint(path: str, records: list):
    pd.DataFrame(records).to_csv(path, index=False)


def _try_load_liver_mask(patient_id: str, phase: str,
                          config: Config,
                          csv_path: Optional[str] = None) -> Optional[sitk.Image]:
    """
    Load phase-specific liver mask. csv_path (from the CSV lookup built in
    validate_dataset) has priority; filename-pattern candidates are the fallback.
    Returns None silently if not found (TLR falls back to ring method).
    """
    candidates = []
    if csv_path:
        candidates.append(csv_path)
    candidates += [
        os.path.join(config.MASK_DIR, f"{patient_id}_liver_{phase}.nii.gz"),
        os.path.join(config.MASK_DIR, f"{patient_id}_liver_{phase}.nii"),
        os.path.join(config.MASK_DIR, f"{patient_id}_liver.nii.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return sitk.ReadImage(p, sitk.sitkUInt8)
            except Exception:
                continue   # file exists but unreadable — try next candidate
    return None


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Liver tumor spatial & peritumoral feature extraction pipeline'
    )
    ap.add_argument('--data-dir', default='./data',
                    help='Root data directory (must contain ct_files/, mask_files/, patient_data.csv)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: <data-dir>/spatial_analysis_v4)')
    ns = ap.parse_args()

    config = Config()
    config.BASE_DIR      = ns.data_dir
    config.CT_DIR        = f"{ns.data_dir}/ct_files"
    config.MASK_DIR      = f"{ns.data_dir}/mask_files"
    config.CLINICAL_PATH = f"{ns.data_dir}/patient_data.csv"
    config.OUTPUT_DIR    = ns.output_dir or f"{ns.data_dir}/spatial_analysis_v4"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logger = AnalysisLogger(config.OUTPUT_DIR)

    # P3-B: save run config snapshot before anything else
    run_id, cfg_path = save_run_config(config, config.OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("SPATIAL ENHANCEMENT & PERITUMORAL ANALYSIS (v4.0)")
    logger.info("Run ID: %s | Config: %s", run_id, cfg_path)
    logger.info("=" * 70)

    # P0-D: validate before any processing
    valid_df, liver_mask_lookup = validate_dataset(config, logger)
    if len(valid_df) == 0:
        logger.summarize(raise_on_errors=True)
        return

    # Phase 1: Intratumoral heterogeneity
    intra_analyzer = IntratumoralHeterogeneityAnalyzer(config, logger, valid_df)
    intra_df = intra_analyzer.run_analysis()

    # Phase 2: Peritumoral microenvironment
    peri_analyzer = PeritumoralAnalyzer(config, logger, valid_df, liver_mask_lookup)
    peri_df = peri_analyzer.run_analysis()

    # Phase 3: Combine
    combined_df = pd.merge(
        intra_df, peri_df,
        on=['patient_id', 'cancer_type'],
        how='outer', suffixes=('', '_peri')
    )

    # P3-A: optional scanner harmonization (toggle in Config)
    if config.APPLY_SCANNER_HARMONIZATION and config.SCANNER_COL:
        # Carry scanner label from valid_df into combined_df for harmonization
        if config.SCANNER_COL in valid_df.columns:
            scanner_map = valid_df.set_index('patient_id')[config.SCANNER_COL]
            combined_df[config.SCANNER_COL] = (
                combined_df['patient_id'].map(scanner_map)
            )
        combined_df = apply_scanner_harmonization(
            combined_df, config.SCANNER_COL, logger, config.OUTPUT_DIR
        )
        out_name = "combined_spatial_v4_harmonized.csv"
    else:
        out_name = "combined_spatial_v4_full.csv"

    combined_df.to_csv(
        os.path.join(config.OUTPUT_DIR, out_name), index=False
    )
    logger.info("Combined feature file saved: %s", out_name)

    # Phase 4: Statistical analysis (FDR-corrected)
    stat_analyzer = SpatialStatisticalAnalyzer(intra_df, peri_df, config, logger)
    stat_analyzer.statistical_comparison()
    stat_analyzer.plot_key_features()

    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE — %d patients, %d features | run_id=%s",
                len(combined_df), len(combined_df.columns) - 2, run_id)
    logger.info("Outputs: %s", config.OUTPUT_DIR)

    # P0-D: mandatory error summary at end
    logger.summarize(raise_on_errors=False)


if __name__ == "__main__":
    main()

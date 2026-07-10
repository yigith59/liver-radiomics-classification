"""
05_robustness_analysis.py
=========================
Mask Perturbation Robustness Analysis — Top 5 SHAP Features
------------------------------------------------------------
For each patient, dilates and erodes the tumor mask by ~1 mm,
re-extracts the five primary discriminative features, and reports
ICC2(A,1) between original and perturbed values.

Output files (written to OUT_DIR):
  robustness_per_patient.csv   — per-patient feature values (orig/dilated/eroded)
  robustness_icc_summary.csv   — ICC + Spearman r + mean %change per feature
  robustness_icc_summary.txt   — ready-to-paste manuscript sentence

Run locally: set BASE below to your data directory.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
# Paths below are resolved at runtime inside main().
# Override via --data-dir / --output-dir CLI arguments.
BASE     = "./data"
FEAT_CSV = f"{BASE}/spatial_analysis_v4/combined_spatial_v4_full.csv"
CT_DIR   = f"{BASE}/ct_files"
MASK_DIR = f"{BASE}/mask_files"
OUT_DIR  = f"{BASE}/analysis_report_v4"

PERTURB_MM = 1.0   # physical perturbation radius
N_JOBS     = 4
MIN_MASK_VOXELS = 50   # skip patients with tiny masks that disappear on erosion

# (csv_column_name, phase, extractor_key)
TARGET_FEATURES = [
    ("C1_tlr_liver_ref_median", "C1", "tlr_ref"),
    ("C1_peri_10mm_peri_p90",   "C1", "peri10_p90"),
    ("C1_peri_5mm_peri_p90",    "C1", "peri5_p90"),
    ("C1_rim_median",            "C1", "rim_med"),
    ("P_enh_p10",                "P",  "enh_p10"),
]


# ══════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _find_ct_mask(pid: str, phase: str):
    ct_cands = [
        f"{CT_DIR}/{pid}_ct_{phase}.nii.gz",
        f"{CT_DIR}/{pid}_{phase}.nii.gz",
    ]
    mask_cands = [
        f"{MASK_DIR}/{pid}_mask_{phase}.nii.gz",
        f"{MASK_DIR}/{pid}_mask_{phase}.nii",
    ]
    ct   = next((p for p in ct_cands   if os.path.exists(p)), None)
    mask = next((p for p in mask_cands if os.path.exists(p)), None)
    return ct, mask


def _load_pair(ct_path: str, mask_path: str):
    """Load CT (float64) and mask (uint8); resample mask to CT grid if needed."""
    ct_sitk   = sitk.ReadImage(ct_path,   sitk.sitkFloat64)
    mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    # Handle 4D CTs (some ICC files stored as RGBA NIfTI)
    ct_arr = sitk.GetArrayFromImage(ct_sitk)
    if ct_arr.ndim == 4:
        ct_arr = ct_arr.mean(axis=-1).astype(np.float64)
        ct_sitk = sitk.GetImageFromArray(ct_arr)
        ct_sitk.CopyInformation(sitk.ReadImage(ct_path))

    if ct_sitk.GetSize() != mask_sitk.GetSize():
        mask_sitk = sitk.Resample(
            mask_sitk, ct_sitk,
            sitk.Transform(), sitk.sitkNearestNeighbor, 0,
            mask_sitk.GetPixelID()
        )
    return ct_sitk, mask_sitk


# ══════════════════════════════════════════════════════════════════════════
# MASK PERTURBATION
# ══════════════════════════════════════════════════════════════════════════
def _mm_to_vox_radii(sitk_img, radius_mm: float):
    """Convert physical radius (mm) to per-axis voxel radii."""
    spacing = sitk_img.GetSpacing()   # (sx, sy, sz) mm/vox
    return [max(1, round(radius_mm / sp)) for sp in spacing]


def dilate_mask(mask_sitk, radius_mm: float = 1.0):
    radii = _mm_to_vox_radii(mask_sitk, radius_mm)
    return sitk.BinaryDilate(mask_sitk, radii, sitk.sitkBall, 0, 1, False)


def erode_mask(mask_sitk, radius_mm: float = 1.0):
    radii = _mm_to_vox_radii(mask_sitk, radius_mm)
    eroded = sitk.BinaryErode(mask_sitk, radii, sitk.sitkBall, 0, 1, False)
    if sitk.GetArrayFromImage(eroded).sum() < MIN_MASK_VOXELS:
        return None   # mask vanished — signal caller to skip
    return eroded


# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS (minimal re-implementation for the 5 targets)
# ══════════════════════════════════════════════════════════════════════════
def _dist_map_outside(mask_sitk):
    inv = sitk.Cast(sitk.Cast(mask_sitk, sitk.sitkUInt8) == 0, sitk.sitkUInt8)
    return sitk.SignedMaurerDistanceMap(
        inv, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
    )

def _dist_map_inside(mask_sitk):
    m = sitk.Cast(mask_sitk, sitk.sitkUInt8)
    return sitk.SignedMaurerDistanceMap(
        m, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
    )


def feat_tlr_ref(ct_sitk, mask_sitk):
    """C1_tlr_liver_ref_median: median of liver parenchyma 20-30mm from tumor."""
    ct_arr   = sitk.GetArrayFromImage(ct_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    dist_arr = sitk.GetArrayFromImage(_dist_map_outside(mask_sitk))
    # Reference ring: 20-30mm outside tumor, no HU filter (steatosis-safe)
    ref_mask = (dist_arr >= 20) & (dist_arr <= 30) & (mask_arr == 0)
    ref_vals = ct_arr[ref_mask]
    if len(ref_vals) < 50:
        return np.nan
    return float(np.median(ref_vals))


def feat_peri_p90(ct_sitk, mask_sitk, distance_mm: float):
    """C1_peri_{d}mm_peri_p90: 90th pct of peritumoral shell, liver HU range."""
    ct_arr   = sitk.GetArrayFromImage(ct_sitk)
    dist_arr = sitk.GetArrayFromImage(_dist_map_outside(mask_sitk))
    peri     = (dist_arr > 0) & (dist_arr <= distance_mm)
    liver_ok = (ct_arr > -50) & (ct_arr < 200)
    vals     = ct_arr[peri & liver_ok]
    if len(vals) == 0:
        return np.nan
    return float(np.percentile(vals, 90))


def feat_rim_median(ct_sitk, mask_sitk):
    """C1_rim_median: median HU of outermost 20% depth rim inside tumor."""
    ct_arr   = sitk.GetArrayFromImage(ct_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    dist_arr = sitk.GetArrayFromImage(_dist_map_inside(mask_sitk))
    max_dist = dist_arr[mask_arr > 0].max() if (mask_arr > 0).any() else 0
    if max_dist < 3:
        return np.nan
    rim_mask = (mask_arr > 0) & (dist_arr <= max_dist * 0.2)
    vals = ct_arr[rim_mask]
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))


def feat_enh_p10(ct_sitk, mask_sitk):
    """P_enh_p10: 10th percentile HU inside tumor on pre-contrast phase."""
    ct_arr   = sitk.GetArrayFromImage(ct_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    vals     = ct_arr[mask_arr > 0]
    if len(vals) == 0:
        return np.nan
    return float(np.percentile(vals, 10))


# Dispatcher: extractor_key → function (with any extra args curried in)
def _extract(key: str, ct_sitk, mask_sitk):
    if key == "tlr_ref":
        return feat_tlr_ref(ct_sitk, mask_sitk)
    if key == "peri10_p90":
        return feat_peri_p90(ct_sitk, mask_sitk, 10.0)
    if key == "peri5_p90":
        return feat_peri_p90(ct_sitk, mask_sitk, 5.0)
    if key == "rim_med":
        return feat_rim_median(ct_sitk, mask_sitk)
    if key == "enh_p10":
        return feat_enh_p10(ct_sitk, mask_sitk)
    raise ValueError(f"Unknown extractor key: {key}")


# ══════════════════════════════════════════════════════════════════════════
# PER-PATIENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
def analyze_patient(pid: str, cancer_type: str):
    row = {"patient_id": pid, "cancer_type": cancer_type}

    for feat_name, phase, ext_key in TARGET_FEATURES:
        ct_path, mask_path = _find_ct_mask(pid, phase)
        if ct_path is None or mask_path is None:
            for suffix in ("_orig", "_dil", "_ero"):
                row[feat_name + suffix] = np.nan
            continue

        try:
            ct_sitk, mask_sitk = _load_pair(ct_path, mask_path)
        except Exception as e:
            print(f"  [WARN] {pid} {phase}: load error — {e}")
            for suffix in ("_orig", "_dil", "_ero"):
                row[feat_name + suffix] = np.nan
            continue

        # Original
        row[feat_name + "_orig"] = _extract(ext_key, ct_sitk, mask_sitk)

        # Dilated
        try:
            mask_dil = dilate_mask(mask_sitk, PERTURB_MM)
            row[feat_name + "_dil"] = _extract(ext_key, ct_sitk, mask_dil)
        except Exception as e:
            print(f"  [WARN] {pid} {phase} dil: {e}")
            row[feat_name + "_dil"] = np.nan

        # Eroded
        try:
            mask_ero = erode_mask(mask_sitk, PERTURB_MM)
            if mask_ero is None:
                row[feat_name + "_ero"] = np.nan   # mask too small
            else:
                row[feat_name + "_ero"] = _extract(ext_key, ct_sitk, mask_ero)
        except Exception as e:
            print(f"  [WARN] {pid} {phase} ero: {e}")
            row[feat_name + "_ero"] = np.nan

    return row


# ══════════════════════════════════════════════════════════════════════════
# ICC2(A,1) — two-way random effects, absolute agreement, single measures
# ══════════════════════════════════════════════════════════════════════════
def icc2(vals_a: np.ndarray, vals_b: np.ndarray):
    """
    ICC(2,1): two-way random, absolute agreement, single rater.
    Standard for radiomics test-retest; threshold: <0.50 poor, 0.50-0.75 moderate,
    0.75-0.90 good, >0.90 excellent (Koo & Li, 2016).
    """
    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    a, b = vals_a[mask], vals_b[mask]
    n = len(a)
    if n < 10:
        return np.nan, np.nan, n    # too few subjects for reliable estimate

    k = 2
    data = np.column_stack([a, b])
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    SSr = k * np.sum((row_means - grand) ** 2)
    SSc = n * np.sum((col_means - grand) ** 2)
    SSe = np.sum((data - row_means[:, None] - col_means[None, :] + grand) ** 2)

    MSr = SSr / (n - 1)
    MSc = SSc / (k - 1)
    MSe = SSe / ((n - 1) * (k - 1))

    denom = MSr + (k - 1) * MSe + k / n * (MSc - MSe)
    if denom < 1e-10:
        return np.nan, np.nan, n

    icc_val = (MSr - MSe) / denom

    # 95 % CI via Fisher z-transform approximation
    se  = np.sqrt(2 * k * (1 - icc_val) ** 2 * (1 + (k - 1) * icc_val) ** 2 /
                  (k * (n - 1) * (k * icc_val + n * (1 - icc_val)) ** 2 + 1e-10))
    z   = np.arctanh(np.clip(icc_val, -0.9999, 0.9999))
    lo  = float(np.tanh(z - 1.96 * se))
    hi  = float(np.tanh(z + 1.96 * se))

    return float(np.clip(icc_val, -1, 1)), (round(lo, 3), round(hi, 3)), n


def mean_pct_change(orig: np.ndarray, pert: np.ndarray):
    mask = ~(np.isnan(orig) | np.isnan(pert)) & (np.abs(orig) > 1e-6)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pert[mask] - orig[mask]) / np.abs(orig[mask])) * 100)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Mask perturbation robustness analysis (ICC2 for top-5 SHAP features)'
    )
    ap.add_argument('--data-dir', default='./data',
                    help='Root data directory (default: ./data)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: <data-dir>/analysis_report_v4)')
    ns = ap.parse_args()

    global FEAT_CSV, CT_DIR, MASK_DIR, OUT_DIR
    FEAT_CSV = f"{ns.data_dir}/spatial_analysis_v4/combined_spatial_v4_full.csv"
    CT_DIR   = f"{ns.data_dir}/ct_files"
    MASK_DIR = f"{ns.data_dir}/mask_files"
    OUT_DIR  = ns.output_dir or f"{ns.data_dir}/analysis_report_v4"
    os.makedirs(OUT_DIR, exist_ok=True)

    df_feat = pd.read_csv(FEAT_CSV)
    df_feat["patient_id"] = df_feat["patient_id"].astype(str).str.strip()

    patients = df_feat[["patient_id", "cancer_type"]].drop_duplicates().reset_index(drop=True)
    total = len(patients)
    print(f"Patients to process: {total}")

    # ── Per-patient extraction ────────────────────────────────────────────
    results = []
    batch_size = 20
    for start in range(0, total, batch_size):
        batch = patients.iloc[start:start+batch_size]
        end   = min(start + batch_size, total)
        print(f"[Batch] {start+1}–{end} / {total}", flush=True)

        batch_res = Parallel(n_jobs=N_JOBS, backend="threading", verbose=0)(
            delayed(analyze_patient)(row["patient_id"], row["cancer_type"])
            for _, row in batch.iterrows()
        )
        results.extend(batch_res)
        print(f"  done ({len(results)} total)", flush=True)

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{OUT_DIR}/robustness_per_patient.csv", index=False)
    print(f"\nPer-patient results saved ({len(df_res)} rows).")

    # ── ICC summary ───────────────────────────────────────────────────────
    summary_rows = []
    for feat_name, phase, _ in TARGET_FEATURES:
        orig = df_res[feat_name + "_orig"].values
        dil  = df_res[feat_name + "_dil"].values
        ero  = df_res[feat_name + "_ero"].values

        icc_dil, ci_dil, n_dil = icc2(orig, dil)
        icc_ero, ci_ero, n_ero = icc2(orig, ero)

        pct_dil = mean_pct_change(orig, dil)
        pct_ero = mean_pct_change(orig, ero)

        r_dil, p_dil = stats.spearmanr(orig[~np.isnan(orig+dil)],
                                        dil [~np.isnan(orig+dil)])
        r_ero, p_ero = stats.spearmanr(orig[~np.isnan(orig+ero)],
                                        ero [~np.isnan(orig+ero)])

        # Interpretation
        def interp(icc_v):
            if np.isnan(icc_v): return "–"
            if icc_v >= 0.90: return "excellent"
            if icc_v >= 0.75: return "good"
            if icc_v >= 0.50: return "moderate"
            return "poor"

        summary_rows.append({
            "feature":           feat_name,
            "phase":             phase,
            "n_dil":             n_dil,
            "ICC_dil":           round(icc_dil, 3) if not np.isnan(icc_dil) else np.nan,
            "ICC_dil_CI":        str(ci_dil),
            "pct_change_dil":    round(pct_dil, 1) if not np.isnan(pct_dil) else np.nan,
            "spearman_r_dil":    round(r_dil, 3),
            "spearman_p_dil":    round(p_dil, 4),
            "n_ero":             n_ero,
            "ICC_ero":           round(icc_ero, 3) if not np.isnan(icc_ero) else np.nan,
            "ICC_ero_CI":        str(ci_ero),
            "pct_change_ero":    round(pct_ero, 1) if not np.isnan(pct_ero) else np.nan,
            "spearman_r_ero":    round(r_ero, 3),
            "spearman_p_ero":    round(p_ero, 4),
            "interp_dil":        interp(icc_dil),
            "interp_ero":        interp(icc_ero),
        })

    df_icc = pd.DataFrame(summary_rows)
    df_icc.to_csv(f"{OUT_DIR}/robustness_icc_summary.csv", index=False)

    # ── Console report ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ROBUSTNESS SUMMARY")
    print("="*70)
    print(f"{'Feature':<32} {'ICC_dil':>8} {'CI95_dil':>16} "
          f"{'%Δ_dil':>7} {'ICC_ero':>8} {'CI95_ero':>16} {'%Δ_ero':>7}")
    print("-"*70)
    for _, r in df_icc.iterrows():
        print(f"{r['feature']:<32} {r['ICC_dil']:>8.3f} {str(r['ICC_dil_CI']):>16} "
              f"{r['pct_change_dil']:>7.1f} {r['ICC_ero']:>8.3f} "
              f"{str(r['ICC_ero_CI']):>16} {r['pct_change_ero']:>7.1f}")

    # ── Auto-generate manuscript sentence ─────────────────────────────────
    all_icc = df_icc[["ICC_dil", "ICC_ero"]].values.flatten()
    all_icc = all_icc[~np.isnan(all_icc)]
    icc_min  = float(all_icc.min())
    icc_max  = float(all_icc.max())

    peri_rows = df_icc[df_icc["feature"].str.contains("peri|rim")]
    peri_icc  = peri_rows[["ICC_dil","ICC_ero"]].values.flatten()
    peri_icc  = peri_icc[~np.isnan(peri_icc)]

    pct_vals  = df_icc[["pct_change_dil","pct_change_ero"]].values.flatten()
    pct_vals  = pct_vals[~np.isnan(pct_vals)]

    sentence = (
        f"\nMANUSCRIPT SENTENCE (Methods/Limitations):\n"
        f"{'─'*70}\n"
        f"To assess robustness to segmentation uncertainty, we applied simulated "
        f"±{PERTURB_MM:.0f} mm mask dilation and erosion to all patients and "
        f"re-extracted the five primary discriminative features. ICC values ranged "
        f"from {icc_min:.2f} to {icc_max:.2f} across all features and perturbation "
        f"directions, with a mean absolute percentage change of "
        f"{float(pct_vals.mean()):.1f}% (range {float(pct_vals.min()):.1f}–"
        f"{float(pct_vals.max()):.1f}%). "
    )
    if len(peri_icc) > 0:
        sentence += (
            f"Peritumoral and boundary-zone features showed ICC values of "
            f"{float(peri_icc.min()):.2f}–{float(peri_icc.max()):.2f}, "
            f"consistent with their inherent sensitivity to contour placement. "
        )
    sentence += (
        f"These results indicate that feature values are [stable/moderately sensitive] "
        f"to clinically plausible segmentation variation, and that the observed "
        f"inter-class differences are unlikely to be attributable to mask uncertainty alone."
        f"\n{'─'*70}\n"
        f"[EDIT: replace [stable/moderately sensitive] based on ICC values above]\n"
    )
    print(sentence)

    with open(f"{OUT_DIR}/robustness_icc_summary.txt", "w", encoding="utf-8") as f:
        f.write(sentence)
        f.write("\n\nFull ICC table:\n")
        f.write(df_icc.to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("  robustness_per_patient.csv")
    print("  robustness_icc_summary.csv")
    print("  robustness_icc_summary.txt   ← manuscript sentence")


if __name__ == "__main__":
    main()

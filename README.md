# Three-Class Radiomic Differentiation of HCC, ICC, and cHCC-CCA

Code repository for the manuscript:

**"Three-Class Radiomic Differentiation of Hepatocellular Carcinoma, Intrahepatic Cholangiocarcinoma, and Combined Hepatocellular-Cholangiocarcinoma on Multiphasic Contrast-Enhanced CT"**

*Abdominal Radiology* (under revision)

---

## Overview

This repository contains the full analysis pipeline for a radiomics study classifying three primary liver tumor types (HCC, ICC, cHCC-CCA) on multiphasic CECT using custom spatial features and XGBoost with nested cross-validation.

- **N = 138 patients** (HCC = 72, ICC = 43, cHCC-CCA = 23)
- **Four CECT phases**: Pre-contrast (P), Arterial (C1), Portal Venous (C2), Delayed (C3)
- **OOF macro AUC = 0.953** (full nested CV, SHAP top-K arm)

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_feature_extraction.py` | Custom spatial + peritumoral feature extraction (SimpleITK) |
| `02_ml_classification.py` | Nested CV XGBoost pipeline with 4 feature-selection arms |
| `03_statistics.py` | Statistical analysis, FDR correction, SHAP concordance, ablation |
| `04_sensitivity_analysis.py` | HU normalization sensitivity analysis (R2.2) |
| `05_robustness_analysis.py` | Mask perturbation robustness (ICC2 stability, top-5 SHAP features) |

---

## Data Configuration

Each script contains a `BASE_DIR` (or `BASE` / `CSV_PATH`) variable at the top of the configuration section. Set this to your local data directory before running.

Expected directory layout under `BASE_DIR`:

```
data/
├── ct_files/           # NIfTI CT volumes, named <patient_id>_<phase>.nii.gz
├── mask_files/         # NIfTI tumor masks, named <patient_id>_<phase>_mask.nii.gz
├── patient_data.csv    # clinical metadata (columns: patient_id, cancer_type, ...)
└── spatial_analysis_v4/
    └── combined_spatial_v4_full.csv   # output of 01_feature_extraction.py
```

> **Note:** Patient imaging data are not publicly available due to institutional privacy restrictions. Feature matrices and derived outputs can be shared upon reasonable request to the corresponding author.

---

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.9. See `requirements.txt` for pinned versions.

---

## Execution Order

```bash
python 01_feature_extraction.py    # generates combined_spatial_v4_full.csv
python 02_ml_classification.py     # nested CV, saves OOF predictions + SHAP
python 03_statistics.py            # KW tests, FDR, figures, ablation
python 04_sensitivity_analysis.py  # HU normalization sensitivity
python 05_robustness_analysis.py   # mask perturbation ICC
```

---

## Key Methodological Notes

- **Non-IBSI features** (radial profile, rim/core contrast, 3D angular heterogeneity, perilesional invasion slope, necrosis fraction, capsule score, TLR) are defined mathematically in Online Resource 1 of the manuscript.
- **Effect size** is reported as η² = (H − k + 1)/(n − k) where H is the Kruskal-Wallis statistic, k the number of groups, and n the total sample size.
- **FDR correction** uses the Benjamini-Hochberg procedure (backward-cumulative-minimum formulation).
- **Nested CV**: outer 5-fold for performance estimation, inner 3-fold for hyperparameter tuning (RandomizedSearchCV, 25 iterations). All preprocessing (imputation, scaling, variance filter, correlation pruning, SHAP top-K selection) is fitted on the training fold only.

---

## Citation

If you use this code, please cite the manuscript (citation will be updated upon acceptance).

---

## License

MIT License. See `LICENSE` for details.

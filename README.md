# Three-Class Radiomic Differentiation of Primary Liver Cancer

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Abdominal%20Radiology%202026-blue?style=flat-square)](https://doi.org/10.1007/s00261-026-05715-7)
[![PubMed](https://img.shields.io/badge/PubMed-42530595-red?style=flat-square)](https://pubmed.ncbi.nlm.nih.gov/42530595/)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00261--026--05715--7-green?style=flat-square)](https://doi.org/10.1007/s00261-026-05715-7)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue?style=flat-square)](https://www.python.org/)

</div>

---

> **"Three-Class Radiomic Differentiation of Hepatocellular Carcinoma, Intrahepatic Cholangiocarcinoma, and Combined Hepatocellular-Cholangiocarcinoma on Multiphasic Contrast-Enhanced CT"**  
> Yiğit Hasan Arı · *Abdominal Radiology*, 2026  
> DOI: [10.1007/s00261-026-05715-7](https://doi.org/10.1007/s00261-026-05715-7) · PMID: [42530595](https://pubmed.ncbi.nlm.nih.gov/42530595/)

---

## Overview

Full analysis pipeline for a radiomics study classifying three primary liver tumor types — **HCC**, **ICC**, and **cHCC-CCA** — on four-phase multiphasic CECT using custom physically-defined spatial features and XGBoost with a bias-aware nested cross-validation framework.

| | |
|---|---|
| **Cohort** | N = 278 (HCC = 94, ICC = 99, cHCC-CCA = 85) |
| **Imaging** | Four-phase CECT: Pre-contrast · Arterial · Portal venous · Delayed |
| **Primary model** | XGBoost + SHAP top-K feature selection |
| **OOF macro AUC** | 0.948 (nested CV, pooled out-of-fold predictions) |
| **Nested CV AUC** | 0.953 ± 0.030 |
| **Permutation p** | < 0.001 (B = 1000) |

---

## Repository Structure

```
.
├── 01_feature_extraction.py     # Custom spatial + peritumoral feature extraction (SimpleITK)
├── 02_ml_classification.py      # Nested CV XGBoost pipeline — 4 feature-selection arms
├── 03_statistics.py             # KW tests, FDR correction, SHAP concordance, ablation
├── 04_sensitivity_analysis.py   # HU normalization sensitivity analysis (liver-reference subtraction)
├── 05_robustness_analysis.py    # Mask perturbation robustness (±1 mm dilation/erosion, ICC)
├── requirements.txt
└── README.md
```

---

## Data

The imaging dataset is publicly available:

> Luo X, et al. *A comprehensive multi-phase 3D contrast-enhanced CT imaging dataset for primary liver cancer.* Scientific Data, 2025.  

Download the dataset and set `BASE_DIR` at the top of each script to your local data directory.

**Expected directory layout:**

```
data/
├── ct_files/                          # NIfTI CT volumes: <patient_id>_<phase>.nii.gz
├── mask_files/                        # NIfTI tumor masks: <patient_id>_<phase>_mask.nii.gz
├── patient_data.csv                   # Clinical metadata (patient_id, cancer_type, ...)
└── spatial_analysis_v4/
    └── combined_spatial_v4_full.csv   # Output of 01_feature_extraction.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with **Python 3.9**. See `requirements.txt` for pinned package versions.

---

## Execution Order

```bash
python 01_feature_extraction.py    # → combined_spatial_v4_full.csv
python 02_ml_classification.py     # → OOF predictions + SHAP values
python 03_statistics.py            # → KW results, FDR, figures, ablation table
python 04_sensitivity_analysis.py  # → HU normalization sensitivity (ΔAUC)
python 05_robustness_analysis.py   # → ICC per feature under mask perturbation
```

---

## Key Methodological Notes

**Feature extraction**
- Custom SimpleITK pipeline with physically-defined ROIs: intratumoral radial zones, 5 mm / 10 mm peritumoral rings, ±1 mm boundary zone, five 3 mm perilesional bands
- Non-IBSI features (radial profile, rim/core contrast, 3D angular heterogeneity, perilesional attenuation slope, necrosis fraction, capsule score, tumor-to-liver ratio) are defined mathematically in **Supplementary Material 1** of the manuscript
- Standard first-order statistics conform to IBSI definitions; gradient magnitude uses σ = 0.5 mm Gaussian (minor deviation from IBSI 3.7, noted in 

**FDR correction**
- Benjamini-Hochberg procedure, backward-cumulative-minimum formulation

**Nested cross-validation**
- Outer 5-fold (performance estimation) · Inner 3-fold (hyperparameter tuning, RandomizedSearchCV, 25 iterations)
- All preprocessing steps — imputation, z-score scaling, variance filtering, correlation pruning (|r| > 0.90), feature selection — fitted on training folds only; no information leakage into test folds

**Permutation testing**
- B = 1000 label permutations, fixed-parameter surrogate classifier
- Exact p-value per Phipson & Smyth (2010): p = (r + 1)/(B + 1)

---

## Supplementary Materials

| File | Contents |
|---|---|
| Supplementary Material 1 | Mathematical definitions of all non-standard features |
| Supplementary Material 2 | TRIPOD+AI reporting checklist |
| Supplementary Material 3 | Univariable statistics for all 373 features |
| Supplementary Material 4 | Fold-level feature selection stability |
| Supplementary Material 5 | Ablation and mechanistic model comparison |
| Supplementary Material 6 | Per-patient misclassification data |

---

## Citation

If you use this code, please cite:

```bibtex
@article{ari2026liver,
  author  = {Arı, Yiğit Hasan},
  title   = {Three-class radiomic differentiation of hepatocellular carcinoma,
             intrahepatic cholangiocarcinoma, and combined hepatocellular-cholangiocarcinoma
             on multiphasic contrast-enhanced {CT}},
  journal = {Abdominal Radiology},
  year    = {2026},
  doi     = {10.1007/s00261-026-05715-7},
  pmid    = {42530595}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Istanbul Faculty of Medicine · Istanbul University · Istanbul, Turkey</sub>
</div>

"""
Statistical Analysis & Visualization
======================================
Version 4.0

Reads outputs from 01_feature_extraction.py and 02_ml_classification.py.
Produces:
  - Descriptive statistics table (median ± IQR per class)
  - FDR-corrected Kruskal-Wallis results table
  - Top feature boxplots (q < 0.05, sorted by effect size)
  - Per-class ROC curves (from nested CV OOF predictions)
  - Confusion matrix heatmap
  - SHAP importance bar chart (fold-aggregate)
  - Radial profile comparison across phases
  - Feature correlation heatmap (top 30 by variance)
  - Class distribution pie chart
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report, f1_score, roc_auc_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    BASE_DIR     = "./data"  # set this to your local data directory
    SPATIAL_DIR  = f"{BASE_DIR}/spatial_analysis_v4"
    ML_DIR       = f"{BASE_DIR}/ml_classification_v4"
    OUTPUT_DIR   = f"{BASE_DIR}/analysis_report_v4"

    FEATURES_FILE    = f"{SPATIAL_DIR}/combined_spatial_v4_full.csv"
    STATS_FILE       = f"{SPATIAL_DIR}/spatial_statistics_v4.csv"

    # AUTO_SELECT_ARM=True (default): reads model_comparison_v4.csv at runtime and
    # selects the arm with highest cv_auc_ovr_mean — no manual update needed.
    # Set to False when making a parsimony choice over strict AUC-maximization
    # (e.g. shap_topk vs full_pipeline differ by <0.002 AUC but feature counts differ).
    # With False, Config.SELECTED_ARM is used as-is without override.
    AUTO_SELECT_ARM  = True
    SELECTED_ARM     = 'shap_topk'   # placeholder; overridden by auto-detect when True

    # Set True to skip permutation test and reuse existing permutation_test_all_features.csv.
    # Permutation result does not depend on SELECTED_ARM (uses all imaging features),
    # so a re-run for arm change does not require re-running B=1000 permutations.
    SKIP_PERMUTATION_TEST = False
    CV_RESULTS_FILE  = f"{ML_DIR}/nested_cv_fold_results_{SELECTED_ARM}.csv"
    SHAP_FILE        = f"{ML_DIR}/shap_oof_importance_{SELECTED_ARM}.csv"
    FOLD_FEATURES_FILE = f"{ML_DIR}/fold_selected_features_{SELECTED_ARM}.csv"
    COMPARISON_FILE  = f"{ML_DIR}/model_comparison_v4.csv"

    CLASSES      = ['CHCC', 'HCC', 'ICC']   # alphabetical = LabelEncoder order
    CLASS_COLORS = {'HCC': '#e74c3c', 'ICC': '#3498db', 'CHCC': '#9b59b6'}
    CLASS_LABELS = {0: 'CHCC', 1: 'HCC', 2: 'ICC'}

    FDR_THRESHOLD = 0.05
    TOP_N_FEATURES = 16   # boxplots
    TOP_N_SHAP     = 20   # SHAP bar
    CORR_CLUSTER_THRESHOLD = 0.90

    # Extended statistical analyses (v4.1)
    ABLATION_SPLITS = 5
    ABLATION_RANDOM_STATE = 42
    PERMUTATION_N = 1000
    TOP_CONCORDANCE_N = 20
    CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)
    RUN_V42_BIAS_MECHANISM = True

    # Lightweight fixed model for ablation/permutation
    ABLATION_XGB = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': CPU_THREADS,
    }

log = logging.getLogger("stats_report")


# ============================================================
# DATA LOADING
# ============================================================
def load_data(config: Config) -> dict:
    data = {}

    # Arm selection — conditional on AUTO_SELECT_ARM flag
    if config.AUTO_SELECT_ARM and os.path.exists(config.COMPARISON_FILE):
        try:
            comp = pd.read_csv(config.COMPARISON_FILE)
            if not comp.empty and 'cv_auc_ovr_mean' in comp.columns:
                best_idx   = comp['cv_auc_ovr_mean'].idxmax()
                best_arm   = str(comp.loc[best_idx, 'arm'])
                best_auc   = float(comp.loc[best_idx, 'cv_auc_ovr_mean'])
                best_nfeat = comp.loc[best_idx].get('mean_n_features', 'N/A')
                if best_arm != config.SELECTED_ARM:
                    log.info(
                        "Auto-select: '%s' (cv_auc=%.3f) differs from "
                        "Config.SELECTED_ARM='%s' — overriding.",
                        best_arm, best_auc, config.SELECTED_ARM
                    )
                config.SELECTED_ARM = best_arm
                log.info(
                    "Selected arm (auto): %s | cv_auc_ovr=%.3f | mean_n_features=%s",
                    best_arm, best_auc, best_nfeat
                )
        except Exception as exc:
            log.warning(
                "Auto-arm detection failed (%s) — keeping Config.SELECTED_ARM='%s'.",
                exc, config.SELECTED_ARM
            )
    elif not config.AUTO_SELECT_ARM:
        log.info(
            "AUTO_SELECT_ARM=False — using Config.SELECTED_ARM='%s' (manual override).",
            config.SELECTED_ARM
        )
    else:
        log.info(
            "Comparison file not found — using Config.SELECTED_ARM='%s' (placeholder).",
            config.SELECTED_ARM
        )

    # Sync all arm-dependent paths to the current SELECTED_ARM (works in all code paths)
    config.CV_RESULTS_FILE   = os.path.join(config.ML_DIR, f"nested_cv_fold_results_{config.SELECTED_ARM}.csv")
    config.SHAP_FILE          = os.path.join(config.ML_DIR, f"shap_oof_importance_{config.SELECTED_ARM}.csv")
    config.FOLD_FEATURES_FILE = os.path.join(config.ML_DIR, f"fold_selected_features_{config.SELECTED_ARM}.csv")

    log.info("Loading feature file...")
    df = pd.read_csv(config.FEATURES_FILE)
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    data['features'] = df
    log.info("Features: %d patients, %d columns", len(df), len(df.columns))

    if os.path.exists(config.STATS_FILE):
        data['stats'] = pd.read_csv(config.STATS_FILE)
        log.info("Stats file loaded: %d features tested", len(data['stats']))
    else:
        log.warning("Stats file not found: %s", config.STATS_FILE)
        data['stats'] = pd.DataFrame()

    if os.path.exists(config.CV_RESULTS_FILE):
        data['cv'] = pd.read_csv(config.CV_RESULTS_FILE)
    else:
        log.warning("CV results not found for arm '%s': %s",
                    config.SELECTED_ARM, config.CV_RESULTS_FILE)
        data['cv'] = pd.DataFrame()

    if os.path.exists(config.SHAP_FILE):
        data['shap'] = pd.read_csv(config.SHAP_FILE)
    else:
        log.warning("SHAP file not found for arm '%s': %s",
                    config.SELECTED_ARM, config.SHAP_FILE)
        data['shap'] = pd.DataFrame()

    if os.path.exists(config.COMPARISON_FILE):
        comp_df  = pd.read_csv(config.COMPARISON_FILE)
        arm_rows = comp_df[comp_df['arm'] == config.SELECTED_ARM]
        if not arm_rows.empty:
            data['summary'] = arm_rows.iloc[0].to_dict()
            log.info("Model summary loaded for arm: %s", config.SELECTED_ARM)
        else:
            log.warning("ARM '%s' not found in comparison file — summary empty.",
                        config.SELECTED_ARM)
            data['summary'] = {}
    else:
        log.warning("Comparison file not found: %s", config.COMPARISON_FILE)
        data['summary'] = {}

    return data


def make_label_encoder(df: pd.DataFrame) -> LabelEncoder:
    """
    Deterministic, data-driven label encoding.
    Keeps 03_statistics.py consistent with 02_ml_classification.py by fitting on actual class values
    present in the loaded dataset (sorted order).
    """
    le = LabelEncoder().fit(sorted(df['cancer_type'].astype(str).unique()))
    return le


# ============================================================
# TABLE 1: DESCRIPTIVE STATISTICS
# ============================================================
def make_descriptive_table(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Median [IQR] per class for numeric features."""
    log.info("Building descriptive statistics table...")

    exclude = {'patient_id', 'cancer_type', 'multifocal_auth_phase'}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    rows = []
    for feat in numeric_cols:
        row = {'feature': feat}
        for cls in config.CLASSES:
            vals = df[df['cancer_type'] == cls][feat].dropna()
            if len(vals) > 0:
                med = np.median(vals)
                q1  = np.percentile(vals, 25)
                q3  = np.percentile(vals, 75)
                row[f'{cls}_median_IQR'] = f"{med:.3f} [{q1:.3f}–{q3:.3f}]"
                row[f'{cls}_n'] = len(vals)
            else:
                row[f'{cls}_median_IQR'] = 'N/A'
                row[f'{cls}_n'] = 0
        rows.append(row)

    table = pd.DataFrame(rows)
    out = os.path.join(config.OUTPUT_DIR, "descriptive_statistics.csv")
    table.to_csv(out, index=False)
    log.info("Descriptive table saved: %d features", len(table))
    return table


# ============================================================
# TABLE 2: SIGNIFICANT FEATURES (FDR-corrected)
# ============================================================
def make_significance_table(stats_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    if stats_df.empty:
        log.warning("No stats data — skipping significance table.")
        return pd.DataFrame()

    sig = (
        stats_df
        .dropna(subset=['KW_q_fdr'])
        .query(f"KW_q_fdr < {config.FDR_THRESHOLD}")
        .sort_values('eta_squared', ascending=False)
        .reset_index(drop=True)
    )

    log.info("Significant features (q < %.2f): %d / %d",
             config.FDR_THRESHOLD, len(sig), len(stats_df))

    out = os.path.join(config.OUTPUT_DIR, "significant_features_fdr.csv")
    sig.to_csv(out, index=False)
    return sig


# ============================================================
# EXTENDED ANALYSIS 1: FEATURE GROUP ABLATION (biology-first)
# ============================================================
def get_numeric_feature_columns(df: pd.DataFrame) -> list:
    exclude = {'patient_id', 'cancer_type'}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def build_feature_groups(df: pd.DataFrame) -> dict:
    """
    Group features into biologically meaningful sets.
    Prefix conventions from the pipeline:
      - Intratumoral: C?/P_ zone/rim/radial/local_hetero/enh/necrosis...
      - Peritumoral: *_peri_*, *_boundary_*, *_invasion_*, *_capsule_*
      - Background liver: *_tlr_*, liver_ref*
    """
    # Patient-level flags are useful descriptors but not pure imaging signals.
    patient_level = {'is_multifocal', 'n_lesions', 'largest_lesion_fraction'}

    all_num = get_numeric_feature_columns(df)
    imaging = [c for c in all_num if c not in patient_level]

    background = [c for c in imaging if '_tlr_' in c or 'liver_ref' in c]
    peri = [c for c in imaging if (
        '_peri_' in c or '_boundary_' in c or '_invasion_' in c or '_capsule_' in c
    )]
    intra = [c for c in imaging if c not in set(background + peri)]

    return {
        'all_features': all_num,
        'imaging_features': imaging,
        'intratumoral_only': intra,
        'peritumoral_only': peri,
        'background_liver_only': background,
        'intra_plus_peri': sorted(set(intra + peri)),
        'intra_plus_background': sorted(set(intra + background)),
        'peri_plus_background': sorted(set(peri + background)),
    }


def _evaluate_group_cv(X: pd.DataFrame, y: np.ndarray, feature_cols: list,
                       config: Config) -> dict:
    if len(feature_cols) == 0:
        return {'n_features': 0, 'accuracy_mean': np.nan, 'accuracy_std': np.nan,
                'f1_macro_mean': np.nan, 'f1_macro_std': np.nan,
                'auc_ovr_mean': np.nan, 'auc_ovr_std': np.nan}

    Xg = X[feature_cols].copy()
    cv = StratifiedKFold(
        n_splits=config.ABLATION_SPLITS, shuffle=True,
        random_state=config.ABLATION_RANDOM_STATE
    )

    accs, f1s, aucs = [], [], []
    for tr, te in cv.split(Xg, y):
        Xtr, Xte = Xg.iloc[tr], Xg.iloc[te]
        ytr, yte = y[tr], y[te]

        # Fold-local preprocessing only (no label leakage):
        # 1) median imputation
        # 2) very-low variance filter
        # 3) high-correlation pruning
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)

        var = Xtr.var(ddof=0)
        keep_var = var[var > 1e-10].index.tolist()
        if len(keep_var) == 0:
            return {'n_features': 0, 'accuracy_mean': np.nan, 'accuracy_std': np.nan,
                    'f1_macro_mean': np.nan, 'f1_macro_std': np.nan,
                    'auc_ovr_mean': np.nan, 'auc_ovr_std': np.nan}
        Xtr = Xtr[keep_var]
        Xte = Xte[keep_var]

        corr = Xtr.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = [c for c in upper.columns if any(upper[c] > config.CORR_CLUSTER_THRESHOLD)]
        if to_drop:
            Xtr = Xtr.drop(columns=to_drop)
            Xte = Xte.drop(columns=[c for c in to_drop if c in Xte.columns])

        sw = compute_sample_weight('balanced', ytr)
        model = XGBClassifier(**config.ABLATION_XGB)
        model.fit(Xtr, ytr, sample_weight=sw)

        yhat = model.predict(Xte)
        ypr = model.predict_proba(Xte)

        accs.append(accuracy_score(yte, yhat))
        f1s.append(f1_score(yte, yhat, average='macro'))
        try:
            aucs.append(roc_auc_score(yte, ypr, multi_class='ovr', average='macro'))
        except ValueError:
            aucs.append(np.nan)

    return {
        'n_features': len(feature_cols),
        'accuracy_mean': float(np.nanmean(accs)),
        'accuracy_std': float(np.nanstd(accs)),
        'f1_macro_mean': float(np.nanmean(f1s)),
        'f1_macro_std': float(np.nanstd(f1s)),
        'auc_ovr_mean': float(np.nanmean(aucs)),
        'auc_ovr_std': float(np.nanstd(aucs)),
    }


def run_ablation_analysis(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    log.info("Running feature-group ablation analysis...")
    X = df[get_numeric_feature_columns(df)].copy()
    le = make_label_encoder(df)
    y = le.transform(df['cancer_type'].astype(str).values)

    groups = build_feature_groups(df)
    rows = []
    for gname, cols in groups.items():
        res = _evaluate_group_cv(X, y, cols, config)
        res['group'] = gname
        rows.append(res)

    ab = pd.DataFrame(rows).sort_values('auc_ovr_mean', ascending=False)
    out = os.path.join(config.OUTPUT_DIR, "ablation_feature_groups.csv")
    ab.to_csv(out, index=False)
    note_path = os.path.join(config.OUTPUT_DIR, "ablation_method_note.txt")
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(
            "Ablation methodology note:\n"
            "- Group definitions are feature-name based.\n"
            "- Within each CV fold, preprocessing is train-only: median imputation, "
            "low-variance removal, and high-correlation pruning.\n"
            "- This ablation is intended for mechanistic comparison across feature groups, "
            "not as a replacement for the full nested-CV pipeline.\n"
        )
    log.info("Ablation table saved: %s", out)
    return ab


def plot_ablation_performance(ablation_df: pd.DataFrame, config: Config):
    if ablation_df.empty:
        return
    d = ablation_df.copy()
    d['label'] = d['group'] + " (n=" + d['n_features'].astype(int).astype(str) + ")"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    order = d.sort_values('f1_macro_mean', ascending=True)
    axes[0].barh(order['label'], order['f1_macro_mean'], color='#3498db', alpha=0.85)
    axes[0].set_xlabel('F1 Macro (mean)')
    axes[0].set_title('Ablation by Feature Group — F1 Macro', fontsize=11, fontweight='bold')

    order2 = d.sort_values('auc_ovr_mean', ascending=True)
    axes[1].barh(order2['label'], order2['auc_ovr_mean'], color='#9b59b6', alpha=0.85)
    axes[1].set_xlabel('AUC OvR (mean)')
    axes[1].set_title('Ablation by Feature Group — AUC', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "ablation_feature_groups.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("[Plot] Ablation feature groups saved.", flush=True)


# ============================================================
# EXTENDED ANALYSIS 2: SHAP ↔ Statistical Concordance
# ============================================================
def build_shap_kw_concordance(sig_df: pd.DataFrame, shap_df: pd.DataFrame,
                              config: Config) -> pd.DataFrame:
    if sig_df.empty or shap_df.empty:
        return pd.DataFrame()

    # Robust key normalization: avoid object/int merge crashes
    left = sig_df[['feature', 'eta_squared', 'KW_q_fdr']].copy()
    right = shap_df[['feature', 'shap_weighted_mean_abs']].copy()
    left['feature'] = left['feature'].astype(str).str.strip()
    right['feature'] = right['feature'].astype(str).str.strip()

    merged = pd.merge(
        left,
        right,
        on='feature', how='inner'
    ).dropna()

    if merged.empty:
        return merged

    rho, pval = stats.spearmanr(merged['eta_squared'], merged['shap_weighted_mean_abs'])
    merged['spearman_rho_eta_vs_shap'] = rho
    merged['spearman_p'] = pval

    out = os.path.join(config.OUTPUT_DIR, "shap_kw_concordance.csv")
    merged.sort_values('shap_weighted_mean_abs', ascending=False).to_csv(out, index=False)
    log.info("SHAP-KW concordance saved: %s (rho=%.3f, p=%.3g)", out, rho, pval)
    return merged


# ============================================================
# EXTENDED ANALYSIS 3: Correlation families of significant features
# ============================================================
def build_correlation_families(df: pd.DataFrame, sig_df: pd.DataFrame,
                               shap_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    if sig_df.empty:
        return pd.DataFrame()

    sig_feats = [f for f in sig_df['feature'].tolist() if f in df.columns]
    sig_feats = [f for f in sig_feats if np.issubdtype(df[f].dtype, np.number)]
    if len(sig_feats) < 2:
        return pd.DataFrame()

    corr = df[sig_feats].corr().abs().fillna(0.0)
    thresh = config.CORR_CLUSTER_THRESHOLD
    visited = set()
    families = []

    stat_rank = (
        sig_df.assign(feature=sig_df['feature'].astype(str).str.strip())
             .set_index('feature')['eta_squared']
             .to_dict()
    )
    shap_rank = {}
    if not shap_df.empty:
        shap_rank = (
            shap_df.assign(feature=shap_df['feature'].astype(str).str.strip())
                   .set_index('feature')['shap_weighted_mean_abs']
                   .to_dict()
        )

    for f in sig_feats:
        if f in visited:
            continue
        stack = [f]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            neigh = corr.index[corr.loc[cur] >= thresh].tolist()
            for n in neigh:
                if n not in comp:
                    stack.append(n)
        families.append(sorted(comp))

    rows = []
    for i, fam in enumerate(families, start=1):
        rep = sorted(
            fam,
            key=lambda x: (stat_rank.get(x, -1), shap_rank.get(x, -1)),
            reverse=True
        )[0]
        rows.append({
            'family_id': i,
            'family_size': len(fam),
            'representative_feature': rep,
            'representative_eta_squared': stat_rank.get(rep, np.nan),
            'representative_shap': shap_rank.get(rep, np.nan),
            'members': ';'.join(fam),
        })

    fam_df = pd.DataFrame(rows).sort_values(
        ['family_size', 'representative_eta_squared'], ascending=[False, False]
    )
    out = os.path.join(config.OUTPUT_DIR, "significant_feature_families.csv")
    fam_df.to_csv(out, index=False)
    log.info("Correlation families saved: %s (%d families)", out, len(fam_df))
    return fam_df


# ============================================================
# EXTENDED ANALYSIS 4: Permutation test (model-level significance)
# ============================================================
def permutation_test_all_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    # Skip if result already exists and flag is set (arm change re-runs don't need new test)
    if getattr(config, 'SKIP_PERMUTATION_TEST', False):
        existing = os.path.join(config.OUTPUT_DIR, "permutation_test_all_features.csv")
        if os.path.exists(existing):
            log.info("SKIP_PERMUTATION_TEST=True — loading existing result: %s", existing)
            return pd.read_csv(existing)
        log.warning("SKIP_PERMUTATION_TEST=True but no existing file — running test.")

    groups = build_feature_groups(df)
    imaging_cols = groups.get('imaging_features', [])
    log.info("Running permutation test (imaging features, n=%d)...", config.PERMUTATION_N)
    if len(imaging_cols) == 0:
        raise ValueError(
            "Permutation aborted: 'imaging_features' is empty. "
            "Check build_feature_groups() output and feature columns."
        )

    X = df[imaging_cols].copy()
    le = make_label_encoder(df)
    y = le.transform(df['cancer_type'].astype(str).values)

    # observed
    obs = _evaluate_group_cv(X, y, imaging_cols, config)
    observed_f1 = obs['f1_macro_mean']

    # permutations
    rng = np.random.default_rng(config.ABLATION_RANDOM_STATE)
    perm_scores = []
    for _ in range(config.PERMUTATION_N):
        yp = rng.permutation(y)
        sc = _evaluate_group_cv(X, yp, imaging_cols, config)
        perm_scores.append(sc['f1_macro_mean'])

    perm_scores = np.array(perm_scores, dtype=float)
    p_val = float((np.sum(perm_scores >= observed_f1) + 1) / (len(perm_scores) + 1))

    out_df = pd.DataFrame([{
        'observed_f1_macro_mean': observed_f1,
        'perm_mean_f1_macro': float(np.nanmean(perm_scores)),
        'perm_std_f1_macro': float(np.nanstd(perm_scores)),
        'permutation_n': int(config.PERMUTATION_N),
        'p_value': p_val,
    }])
    out = os.path.join(config.OUTPUT_DIR, "permutation_test_all_features.csv")
    out_df.to_csv(out, index=False)
    log.info("Permutation test saved: %s (p=%.4g)", out, p_val)

    note_path = os.path.join(config.OUTPUT_DIR, "permutation_method_note.txt")
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(
            "Permutation test methodology note:\n"
            "- Null hypothesis: imaging features do not classify better than chance.\n"
            "- Model: simplified fixed-hyperparameter XGBoost (ABLATION_XGB), NOT the\n"
            "  full nested-tuned pipeline from hccmakale_ml.py. This is computationally\n"
            "  tractable and appropriate for an omnibus significance test, but tests a\n"
            "  surrogate model, not the reported nested-CV AUC=0.943 pipeline.\n"
            "- Methods must state: 'Permutation significance was assessed using a\n"
            "  simplified fixed-hyperparameter surrogate (n_estimators=200, max_depth=4)\n"
            "  for computational tractability (B=%d permutations).'\n"
            "- Correlation pruning threshold: config.CORR_CLUSTER_THRESHOLD = %.2f\n"
            "- p-value formula: (sum(perm >= observed) + 1) / (B + 1) [Phipson & Smyth 2010]\n"
            % (config.PERMUTATION_N, config.CORR_CLUSTER_THRESHOLD)
        )

    return out_df


def plot_shap_kw_scatter(conc_df: pd.DataFrame, config: Config):
    if conc_df.empty:
        return
    rho = conc_df['spearman_rho_eta_vs_shap'].iloc[0]
    pval = conc_df['spearman_p'].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(conc_df['eta_squared'], conc_df['shap_weighted_mean_abs'],
               alpha=0.65, s=26, color='#2c7fb8')
    ax.set_xlabel('Effect Size (η²)')
    ax.set_ylabel('SHAP Weighted Mean |value|')
    ax.set_title(f'SHAP vs Statistical Effect Size\nSpearman rho={rho:.3f}, p={pval:.3g}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "shap_kw_concordance_scatter.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("[Plot] SHAP-KW concordance scatter saved.", flush=True)


# ============================================================
# EXTENDED ANALYSIS v4.2: Bias/Mechanism block
# ============================================================
def _run_oof_for_feature_set(X: pd.DataFrame, y: np.ndarray, feature_cols: list,
                             config: Config) -> dict:
    cols = [c for c in feature_cols if c in X.columns]
    if len(cols) == 0:
        return {'metrics': {}, 'oof_true': np.array([]), 'oof_pred': np.array([]),
                'oof_prob': np.array([[]]), 'used_features': []}

    Xg = X[cols].copy()
    cv = StratifiedKFold(
        n_splits=config.ABLATION_SPLITS, shuffle=True,
        random_state=config.ABLATION_RANDOM_STATE
    )

    oof_true = np.empty(len(y), dtype=int)
    oof_pred = np.empty(len(y), dtype=int)
    oof_prob = np.zeros((len(y), len(config.CLASSES)), dtype=float)

    for tr, te in cv.split(Xg, y):
        Xtr, Xte = Xg.iloc[tr], Xg.iloc[te]
        ytr, yte = y[tr], y[te]

        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)

        var = Xtr.var(ddof=0)
        keep = var[var > 1e-10].index.tolist()
        Xtr = Xtr[keep]
        Xte = Xte[[c for c in keep if c in Xte.columns]]

        corr = Xtr.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = [c for c in upper.columns if any(upper[c] > config.CORR_CLUSTER_THRESHOLD)]
        if to_drop:
            Xtr = Xtr.drop(columns=to_drop)
            Xte = Xte.drop(columns=[c for c in to_drop if c in Xte.columns])

        sw = compute_sample_weight('balanced', ytr)
        m = XGBClassifier(**config.ABLATION_XGB)
        m.fit(Xtr, ytr, sample_weight=sw)

        yp = m.predict(Xte)
        ypr = m.predict_proba(Xte)
        oof_true[te] = yte
        oof_pred[te] = yp
        oof_prob[te] = ypr

    metrics = {
        'accuracy': float(accuracy_score(oof_true, oof_pred)),
        'f1_macro': float(f1_score(oof_true, oof_pred, average='macro')),
    }
    try:
        metrics['auc_ovr'] = float(roc_auc_score(oof_true, oof_prob, multi_class='ovr', average='macro'))
    except ValueError:
        metrics['auc_ovr'] = np.nan
    return {'metrics': metrics, 'oof_true': oof_true, 'oof_pred': oof_pred, 'oof_prob': oof_prob, 'used_features': cols}


def _global_shap_importance(X: pd.DataFrame, y: np.ndarray, feature_cols: list,
                            config: Config) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in X.columns]
    if len(cols) == 0:
        return pd.DataFrame(columns=['feature', 'shap_mean_abs'])
    Xg = X[cols].copy()
    med = Xg.median(numeric_only=True)
    Xg = Xg.fillna(med)
    sw = compute_sample_weight('balanced', y)
    m = XGBClassifier(**config.ABLATION_XGB)
    m.fit(Xg, y, sample_weight=sw)
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(Xg.values)

    # Robust multi-class handling across SHAP versions:
    # - list[class] of (n_samples, n_features)
    # - ndarray (n_samples, n_features, n_classes)
    # - ndarray (n_classes, n_samples, n_features)
    # - ndarray (n_samples, n_features)
    if isinstance(sv, list):
        imp = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            # Identify feature axis by matching n_features (= len(cols))
            feat_axis = None
            for ax in range(3):
                if arr.shape[ax] == len(cols):
                    feat_axis = ax
                    break
            if feat_axis is None:
                # Fallback: assume middle axis is feature
                feat_axis = 1
            # Reduce all non-feature axes
            reduce_axes = tuple(ax for ax in range(3) if ax != feat_axis)
            imp = np.abs(arr).mean(axis=reduce_axes)
        elif arr.ndim == 2:
            imp = np.abs(arr).mean(axis=0)
        else:
            imp = np.ravel(np.abs(arr))

    imp = np.asarray(imp).ravel()
    n = min(len(cols), len(imp))
    out_df = pd.DataFrame({
        'feature': cols[:n],
        'shap_mean_abs': imp[:n],
    }).sort_values('shap_mean_abs', ascending=False)
    return out_df


def run_v42_bias_mechanism(df: pd.DataFrame, config: Config,
                            sig_df: pd.DataFrame = None) -> dict:
    log.info("Running v4.2 bias/mechanism analyses...")
    X = df[get_numeric_feature_columns(df)].copy()
    le = make_label_encoder(df)
    y = le.transform(df['cancer_type'].astype(str).values)
    groups = build_feature_groups(df)

    all_cols = groups['all_features']
    tumor_only = groups['intratumoral_only']
    no_background_abs = [c for c in all_cols if 'liver_ref' not in c]  # keep TLR ratios
    background_plus_peri = sorted(set(groups['background_liver_only'] + groups['peritumoral_only']))

    model_sets = {
        'comprehensive': all_cols,
        'tumor_only': tumor_only,
        'no_background_absolute': no_background_abs,
        'background_plus_peri_only': background_plus_peri,
    }

    rows = []
    model_runs = {}
    for name, cols in model_sets.items():
        r = _run_oof_for_feature_set(X, y, cols, config)
        model_runs[name] = r
        rows.append({
            'model': name,
            'n_features': len(cols),
            'accuracy': r['metrics'].get('accuracy', np.nan),
            'f1_macro': r['metrics'].get('f1_macro', np.nan),
            'auc_ovr': r['metrics'].get('auc_ovr', np.nan),
        })

    ab = pd.DataFrame(rows).sort_values('auc_ovr', ascending=False)
    ab.to_csv(os.path.join(config.OUTPUT_DIR, "ablation_models_comparison.csv"), index=False)

    # Error analysis on comprehensive model
    comp = model_runs['comprehensive']
    inv = {i: c for i, c in enumerate(config.CLASSES)}
    pred_label = np.vectorize(inv.get)(comp['oof_pred'])
    true_label = np.vectorize(inv.get)(comp['oof_true'])
    err_df = pd.DataFrame({
        'patient_id': df['patient_id'].values,
        'true_label': true_label,
        'pred_label': pred_label,
        'is_error': (comp['oof_true'] != comp['oof_pred']).astype(int),
    })
    # Dynamic top feature — re-derives from sig_df after TLR fix rather than
    # hard-coding C1_tlr_liver_ref_median which may no longer rank #1
    if sig_df is not None and not sig_df.empty and 'feature' in sig_df.columns:
        key_col = str(sig_df.iloc[0]['feature'])
        log.info("Error analysis key feature (dynamic): %s", key_col)
    else:
        key_col = 'C1_tlr_liver_ref_median'
        log.info("Error analysis key feature (fallback): %s", key_col)
    if key_col in df.columns:
        err_df[key_col] = df[key_col].values
    err_df.to_csv(os.path.join(config.OUTPUT_DIR, "error_analysis_misclassified.csv"), index=False)

    # SHAP shift: comprehensive vs tumor-only
    shap_comp = _global_shap_importance(X, y, model_sets['comprehensive'], config)
    shap_tum = _global_shap_importance(X, y, model_sets['tumor_only'], config)
    m = pd.merge(
        shap_comp.rename(columns={'shap_mean_abs': 'shap_comprehensive'}),
        shap_tum.rename(columns={'shap_mean_abs': 'shap_tumor_only'}),
        on='feature', how='outer'
    ).fillna(0.0)
    m['rank_comp'] = m['shap_comprehensive'].rank(method='min', ascending=False)
    m['rank_tumor'] = m['shap_tumor_only'].rank(method='min', ascending=False)
    m['rank_shift'] = m['rank_tumor'] - m['rank_comp']
    m.sort_values('shap_comprehensive', ascending=False).to_csv(
        os.path.join(config.OUTPUT_DIR, "shap_shift_comparison.csv"), index=False
    )

    # Discussion-ready summary
    disc = ab.copy()
    disc['interpretation_hint'] = ''
    disc.loc[disc['model'] == 'tumor_only', 'interpretation_hint'] = 'Tumor-internal signal only'
    disc.loc[disc['model'] == 'background_plus_peri_only', 'interpretation_hint'] = 'Microenvironment/background reliance'
    disc.loc[disc['model'] == 'no_background_absolute', 'interpretation_hint'] = 'Absolute liver removed; ratio-driven context'
    disc.loc[disc['model'] == 'comprehensive', 'interpretation_hint'] = 'Full biologic context'
    disc.to_csv(os.path.join(config.OUTPUT_DIR, "discussion_ready_summary.csv"), index=False)

    return {
        'ablation_models': ab,
        'errors': err_df,
        'shap_shift': m,
        'key_col': key_col,
    }


def plot_v42_outputs(v42: dict, config: Config):
    if not v42:
        return
    ab = v42.get('ablation_models', pd.DataFrame())
    if not ab.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        d = ab.sort_values('auc_ovr', ascending=True)
        ax.barh(d['model'], d['auc_ovr'], color='#34495e', alpha=0.85)
        ax.set_xlabel('AUC OvR')
        ax.set_title('v4.2 Model Comparison (Bias/Mechanism)')
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "ablation_models_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("[Plot] v4.2 ablation model comparison saved.", flush=True)

    err = v42.get('errors', pd.DataFrame())
    key_col = v42.get('key_col', 'C1_tlr_liver_ref_median')
    if (not err.empty) and (key_col in err.columns):
        sub = err[err['true_label'].isin(['ICC', 'HCC'])].copy()
        if not sub.empty:
            sub['group'] = np.where(
                (sub['true_label'] == 'ICC') & (sub['pred_label'] == 'HCC'),
                'ICC->HCC (misclassified)',
                np.where(sub['true_label'] == 'ICC', 'ICC (correct/other)', 'HCC')
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=sub, x='group', y=key_col, ax=ax)
            ax.set_title(f'Error Analysis: {key_col}')
            ax.tick_params(axis='x', rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(config.OUTPUT_DIR, "error_analysis_liver_ref.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("[Plot] v4.2 error analysis saved.", flush=True)

# ============================================================
# EXTENDED ANALYSIS 5: FEATURE / FAMILY STABILITY (fold recurrence)
# ============================================================
def parse_feature_name(fname: str) -> dict:
    """
    Parse a feature column name into {phase, region, stat_type, family_key}.
    Searches substrings in the FULL feature name with priority ordering.
    family_key = phase + '_' + region groups correlated feature variants.

    Ordering notes (critical):
    - 'capsule' before 'boundary': capsule features have both substrings
      (e.g. C2_boundary_capsule_hu_diff), so boundary would steal them otherwise.
    - 'zone_' before 'radial_': belt-and-suspenders against future naming overlap.
    - 'ring_enhancement', 'rim_', 'core_' all map to 'ring_core': they are produced
      by the same extraction block and represent the same ring-core contrast signal.
    - 'high_hetero_fraction' before 'local_hetero': more specific match needed
      because high_hetero_fraction does not contain 'local_hetero' as a substring.
    - Intratumoral blocks use NO block-name prefix (results[f'{phase}_{k}']),
      so 'angular_' matches angular_median_cv etc., NOT the old 'angular_hetero'.
      Similarly 'radial_' (not 'radial_profile') and 'enh_' (not 'enh').
    """
    phases = {'P', 'C1', 'C2', 'C3'}
    phase  = fname.split('_')[0] if fname.split('_')[0] in phases else 'patient_level'

    region_patterns = [
        # Peritumoral — specific before general
        ('peri_10mm',            'peri_10mm'),
        ('peri_5mm',             'peri_5mm'),
        ('capsule',              'capsule'),       # must precede 'boundary'
        ('boundary',             'boundary'),
        ('invasion',             'invasion'),
        ('tlr',                  'tlr'),
        # Intratumoral — specific before general
        ('zone_',                'radial_zone'),   # must precede 'radial_'
        ('radial_',              'radial_profile'),
        ('ring_enhancement',     'ring_core'),     # boolean flag
        ('rim_',                 'ring_core'),     # rim_median, rim_core_diff, etc.
        ('core_',                'ring_core'),     # core_median, core_entropy, etc.
        ('angular_',             'angular'),       # angular_median_cv, angular_range, etc.
        ('high_hetero_fraction', 'local_hetero'),  # must precede 'local_hetero'
        ('local_hetero',         'local_hetero'),
        ('necrosis',             'necrosis'),
        ('enh_',                 'enh'),
    ]

    region = next(
        (label for pattern, label in region_patterns if pattern in fname),
        'other'
    )

    return {
        'phase':      phase,
        'region':     region,
        'stat_type':  fname,
        'family_key': f"{phase}_{region}",
    }


def build_feature_stability(config: Config) -> pd.DataFrame:
    """
    Read fold_selected_features_{arm}.csv (produced by ML script) and compute:
      - Per-feature selection rate across outer folds
      - Per-family selection rate (≥1 family member selected in fold)
    Saves feature_stability_{arm}.csv and family_stability_{arm}.csv.
    """
    if not os.path.exists(config.FOLD_FEATURES_FILE):
        log.warning("Fold features file not found: %s — skipping stability analysis.",
                    config.FOLD_FEATURES_FILE)
        return pd.DataFrame()

    fold_df = pd.read_csv(config.FOLD_FEATURES_FILE)
    if fold_df.empty:
        return pd.DataFrame()

    n_folds = fold_df['fold'].nunique()
    log.info("Feature stability: %d folds, %d feature selections (arm=%s)",
             n_folds, len(fold_df), config.SELECTED_ARM)

    # Feature-level recurrence
    feat_counts = (
        fold_df.groupby('feature')['fold']
        .count()
        .reset_index()
        .rename(columns={'fold': 'n_folds_selected'})
    )
    feat_counts['selection_rate'] = feat_counts['n_folds_selected'] / n_folds
    parsed = feat_counts['feature'].apply(parse_feature_name).apply(pd.Series)
    feat_counts = pd.concat([feat_counts, parsed], axis=1)
    feat_counts = feat_counts.sort_values('selection_rate', ascending=False)

    # Family-level: fraction of folds where ≥1 family member was selected
    fam_rows = []
    for fkey, grp in feat_counts.groupby('family_key'):
        n_folds_any = fold_df[fold_df['feature'].isin(grp['feature'])]['fold'].nunique()
        top_row = grp.loc[grp['n_folds_selected'].idxmax()]
        fam_rows.append({
            'family_key':            fkey,
            'n_members':             len(grp),
            'n_folds_any_member':    n_folds_any,
            'family_selection_rate': n_folds_any / n_folds,
            'top_member':            top_row['feature'],
            'top_member_rate':       top_row['n_folds_selected'] / n_folds,
        })
    fam_df = pd.DataFrame(fam_rows).sort_values('family_selection_rate', ascending=False)

    arm = config.SELECTED_ARM
    feat_counts.to_csv(
        os.path.join(config.OUTPUT_DIR, f"feature_stability_{arm}.csv"), index=False
    )
    fam_df.to_csv(
        os.path.join(config.OUTPUT_DIR, f"family_stability_{arm}.csv"), index=False
    )
    log.info("Stability saved: %d features, %d families (arm=%s)",
             len(feat_counts), len(fam_df), arm)

    # Log top stable and top unstable families
    top_stable   = fam_df.head(5)[['family_key', 'family_selection_rate', 'n_members']].to_string(index=False)
    top_unstable = fam_df.tail(5)[['family_key', 'family_selection_rate', 'n_members']].to_string(index=False)
    log.info("Most stable families:\n%s", top_stable)
    log.info("Least stable families:\n%s", top_unstable)

    return feat_counts


# ============================================================
# PLOT 1: CLASS DISTRIBUTION
# ============================================================
def plot_class_distribution(df: pd.DataFrame, config: Config):
    counts = df['cancer_type'].value_counts()
    colors = [config.CLASS_COLORS.get(c, '#888') for c in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Pie
    axes[0].pie(
        counts.values, labels=counts.index, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')

    # Bar
    bars = axes[1].bar(counts.index, counts.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     val + 1, str(val), ha='center', fontsize=11)
    axes[1].set_ylabel('Number of Patients')
    axes[1].set_title('Patient Count per Class', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Class distribution saved.", flush=True)


# ============================================================
# PLOT 2: TOP FEATURE BOXPLOTS (FDR significant, sorted by η²)
# ============================================================
def plot_top_feature_boxplots(df: pd.DataFrame, sig_df: pd.DataFrame,
                               config: Config):
    if sig_df.empty:
        log.warning("No significant features to plot.")
        return

    top_feats = sig_df.head(config.TOP_N_FEATURES)['feature'].tolist()
    n_cols = 4
    n_rows = (len(top_feats) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.5, n_rows * 4))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, top_feats):
        data_per_class = [
            df[df['cancer_type'] == cls][feat].dropna().values
            for cls in config.CLASSES
        ]
        bp = ax.boxplot(data_per_class, labels=config.CLASSES,
                        patch_artist=True, notch=False,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, cls in zip(bp['boxes'], config.CLASSES):
            patch.set_facecolor(config.CLASS_COLORS.get(cls, '#888'))
            patch.set_alpha(0.75)

        row = sig_df[sig_df['feature'] == feat]
        if not row.empty:
            r = row.iloc[0]
            eta = r.get('eta_squared', np.nan)
            q   = r.get('KW_q_fdr', np.nan)
            title = f"{feat}\nq={q:.3f}, η²={eta:.3f}"
        else:
            title = feat
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(axis='x', labelsize=8)

    for ax in axes[len(top_feats):]:
        ax.set_visible(False)

    plt.suptitle(f'Top {len(top_feats)} Features by Effect Size (FDR q<{config.FDR_THRESHOLD})',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "top_features_boxplots.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Top feature boxplots saved.", flush=True)


# ============================================================
# PLOT 3: EFFECT SIZE RANKING (η² bar chart)
# ============================================================
def plot_effect_size_ranking(sig_df: pd.DataFrame, config: Config,
                              top_n: int = 25):
    if sig_df.empty:
        return

    top = sig_df.dropna(subset=['eta_squared']).head(top_n)
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(top)))
    ax.barh(range(len(top)), top['eta_squared'].values[::-1],
            color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=8)
    ax.set_xlabel("Effect Size (η²)")
    ax.set_title(f"Top {len(top)} Features — Effect Size Ranking\n"
                 f"(Kruskal-Wallis, FDR q < {config.FDR_THRESHOLD})",
                 fontsize=11, fontweight='bold')
    ax.axvline(0.01, color='gray', linestyle='--', lw=1, label='small (0.01)')
    ax.axvline(0.06, color='orange', linestyle='--', lw=1, label='medium (0.06)')
    ax.axvline(0.14, color='red', linestyle='--', lw=1, label='large (0.14)')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "effect_size_ranking.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Effect size ranking saved.", flush=True)


# ============================================================
# PLOT 4: SHAP IMPORTANCE
# ============================================================
def plot_shap_importance(shap_df: pd.DataFrame, config: Config):
    if shap_df.empty:
        log.warning("No SHAP data.")
        return

    top = shap_df.head(config.TOP_N_SHAP)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.4)))
    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(top)))
    ax.barh(range(len(top)),
            top['shap_weighted_mean_abs'].values[::-1],
            color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=8)
    ax.set_xlabel("Weighted Mean |SHAP Value| (OOF)")
    ax.set_title(f"Top {len(top)} Features — SHAP OOF Aggregate\n"
                 "(Outer-fold aggregate, not final refit)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "shap_importance.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] SHAP importance saved.", flush=True)


# ============================================================
# PLOT 5: CV FOLD PERFORMANCE
# ============================================================
def plot_cv_performance(cv_df: pd.DataFrame, summary: dict, config: Config):
    if cv_df.empty:
        return

    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'auc_ovr']
    labels  = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'AUC (OvR)']
    colors  = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = cv_df[metric].values
        ax.bar(range(1, len(vals) + 1), vals, color=color, alpha=0.8,
               edgecolor='white')
        ax.axhline(vals.mean(), color='black', linestyle='--', lw=2,
                   label=f'Mean: {vals.mean():.3f}')
        ax.fill_between(
            [0.5, len(vals) + 0.5],
            vals.mean() - vals.std(),
            vals.mean() + vals.std(),
            alpha=0.15, color=color, label=f'±SD: {vals.std():.3f}'
        )
        ax.set_xlabel('Fold')
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylim([max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.05)])
        ax.set_xticks(range(1, len(vals) + 1))
        ax.legend(fontsize=8)

    plt.suptitle('Nested CV Performance (5-fold outer, 3-fold inner)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "cv_performance.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] CV performance saved.", flush=True)


# ============================================================
# PLOT 6: FEATURE CORRELATION HEATMAP
# ============================================================
def plot_correlation_heatmap(df: pd.DataFrame, config: Config, top_n: int = 30):
    exclude = {'patient_id', 'cancer_type'}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    # Top N by variance
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_cols  = variances.head(top_n).index.tolist()
    corr      = df[top_cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap='RdBu_r', center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.3,
        cbar_kws={'shrink': 0.8}, ax=ax, annot=False
    )
    ax.set_title(f'Feature Correlation Matrix (Top {top_n} by Variance)',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "feature_correlation_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Correlation heatmap saved.", flush=True)


# ============================================================
# PLOT 7: RADIAL PROFILES BY PHASE
# ============================================================
def plot_radial_profiles(df: pd.DataFrame, config: Config):
    phases = ['C1', 'C2', 'C3']
    n_bins = 5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, phase in zip(axes, phases):
        for cls in config.CLASSES:
            subset = df[df['cancer_type'] == cls]
            means, sems = [], []
            for z in range(n_bins):
                col = f'{phase}_zone_{z}_median'
                if col not in subset.columns:
                    means.append(np.nan); sems.append(np.nan)
                    continue
                vals = subset[col].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem()  if len(vals) > 0 else np.nan)

            x = list(range(n_bins))
            means = np.array(means)
            sems  = np.array(sems)

            valid = ~np.isnan(means)
            if valid.sum() < 2:
                continue

            color = config.CLASS_COLORS.get(cls, '#888')
            ax.plot(np.array(x)[valid], means[valid], 'o-',
                    color=color, lw=2, ms=6, label=cls)
            ax.fill_between(
                np.array(x)[valid],
                (means - sems)[valid],
                (means + sems)[valid],
                color=color, alpha=0.18
            )

        ax.set_xlabel('Zone (0 = Boundary, 4 = Core)')
        ax.set_ylabel('Median HU')
        ax.set_title(
            f"{'Arterial' if phase=='C1' else 'Portal Venous' if phase=='C2' else 'Delayed'} Phase\n"
            f"Radial Enhancement Profile",
            fontsize=10, fontweight='bold'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Intratumoral Radial HU Profiles by Cancer Type',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "radial_profiles.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Radial profiles saved.", flush=True)


# ============================================================
# PLOT 8: PHASE COMPARISON — KEY FEATURES ACROSS PHASES
# ============================================================
def plot_phase_comparison(df: pd.DataFrame, config: Config):
    """Enhancement and heterogeneity across all 4 phases per class."""

    enh_col  = 'enh_median'
    hetero_col = 'local_hetero_median'
    phases   = ['P', 'C1', 'C2', 'C3']
    phase_labels = {'P': 'Pre-contrast', 'C1': 'Arterial',
                    'C2': 'Portal Venous', 'C3': 'Delayed'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, feat_base, ylabel, title in [
        (axes[0], enh_col,   'Median HU',
         'Tumour Enhancement Across Phases'),
        (axes[1], hetero_col, 'Local Heterogeneity (Median)',
         '3D Local Heterogeneity Across Phases'),
    ]:
        x = np.arange(len(phases))
        width = 0.25
        offsets = np.linspace(-width, width, len(config.CLASSES))

        for cls, offset in zip(config.CLASSES, offsets):
            subset = df[df['cancer_type'] == cls]
            means, sems = [], []
            for phase in phases:
                col = f'{phase}_{feat_base}'
                if col not in subset.columns:
                    means.append(np.nan); sems.append(np.nan)
                    continue
                vals = subset[col].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem()  if len(vals) > 0 else np.nan)

            means = np.array(means)
            sems  = np.array(sems)
            color = config.CLASS_COLORS.get(cls, '#888')
            ax.bar(x + offset, means, width * 0.9,
                   color=color, alpha=0.8, label=cls,
                   yerr=sems, capsize=3, error_kw={'linewidth': 1})

        ax.set_xticks(x)
        ax.set_xticklabels([phase_labels[p] for p in phases], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "phase_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Phase comparison saved.", flush=True)


# ============================================================
# PRINT: SUMMARY TABLE TO CONSOLE
# ============================================================
def print_summary(data: dict, sig_df: pd.DataFrame, config: Config,
                  ablation_df: pd.DataFrame = pd.DataFrame(),
                  conc_df: pd.DataFrame = pd.DataFrame(),
                  perm_df: pd.DataFrame = pd.DataFrame()):
    summary = data.get('summary', {})
    cv_df   = data.get('cv', pd.DataFrame())

    print("\n" + "=" * 65, flush=True)
    print("ANALYSIS SUMMARY", flush=True)
    print("=" * 65, flush=True)

    print(f"\nDataset: {summary.get('n_patients', '?')} patients", flush=True)
    print(f"Features (raw): {summary.get('n_features_raw', '?')}", flush=True)

    print(f"\nClass distribution:", flush=True)
    df = data['features']
    for cls in config.CLASSES:
        n = (df['cancer_type'] == cls).sum()
        print(f"  {cls}: {n}", flush=True)

    print(f"\nStatistical comparison:", flush=True)
    stats_df = data.get('stats', pd.DataFrame())
    if not stats_df.empty:
        total = len(stats_df)
        n_sig = len(sig_df)
        print(f"  Features tested: {total}", flush=True)
        print(f"  Significant (FDR q<{config.FDR_THRESHOLD}): {n_sig}", flush=True)
        if not sig_df.empty:
            top5 = sig_df.head(5)
            print(f"\n  Top 5 by effect size (η²):", flush=True)
            for _, row in top5.iterrows():
                print(f"    {row['feature']:<45} "
                      f"η²={row.get('eta_squared', np.nan):.3f}  "
                      f"q={row.get('KW_q_fdr', np.nan):.4f}", flush=True)

    if not cv_df.empty:
        print(f"\nNested CV Performance:", flush=True)
        for metric, label in [('accuracy','Accuracy'), ('f1_macro','F1 Macro'),
                               ('auc_ovr','AUC OvR')]:
            if metric in cv_df.columns:
                v = cv_df[metric]
                print(f"  {label:<12}: {v.mean():.3f} ± {v.std():.3f}", flush=True)

    if not ablation_df.empty:
        print(f"\nAblation (feature groups):", flush=True)
        top_ab = ablation_df.sort_values('auc_ovr_mean', ascending=False).head(4)
        for _, r in top_ab.iterrows():
            print(f"  {r['group']:<22} AUC={r['auc_ovr_mean']:.3f} "
                  f"F1={r['f1_macro_mean']:.3f} (n_feat={int(r['n_features'])})", flush=True)

    if not conc_df.empty:
        rho = conc_df['spearman_rho_eta_vs_shap'].iloc[0]
        pvl = conc_df['spearman_p'].iloc[0]
        print(f"\nSHAP ↔ KW concordance: rho={rho:.3f}, p={pvl:.3g}", flush=True)

    if not perm_df.empty:
        row = perm_df.iloc[0]
        print(f"\nPermutation test (imaging features):", flush=True)
        print(f"  Observed F1 macro: {row['observed_f1_macro_mean']:.3f}", flush=True)
        print(f"  Permutation p-value: {row['p_value']:.4g}", flush=True)

    print("\n" + "=" * 65 + "\n", flush=True)


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Statistical analysis and visualization for liver tumor radiomics'
    )
    ap.add_argument('--data-dir', default='./data',
                    help='Root data directory (default: ./data)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: <data-dir>/analysis_report_v4)')
    ns = ap.parse_args()

    config = Config()
    config.BASE_DIR    = ns.data_dir
    config.SPATIAL_DIR = f"{ns.data_dir}/spatial_analysis_v4"
    config.ML_DIR      = f"{ns.data_dir}/ml_classification_v4"
    config.OUTPUT_DIR  = ns.output_dir or f"{ns.data_dir}/analysis_report_v4"
    config.FEATURES_FILE     = f"{config.SPATIAL_DIR}/combined_spatial_v4_full.csv"
    config.STATS_FILE        = f"{config.SPATIAL_DIR}/spatial_statistics_v4.csv"
    config.COMPARISON_FILE   = f"{config.ML_DIR}/model_comparison_v4.csv"
    config.CV_RESULTS_FILE   = f"{config.ML_DIR}/nested_cv_fold_results_{config.SELECTED_ARM}.csv"
    config.SHAP_FILE         = f"{config.ML_DIR}/shap_oof_importance_{config.SELECTED_ARM}.csv"
    config.FOLD_FEATURES_FILE = f"{config.ML_DIR}/fold_selected_features_{config.SELECTED_ARM}.csv"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    print("=" * 65, flush=True)
    print("STATISTICAL ANALYSIS & VISUALIZATION (v4.0)", flush=True)
    print("=" * 65, flush=True)

    # ── load ──────────────────────────────────────────────────
    data = load_data(config)
    df   = data['features']

    # ── tables ────────────────────────────────────────────────
    make_descriptive_table(df, config)
    sig_df = make_significance_table(data['stats'], config)

    # ── extended analyses (v4.1) ──────────────────────────────
    stability_df = build_feature_stability(config)
    ablation_df = run_ablation_analysis(df, config)
    conc_df = build_shap_kw_concordance(sig_df, data['shap'], config)
    fam_df = build_correlation_families(df, sig_df, data['shap'], config)
    perm_df = permutation_test_all_features(df, config)
    v42 = {}
    if config.RUN_V42_BIAS_MECHANISM:
        v42 = run_v42_bias_mechanism(df, config, sig_df=sig_df)

    # ── console summary ───────────────────────────────────────
    print_summary(data, sig_df, config, ablation_df, conc_df, perm_df)

    # ── plots ─────────────────────────────────────────────────
    plot_class_distribution(df, config)
    plot_top_feature_boxplots(df, sig_df, config)
    plot_effect_size_ranking(sig_df, config)
    plot_shap_importance(data['shap'], config)
    plot_cv_performance(data['cv'], data['summary'], config)
    plot_correlation_heatmap(df, config)
    plot_radial_profiles(df, config)
    plot_phase_comparison(df, config)
    plot_ablation_performance(ablation_df, config)
    plot_shap_kw_scatter(conc_df, config)
    if config.RUN_V42_BIAS_MECHANISM:
        plot_v42_outputs(v42, config)

    # ── final ─────────────────────────────────────────────────
    print("=" * 65, flush=True)
    print(f"All outputs saved to: {config.OUTPUT_DIR}", flush=True)
    files = sorted(os.listdir(config.OUTPUT_DIR))
    for f in files:
        print(f"  {f}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()

"""
Machine Learning Classification for Primary Liver Tumors
=========================================================
Version 4.0 — Leakage-Free Nested CV Pipeline

Key changes from v3.0:
- P0-C: All preprocessing inside sklearn Pipeline (no leakage)
- P0-C: VarianceFilter and CorrelationFilter as custom transformers
          fit() learns on train fold only; transform() applies stored state
- P0-C: Nested CV — outer loop = performance estimate,
          inner loop = hyperparameter tuning
- P0-C / P0-F: Class weights computed fold-local (no global weight leakage)
- P0-F: VarianceFilter runs AFTER StandardScaler (scale-fair)
- P3-A (SHAP): SHAP computed on outer-fold OOF predictions,
          importance aggregated across folds — not on final refit model
- Config consolidated: single source of truth for all thresholds
"""

import os
import sys
import json
import warnings
import logging
import platform
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
)

from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION  (single source — P2-C)
# ============================================================
class Config:
    BASE_DIR    = "./data"  # set this to your local data directory
    SPATIAL_DIR = f"{BASE_DIR}/spatial_analysis_v4"
    OUTPUT_DIR  = f"{BASE_DIR}/ml_classification_v4"

    FEATURES_FILE = f"{SPATIAL_DIR}/combined_spatial_v4_full.csv"

    CLASSES      = ['HCC', 'ICC', 'CHCC']
    CLASS_LABELS = {0: 'HCC', 1: 'ICC', 2: 'CHCC'}

    # CV structure
    OUTER_SPLITS = 5
    INNER_SPLITS = 3    # inner CV for hyperparameter search
    RANDOM_STATE = 42

    # Preprocessing thresholds (shared with 01_feature_extraction.py via this config)
    CORRELATION_THRESHOLD  = 0.90   # |r| above this → drop lower-variance column
    MIN_VARIANCE_THRESHOLD = 0.01   # after z-score scaling; drops near-constant features
    MAX_FEATURES           = 30     # top-N by XGBoost importance inside inner CV

    # XGBoost base params (tuned inside inner CV)
    XGB_BASE = dict(
        objective        = 'multi:softprob',
        eval_metric      = 'mlogloss',
        use_label_encoder= False,
        random_state     = RANDOM_STATE,
        n_jobs           = 1,          # parallelism handled at CV level
    )

    # Hyperparameter search space
    XGB_PARAM_DIST = dict(
        model__n_estimators      = [100, 200, 300],
        model__max_depth         = [3, 4, 5],
        model__learning_rate     = [0.05, 0.1, 0.15],
        model__subsample         = [0.7, 0.8, 0.9],
        model__colsample_bytree  = [0.7, 0.8, 0.9],
        model__min_child_weight  = [1, 3, 5],
        model__gamma             = [0, 0.1, 0.2],
        model__reg_alpha         = [0, 0.1, 0.5],
        model__reg_lambda        = [0.5, 1.0, 2.0],
    )

    N_ITER_SEARCH = 25      # default RandomizedSearchCV iterations

    # Equal search budget across all arms — required for fair performance comparison.
    # 25 iterations × 3 inner splits × 5 outer folds = 375 calls per arm.
    # For shap_topk this means 375 SHAP computations (probe XGBoost + TreeExplainer);
    # for gain-based arms it is 375 plain probe fits. Runtime difference is real but
    # the method comparison remains unbiased — methods section can state
    # "all four arms were evaluated with identical n_iter=25 randomised search budget."
    ARM_N_ITER = {
        'no_selection':        25,
        'correlation_pruning': 25,
        'full_pipeline':       25,
        'shap_topk':           25,
    }

    # Four selection arms for ablation comparison
    ARMS = ['no_selection', 'correlation_pruning', 'full_pipeline', 'shap_topk']

    # MAX_FEATURES is tuned inside inner CV for full_pipeline arm
    MAX_FEATURES_CANDIDATES = [20, 30, 50, 75]

    # P3-A: if harmonized file exists, point FEATURES_FILE there instead
    # FEATURES_FILE = f"{SPATIAL_DIR}/combined_spatial_v4_harmonized.csv"

    # P3-B: seed everything for reproducibility
    NUMPY_SEED = 42   # set at run start via np.random.seed()

log = logging.getLogger("ml_pipeline")


# ============================================================
# P3-B: REPRODUCIBILITY — run config snapshot
# ============================================================
def save_run_config(config: Config, output_dir: str) -> tuple:
    """
    Save a JSON snapshot of Config + library versions + Python/platform.
    Returns (run_id, path).
    """
    def _ver(mod_name: str) -> str:
        try:
            return importlib.import_module(mod_name).__version__
        except Exception:
            return 'unknown'

    run_id   = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot = {
        'run_id':    run_id,
        'timestamp': datetime.now().isoformat(),
        'python':    sys.version,
        'platform':  platform.platform(),
        'libraries': {
            'numpy':        _ver('numpy'),
            'pandas':       _ver('pandas'),
            'scikit-learn': _ver('sklearn'),
            'xgboost':      _ver('xgboost'),
            'shap':         _ver('shap'),
            'matplotlib':   _ver('matplotlib'),
        },
        'config': {k: v for k, v in vars(config.__class__).items()
                   if not k.startswith('_') and not callable(v)},
    }

    out_path = os.path.join(output_dir, f"ml_run_config_{run_id}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, default=str)

    return run_id, out_path


# ============================================================
# P0-C: CUSTOM TRANSFORMERS  (fit/transform strict separation)
# ============================================================
class VarianceFilter(BaseEstimator, TransformerMixin):
    """
    Remove features whose variance (after scaling) is below threshold.
    P0-F: called AFTER StandardScaler so ratio-features are not
    penalised against HU-scale features.

    fit()      → learn which columns to keep (on train fold only)
    transform()→ apply stored keep_cols_; safe on unseen column sets
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            variances = X.var(ddof=0)
            self.keep_cols_  = variances.index[variances >= self.threshold].tolist()
            self.drop_cols_  = variances.index[variances <  self.threshold].tolist()
            self.feature_names_in_ = list(X.columns)
        else:
            variances = np.var(X, axis=0, ddof=0)
            self.keep_cols_  = np.where(variances >= self.threshold)[0].tolist()
            self.drop_cols_  = np.where(variances <  self.threshold)[0].tolist()
            self.feature_names_in_ = None
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Only keep columns that exist in both train set and this X
            cols = [c for c in self.keep_cols_ if c in X.columns]
            return X[cols]
        else:
            return X[:, self.keep_cols_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove highly correlated features (Pearson |r| > threshold).
    When two features exceed the threshold, the one with lower variance
    in the training fold is dropped.

    fit()      → build drop_cols_ and keep_cols_ from train fold only
    transform()→ drop stored columns; handles missing/extra cols safely
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        corr  = X.corr(method='pearson').abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

        # Pre-compute F-scores once for tie-breaking.
        # Variance-based tie-break is meaningless after StandardScaler because
        # all non-constant features have variance ≈ 1.0 at that point — making
        # the old rule a near-random floating-point comparison, not a signal filter.
        if y is not None:
            f_vals, _ = f_classif(X, y)
            f_score_map = dict(zip(X.columns, f_vals))
        else:
            f_score_map = None

        to_drop = set()
        cols    = list(X.columns)
        n       = len(cols)

        for i in range(n):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, n):
                if cols[j] in to_drop:
                    continue
                if upper.iloc[i, j] > self.threshold:
                    if f_score_map is not None:
                        # Keep the feature with higher F-score (more target-associated)
                        if f_score_map[cols[i]] >= f_score_map[cols[j]]:
                            to_drop.add(cols[j])
                        else:
                            to_drop.add(cols[i])
                    else:
                        to_drop.add(cols[j])   # deterministic fallback: drop latter

        self.drop_cols_ = list(to_drop)
        self.keep_cols_ = [c for c in cols if c not in to_drop]
        self.feature_names_in_ = cols
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        # Only drop columns that actually exist in this X
        cols_to_drop = [c for c in self.drop_cols_ if c in X.columns]
        return X.drop(columns=cols_to_drop)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-N features by XGBoost feature importance.
    Trained on inner-CV train fold only — no leakage.
    """

    def __init__(self, max_features: int = 30, random_state: int = 42):
        self.max_features  = max_features
        self.random_state  = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TopFeatureSelector requires y during fit.")

        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            X_arr = X.values
        else:
            cols = [str(i) for i in range(X.shape[1])]
            X_arr = X

        probe = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=self.random_state, n_jobs=1,
            objective='multi:softprob',
        )
        # Fold-local class weights
        sw = compute_sample_weight('balanced', y)
        probe.fit(X_arr, y, sample_weight=sw)

        imp = probe.feature_importances_
        order = np.argsort(imp)[::-1]
        n_keep = min(self.max_features, len(cols))
        self.selected_cols_ = [cols[i] for i in order[:n_keep]]
        self.importances_   = dict(zip(cols, imp))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.selected_cols_ if c in X.columns]
            return X[cols]
        else:
            # X is ndarray — we stored integer indices implicitly via probe
            # Re-derive from importances_ ordering if columns not available
            all_cols = [str(i) for i in range(X.shape[1])]
            idxs     = [all_cols.index(c) for c in self.selected_cols_
                        if c in all_cols]
            return X[:, idxs]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_cols_)


class ShapTopKSelector(BaseEstimator, TransformerMixin):
    """
    Select top-K features by mean |SHAP| from a probe XGBClassifier.
    Method is model-consistent: probe and final classifier are the same
    family (XGBoost); attribution uses SHAP's exact tree-path algorithm
    (TreeExplainer), not gain-based feature_importances_.

    fit()      → learn top-K from train fold SHAP values only (no leakage)
    transform()→ apply stored selection; safe on unseen column sets
    """

    def __init__(self, max_features: int = 30, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("ShapTopKSelector requires y during fit.")

        if isinstance(X, pd.DataFrame):
            cols  = list(X.columns)
            X_arr = X.values
        else:
            cols  = [str(i) for i in range(X.shape[1])]
            X_arr = X

        probe = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=self.random_state, n_jobs=1,
            objective='multi:softprob',
        )
        sw = compute_sample_weight('balanced', y)
        probe.fit(X_arr, y, sample_weight=sw)

        explainer = shap.TreeExplainer(probe)
        shap_vals = explainer.shap_values(X_arr)

        # Handle XGBoost/SHAP version differences (same logic as compute_shap_oof)
        sv = np.array(shap_vals)
        if sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=0).mean(axis=-1)
        elif sv.ndim == 2:
            mean_abs = np.abs(sv).mean(axis=0)
        else:
            mean_abs = np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0)
        mean_abs = np.asarray(mean_abs).ravel()

        order  = np.argsort(mean_abs)[::-1]
        n_keep = min(self.max_features, len(cols))
        self.selected_cols_    = [cols[i] for i in order[:n_keep]]
        self.shap_importances_ = dict(zip(cols, mean_abs))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.selected_cols_ if c in X.columns]
            return X[cols]
        else:
            all_cols = [str(i) for i in range(X.shape[1])]
            idxs = [all_cols.index(c) for c in self.selected_cols_ if c in all_cols]
            return X[:, idxs]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_cols_)


# ============================================================
# DATA LOADING
# ============================================================
def load_and_prepare(config: Config):
    log.info("Loading feature file: %s", config.FEATURES_FILE)
    df = pd.read_csv(config.FEATURES_FILE)
    log.info("Loaded %d patients, %d columns", len(df), len(df.columns))
    log.info("Class distribution:\n%s", df['cancer_type'].value_counts().to_string())

    exclude = {'patient_id', 'cancer_type'}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()

    # Drop non-numeric columns (e.g. necrosis_qc_method, tlr_ref_source).
    # These are QC/provenance strings — not model inputs.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.info("Dropping %d non-numeric columns: %s", len(non_numeric), non_numeric)
        X = X.drop(columns=non_numeric)

    # Fit LabelEncoder on actual data values so encoding is deterministic
    # and matches the data — not on config.CLASSES list order which may differ.
    # CLASS_LABELS must be updated to match le.classes_ order.
    le = LabelEncoder().fit(sorted(df['cancer_type'].unique()))
    y  = le.transform(df['cancer_type'])

    # Update CLASS_LABELS to match actual encoding
    config.CLASS_LABELS = {i: cls for i, cls in enumerate(le.classes_)}
    config.CLASSES = list(le.classes_)

    log.info("Feature matrix: %s", X.shape)
    log.info("Label encoding: %s", config.CLASS_LABELS)
    log.info("Class counts: %s",
             {cls: int((y == i).sum()) for i, cls in config.CLASS_LABELS.items()})
    return X, y, le, df['patient_id'].values


# ============================================================
# PIPELINE BUILDER
# ============================================================
def build_pipeline(config: Config, arm: str = 'full_pipeline') -> Pipeline:
    """
    arm options:
      'no_selection'       : imputer → scaler → var_filt → model
      'correlation_pruning': imputer → scaler → var_filt → cor_filt → model
      'full_pipeline'      : imputer → scaler → var_filt → cor_filt → selector → model

    set_output(transform='pandas') propagates column names through all
    transformers — required for VarianceFilter/CorrelationFilter column tracking.
    Requires sklearn >= 1.2.
    """
    steps = [
        ('imputer',  SimpleImputer(strategy='median')),
        ('scaler',   StandardScaler()),
        ('var_filt', VarianceFilter(threshold=config.MIN_VARIANCE_THRESHOLD)),
    ]
    if arm in ('correlation_pruning', 'full_pipeline', 'shap_topk'):
        steps.append(('cor_filt', CorrelationFilter(threshold=config.CORRELATION_THRESHOLD)))
    if arm == 'full_pipeline':
        steps.append(('selector', TopFeatureSelector(max_features=config.MAX_FEATURES,
                                                      random_state=config.RANDOM_STATE)))
    elif arm == 'shap_topk':
        steps.append(('selector', ShapTopKSelector(max_features=config.MAX_FEATURES,
                                                    random_state=config.RANDOM_STATE)))
    steps.append(('model', XGBClassifier(**config.XGB_BASE)))

    pipe = Pipeline(steps)
    pipe.set_output(transform='pandas')
    return pipe


def _arm_param_dist(config: Config, arm: str) -> dict:
    """Return hyperparameter search space for the given arm."""
    dist = dict(config.XGB_PARAM_DIST)
    if arm in ('full_pipeline', 'shap_topk'):
        dist['selector__max_features'] = config.MAX_FEATURES_CANDIDATES
    return dist


# ============================================================
# P0-C: NESTED CV
# ============================================================
def run_nested_cv(X: pd.DataFrame, y: np.ndarray,
                  config: Config,
                  arm: str = 'full_pipeline',
                  outer_cv: StratifiedKFold = None) -> dict:
    """
    Outer CV  → unbiased performance estimate
    Inner CV  → hyperparameter search (RandomizedSearchCV)

    arm controls which pipeline variant is built (see build_pipeline).
    outer_cv should be the same object across all arms so fold membership
    is identical — pass it from main() rather than letting each call create
    its own (even with the same seed, sharing the object is the explicit guarantee).

    Class weights computed INSIDE each outer-train fold (no global leakage).
    """
    log.info("=== NESTED CV  arm=%s  (outer=%d, inner=%d) ===",
             arm, config.OUTER_SPLITS, config.INNER_SPLITS)

    if outer_cv is None:
        outer_cv = StratifiedKFold(n_splits=config.OUTER_SPLITS,
                                    shuffle=True,
                                    random_state=config.RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=config.INNER_SPLITS,
                                shuffle=True,
                                random_state=config.RANDOM_STATE)

    oof_true  = np.empty(len(y), dtype=int)
    oof_prob  = np.zeros((len(y), len(config.CLASSES)))

    fold_metrics   = []
    fold_pipelines = []
    fold_features  = []

    X_arr = X

    for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X_arr, y), start=1):

        X_tr, X_te = X_arr.iloc[train_idx], X_arr.iloc[test_idx]
        y_tr, y_te = y[train_idx],          y[test_idx]

        sw_tr = compute_sample_weight('balanced', y_tr)

        base_pipe  = build_pipeline(config, arm=arm)
        param_dist = _arm_param_dist(config, arm)

        n_iter = config.ARM_N_ITER.get(arm, config.N_ITER_SEARCH)

        search = RandomizedSearchCV(
            estimator            = base_pipe,
            param_distributions  = param_dist,
            n_iter               = n_iter,
            cv                   = inner_cv,
            scoring              = 'f1_macro',
            refit                = True,
            random_state         = config.RANDOM_STATE,
            n_jobs               = -1,
            error_score          = 'raise',
        )

        search.fit(X_tr, y_tr, **{'model__sample_weight': sw_tr})

        best_pipe = search.best_estimator_
        fold_pipelines.append(best_pipe)

        y_pred = best_pipe.predict(X_te)
        y_prob = best_pipe.predict_proba(X_te)

        oof_true[test_idx] = y_te
        oof_prob[test_idx] = y_prob

        acc      = accuracy_score(y_te, y_pred)
        f1_mac   = f1_score(y_te, y_pred, average='macro')
        f1_wt    = f1_score(y_te, y_pred, average='weighted')
        try:
            auc_ovr = roc_auc_score(y_te, y_prob,
                                     multi_class='ovr', average='macro')
        except ValueError:
            auc_ovr = np.nan

        fold_metrics.append(dict(
            fold=fold_idx, accuracy=acc,
            f1_macro=f1_mac, f1_weighted=f1_wt, auc_ovr=auc_ovr,
            best_params=str(search.best_params_),
            n_test=len(y_te),
        ))

        # Feature names after the last selection step (arm-dependent)
        try:
            steps = best_pipe.named_steps
            if 'selector' in steps:
                fold_features.append(list(steps['selector'].selected_cols_))
            elif 'cor_filt' in steps:
                fold_features.append(list(steps['cor_filt'].keep_cols_))
            elif 'var_filt' in steps:
                fold_features.append(list(steps['var_filt'].keep_cols_))
            else:
                fold_features.append([])
        except Exception:
            fold_features.append([])

        best_shown = {k: v for k, v in search.best_params_.items()
                      if k in ['model__n_estimators', 'model__max_depth',
                               'model__learning_rate']}
        print(f"[CV] arm={arm} Fold {fold_idx}/{config.OUTER_SPLITS}: "
              f"Acc={acc:.3f} F1={f1_mac:.3f} AUC={auc_ovr:.3f} | {best_shown}",
              flush=True)
        log.info("arm=%s Fold %d: Acc=%.3f  F1=%.3f  AUC=%.3f | best=%s",
                 arm, fold_idx, acc, f1_mac, auc_ovr, best_shown)

    metrics_df = pd.DataFrame(fold_metrics)

    log.info("\n=== NESTED CV SUMMARY  arm=%s ===", arm)
    for col in ['accuracy', 'f1_macro', 'f1_weighted', 'auc_ovr']:
        log.info("  %-14s %.3f ± %.3f",
                 col, metrics_df[col].mean(), metrics_df[col].std())

    metrics_df.to_csv(
        os.path.join(config.OUTPUT_DIR, f"nested_cv_fold_results_{arm}.csv"),
        index=False
    )

    # Export fold-level selected features — consumed by statistics script for
    # feature/family stability analysis (recurrence across outer folds)
    feat_rows = [
        {'fold': fold_i, 'arm': arm, 'feature': fname}
        for fold_i, fnames in enumerate(fold_features, start=1)
        for fname in fnames
    ]
    if feat_rows:
        pd.DataFrame(feat_rows).to_csv(
            os.path.join(config.OUTPUT_DIR, f"fold_selected_features_{arm}.csv"),
            index=False
        )
        log.info("Fold features exported: %d rows (arm=%s)", len(feat_rows), arm)

    # Save OOF predictions + class probabilities — probabilities needed for ROC figures.
    # patient_id added in main() via merge.
    oof_pred_arr = np.argmax(oof_prob, axis=1)
    pd.DataFrame({
        'true_label': [config.CLASS_LABELS[int(i)] for i in oof_true],
        'pred_label': [config.CLASS_LABELS[int(i)] for i in oof_pred_arr],
        'is_error':   (oof_true != oof_pred_arr).astype(int),
        **{f'prob_{config.CLASS_LABELS[i]}': oof_prob[:, i]
           for i in range(len(config.CLASSES))},
    }).to_csv(
        os.path.join(config.OUTPUT_DIR, f"oof_predictions_{arm}.csv"),
        index=False
    )
    log.info("OOF predictions saved (arm=%s)", arm)

    return dict(
        fold_metrics   = metrics_df,
        oof_true       = oof_true,
        oof_prob       = oof_prob,
        fold_pipelines = fold_pipelines,
        fold_features  = fold_features,
    )


# ============================================================
# FINAL REFIT  (for deployment / SHAP secondary analysis)
# ============================================================
def refit_final_model(X: pd.DataFrame, y: np.ndarray,
                       config: Config, arm: str = 'full_pipeline') -> Pipeline:
    """
    Refit on ALL data with default hyperparameters for the given arm.
    Secondary reference only — reported performance comes from nested CV.
    """
    log.info("Refitting final pipeline on full dataset (arm=%s)...", arm)
    sw = compute_sample_weight('balanced', y)
    pipe = build_pipeline(config, arm=arm)
    pipe.fit(X, y, **{'model__sample_weight': sw})
    pipe.named_steps['model'].save_model(
        os.path.join(config.OUTPUT_DIR, f"xgb_final_refit_{arm}.json")
    )
    log.info("Final model saved.")
    return pipe


# ============================================================
# P3-A: SHAP — OUTER-FOLD AGGREGATE
# ============================================================
def compute_shap_oof(X: pd.DataFrame, y: np.ndarray,
                     cv_results: dict,
                     config: Config,
                     arm: str = '') -> pd.DataFrame:
    """
    SHAP computed on each outer fold's test set using that fold's
    best pipeline (not the final refit).  Importance is aggregated
    (mean |SHAP|) across folds, weighted by fold test-set size.

    This is the methodologically correct approach:
    SHAP reflects model behaviour on truly unseen data.
    """
    log.info("=== SHAP — OUTER-FOLD AGGREGATE ===")

    outer_cv = StratifiedKFold(n_splits=config.OUTER_SPLITS,
                                shuffle=True,
                                random_state=config.RANDOM_STATE)

    fold_importances = []

    for fold_idx, ((_, test_idx), pipe, feat_names) in enumerate(
            zip(outer_cv.split(X, y),
                cv_results['fold_pipelines'],
                cv_results['fold_features']), start=1):

        X_te = X.iloc[test_idx]
        if not feat_names:
            log.warning("Fold %d: no feature names stored, skipping SHAP", fold_idx)
            continue

        # Transform test data through all steps except the model
        try:
            X_te_trans = pipe[:-1].transform(X_te)
            if isinstance(X_te_trans, pd.DataFrame):
                X_te_trans = X_te_trans.values
        except Exception as exc:
            log.warning("Fold %d SHAP transform failed: %s", fold_idx, exc)
            continue

        try:
            explainer = shap.TreeExplainer(pipe.named_steps['model'])
            shap_vals = explainer.shap_values(X_te_trans)
        except Exception as exc:
            log.warning("Fold %d SHAP explainer failed: %s", fold_idx, exc)
            continue

        # shap_vals shape varies by XGBoost/SHAP version:
        #   - list of [n_samples, n_features] per class  (older SHAP)
        #   - 3D ndarray [n_samples, n_features, n_classes]  (newer SHAP)
        #   - 2D ndarray [n_samples, n_features]  (binary, shouldn't happen here)
        # In all cases we want a 1D array of shape [n_features].
        try:
            sv = np.array(shap_vals)
            if sv.ndim == 3:
                # [n_samples, n_features, n_classes] → mean over samples & classes
                mean_abs = np.abs(sv).mean(axis=0).mean(axis=-1)   # → [n_features]
            elif sv.ndim == 2:
                mean_abs = np.abs(sv).mean(axis=0)                  # → [n_features]
            else:
                # list of 2D arrays (one per class)
                mean_abs = np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0)
            mean_abs = np.asarray(mean_abs).ravel()
        except Exception as exc:
            log.warning("Fold %d SHAP aggregation failed: %s", fold_idx, exc)
            continue

        n_feats = min(len(feat_names), len(mean_abs))
        fold_importances.append(
            pd.DataFrame({
                'feature':      feat_names[:n_feats],
                'shap_mean_abs': mean_abs[:n_feats],
                'fold':         fold_idx,
                'n_test':       len(test_idx),
            })
        )
        print(f"[SHAP] Fold {fold_idx} done (n_test={len(test_idx)}, n_feat={n_feats})",
              flush=True)
        log.info("Fold %d SHAP done (n_test=%d, n_feat=%d)",
                 fold_idx, len(test_idx), n_feats)

    if not fold_importances:
        log.warning("No SHAP results collected.")
        return pd.DataFrame()

    all_imp = pd.concat(fold_importances, ignore_index=True)

    # Weighted mean across folds
    agg = (
        all_imp
        .groupby('feature')
        .apply(lambda g: np.average(g['shap_mean_abs'], weights=g['n_test']))
        .reset_index()
        .rename(columns={0: 'shap_weighted_mean_abs'})
        .sort_values('shap_weighted_mean_abs', ascending=False)
        .reset_index(drop=True)
    )

    suffix = f"_{arm}" if arm else ""
    agg.to_csv(
        os.path.join(config.OUTPUT_DIR, f"shap_oof_importance{suffix}.csv"),
        index=False
    )
    log.info("SHAP OOF importance saved (%d unique features)", len(agg))
    return agg


# ============================================================
# VISUALISATIONS
# ============================================================
def plot_oof_confusion_matrix(oof_true, oof_pred, config: Config, arm: str = ''):
    cm = confusion_matrix(oof_true, oof_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix — OOF (Nested CV){" [" + arm + "]" if arm else ""}')
    plt.tight_layout()
    suffix = f"_{arm}" if arm else ""
    out = os.path.join(config.OUTPUT_DIR, f"oof_confusion_matrix{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    log.info("Confusion matrix saved: %s", out)


def plot_roc_curves(oof_true, oof_prob, config: Config, arm: str = ''):
    plt.figure(figsize=(8, 7))
    colors = ['#e74c3c', '#3498db', '#9b59b6']
    for i, (cls, color) in enumerate(zip(config.CLASSES, colors)):
        yb    = (oof_true == i).astype(int)
        ys    = oof_prob[:, i]
        fpr, tpr, _ = roc_curve(yb, ys)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{cls} (AUC={roc_auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves — OOF (Nested CV){" [" + arm + "]" if arm else ""}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    suffix = f"_{arm}" if arm else ""
    out = os.path.join(config.OUTPUT_DIR, f"oof_roc_curves{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    log.info("ROC curves saved: %s", out)


def plot_cv_fold_metrics(metrics_df: pd.DataFrame, config: Config, arm: str = ''):
    cols   = ['accuracy', 'f1_macro', 'f1_weighted', 'auc_ovr']
    titles = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'AUC (OvR)']
    colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, col, title, color in zip(axes, cols, titles, colors):
        vals = metrics_df[col].values
        ax.bar(range(1, len(vals) + 1), vals, color=color, alpha=0.75)
        ax.axhline(vals.mean(), color='red', linestyle='--', lw=2,
                   label=f'Mean: {vals.mean():.3f}±{vals.std():.3f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, len(vals) + 1))
    plt.tight_layout()
    suffix = f"_{arm}" if arm else ""
    out = os.path.join(config.OUTPUT_DIR, f"nested_cv_fold_metrics{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    log.info("CV fold metrics plot saved: %s", out)


def plot_shap_importance(shap_agg: pd.DataFrame, config: Config,
                          top_n: int = 20, arm: str = ''):
    if shap_agg.empty:
        return
    top = shap_agg.head(top_n)
    plt.figure(figsize=(9, 7))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top)))
    plt.barh(range(len(top)),
             top['shap_weighted_mean_abs'].values[::-1],
             color=colors[::-1])
    plt.yticks(range(len(top)),
               top['feature'].values[::-1], fontsize=8)
    plt.xlabel('Weighted Mean |SHAP Value| (OOF)')
    plt.title(f'Top {top_n} Features — SHAP OOF Aggregate{" [" + arm + "]" if arm else ""}')
    plt.tight_layout()
    suffix = f"_{arm}" if arm else ""
    out = os.path.join(config.OUTPUT_DIR, f"shap_oof_importance_bar{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    log.info("SHAP importance bar chart saved: %s", out)


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Nested CV XGBoost classification pipeline for primary liver tumors'
    )
    ap.add_argument('--data-dir', default='./data',
                    help='Root data directory (default: ./data)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory (default: <data-dir>/ml_classification_v4)')
    ns = ap.parse_args()

    config = Config()
    config.BASE_DIR     = ns.data_dir
    config.SPATIAL_DIR  = f"{ns.data_dir}/spatial_analysis_v4"
    config.OUTPUT_DIR   = ns.output_dir or f"{ns.data_dir}/ml_classification_v4"
    config.FEATURES_FILE = f"{config.SPATIAL_DIR}/combined_spatial_v4_full.csv"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(config.OUTPUT_DIR, f"ml_run_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                  logging.StreamHandler()],
        force=True,
    )
    log.setLevel(logging.INFO)

    # P3-B: reproducibility — seed + config snapshot
    np.random.seed(config.NUMPY_SEED)
    run_id, cfg_path = save_run_config(config, config.OUTPUT_DIR)

    log.info("=" * 70)
    log.info("ML CLASSIFICATION — PRIMARY LIVER TUMORS (v4.0)")
    log.info("Run ID: %s | Config: %s", run_id, cfg_path)
    log.info("=" * 70)

    X, y, label_encoder, patient_ids = load_and_prepare(config)

    # Single outer CV shared across all arms — identical fold splits guaranteed
    outer_cv = StratifiedKFold(n_splits=config.OUTER_SPLITS,
                                shuffle=True,
                                random_state=config.RANDOM_STATE)

    all_summaries = []

    for arm in config.ARMS:
        log.info("\n" + "=" * 60)
        log.info("ARM: %s", arm)
        log.info("=" * 60)

        cv_results = run_nested_cv(X, y, config, arm=arm, outer_cv=outer_cv)

        oof_true = cv_results['oof_true']
        oof_pred = np.argmax(cv_results['oof_prob'], axis=1)
        oof_prob = cv_results['oof_prob']

        # Prepend patient_id to OOF predictions file (row order preserved by outer_cv)
        _oof_path = os.path.join(config.OUTPUT_DIR, f"oof_predictions_{arm}.csv")
        if os.path.exists(_oof_path):
            _oof_df = pd.read_csv(_oof_path)
            _oof_df.insert(0, 'patient_id', patient_ids)
            _oof_df.to_csv(_oof_path, index=False)

        log.info("\n=== OOF PERFORMANCE — %s ===", arm)
        log.info("\n%s", classification_report(
            oof_true, oof_pred, target_names=config.CLASSES
        ))
        try:
            oof_auc = roc_auc_score(oof_true, oof_prob,
                                     multi_class='ovr', average='macro')
            log.info("OOF AUC (macro OvR): %.3f", oof_auc)
        except ValueError as exc:
            log.warning("OOF AUC calculation failed: %s", exc)
            oof_auc = np.nan

        shap_agg = compute_shap_oof(X, y, cv_results, config, arm=arm)

        plot_oof_confusion_matrix(oof_true, oof_pred, config, arm=arm)
        plot_roc_curves(oof_true, oof_prob, config, arm=arm)
        plot_cv_fold_metrics(cv_results['fold_metrics'], config, arm=arm)
        plot_shap_importance(shap_agg, config, arm=arm)

        mdf = cv_results['fold_metrics']

        # Feature count per fold — key ablation metric: "AUC held while N_features dropped"
        feat_counts = [len(ff) for ff in cv_results['fold_features'] if ff]
        mean_n_feat = float(np.mean(feat_counts)) if feat_counts else np.nan
        std_n_feat  = float(np.std(feat_counts))  if feat_counts else np.nan

        all_summaries.append(dict(
            arm                  = arm,
            n_patients           = len(y),
            n_features_raw       = X.shape[1],
            mean_n_features      = mean_n_feat,
            std_n_features       = std_n_feat,
            outer_splits         = config.OUTER_SPLITS,
            inner_splits         = config.INNER_SPLITS,
            n_iter_used          = config.ARM_N_ITER.get(arm, config.N_ITER_SEARCH),
            cv_accuracy_mean     = mdf['accuracy'].mean(),
            cv_accuracy_std      = mdf['accuracy'].std(),
            cv_f1_macro_mean     = mdf['f1_macro'].mean(),
            cv_f1_macro_std      = mdf['f1_macro'].std(),
            cv_auc_ovr_mean      = mdf['auc_ovr'].mean(),
            cv_auc_ovr_std       = mdf['auc_ovr'].std(),
            oof_auc              = oof_auc,
            shap_method          = 'outer_fold_oof_aggregate',
            run_id               = run_id,
        ))

    # Arm comparison table — direct input to ablation section of paper
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(
        os.path.join(config.OUTPUT_DIR, "model_comparison_v4.csv"),
        index=False
    )
    log.info("\n=== ARM COMPARISON ===")
    log.info("\n%s", summary_df[['arm', 'cv_accuracy_mean', 'cv_f1_macro_mean',
                                  'cv_auc_ovr_mean', 'oof_auc',
                                  'mean_n_features', 'std_n_features',
                                  'n_iter_used']].to_string(index=False))

    # Final refit with best arm by CV AUC (secondary reference only)
    best_arm = summary_df.loc[summary_df['cv_auc_ovr_mean'].idxmax(), 'arm']
    log.info("Best arm by CV AUC: %s — used for final refit", best_arm)
    refit_final_model(X, y, config, arm=best_arm)

    log.info("\n=== COMPLETE ===")
    log.info("Outputs: %s", config.OUTPUT_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    main()

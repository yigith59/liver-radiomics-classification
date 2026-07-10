# 04_sensitivity_analysis.py
# R2.2 Sensitivity Analysis — Liver-Reference Intensity Normalization
# Compares classification performance before and after removing absolute HU offsets.
# Principled rule: subtract phase-specific background-liver median from location statistics
# only (mean, median, percentiles). Spread statistics (std, iqr, mad, skew, kurt)
# are invariant to additive offset and are left unchanged. Ratio/diff features are
# already liver-relative and are left unchanged.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import f_oneway
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────
# Set CSV_PATH to the combined feature matrix produced by 01_feature_extraction.py
CSV_PATH = './data/spatial_analysis_v4/combined_spatial_v4_full.csv'
# ──────────────────────────────────────────────────────────────────────────────

PHASES       = ['P', 'C1', 'C2', 'C3']
OUTER_SPLITS = 5
SHAP_TOP_K   = 30
VAR_THRESH   = 0.01
CORR_THRESH  = 0.90
RANDOM_STATE = 42

# ─── Normalization configuration ───────────────────────────────────────────────
# Phase-specific background liver reference column (normalization factor)
LIVER_REF = {ph: f'{ph}_tlr_liver_ref_median' for ph in PHASES}

# Suffixes that flag a column as an absolute HU location statistic
LOCATION_SUFFIXES = ('_median', '_mean', '_p10', '_p25', '_p75', '_p90', '_mean_hu')

# If any of these keywords appear in a column name → do NOT normalize.
# Rationale per keyword:
#   ratio / diff         → already liver-relative
#   _std / _iqr / _mad   → spread statistics: E[Var(X+c)] = Var(X), unchanged by offset
#   _skew / _kurt        → standardized moments, offset-invariant
#   _entropy / _cv_      → information-theoretic / coefficient of variation
#   gradient             → spatial derivative removes constant offset
#   local_entropy        → texture entropy, not absolute HU
#   _fraction / _count / _flag / _hybrid → non-HU
#   tlr_liver_ref        → the normalization reference itself (removed from set)
#   _range_              → max−min spread, offset-invariant
#   of_sectors           → angular_mean_of_sectors; covered indirectly by max/min_sector_mean
EXCLUDE_KEYWORDS = [
    'ratio', 'diff',
    '_std', '_iqr', '_mad', '_range_',
    '_skew', '_kurt',
    '_entropy', '_cv_',
    'gradient', 'local_entropy',
    '_fraction', '_count', '_flag', '_hybrid',
    'tlr_liver_ref',
    'of_sectors',
]


def classify_features(feat_cols):
    """
    Returns:
      to_normalize : columns to subtract phase liver_ref_median from
      to_remove    : liver_ref location columns (removed from normalized set entirely,
                     because after subtraction they become identically 0)
    """
    to_normalize, to_remove = [], []
    for col in feat_cols:
        phase   = next((ph for ph in PHASES if col.startswith(ph + '_')), None)
        if phase is None:
            continue
        is_loc  = any(col.endswith(s) for s in LOCATION_SUFFIXES)
        is_ref  = 'tlr_liver_ref' in col
        is_excl = any(kw in col for kw in EXCLUDE_KEYWORDS)

        if is_ref and is_loc:
            to_remove.append(col)      # e.g. C1_tlr_liver_ref_median → drop entirely
        elif is_loc and not is_excl:
            to_normalize.append(col)   # absolute HU location stat → normalize

    return to_normalize, to_remove


def normalize_features(df_feat, to_normalize, to_remove):
    """
    Per-patient, phase-specific subtraction:
        normalized_col = original_col - liver_ref_median (same phase)
    Liver_ref location columns are then dropped from the result because
    they become 0 for all patients (VarianceFilter would eliminate them anyway,
    but explicit removal is cleaner and makes the intent transparent).
    """
    df_n = df_feat.copy()
    for col in to_normalize:
        ph  = next(ph for ph in PHASES if col.startswith(ph + '_'))
        ref = LIVER_REF[ph]
        if ref in df_n.columns:
            df_n[col] = df_n[col] - df_n[ref]
        else:
            print(f'  [WARN] Reference not found for {col}: {ref}')
    df_n = df_n.drop(columns=[c for c in to_remove if c in df_n.columns])
    return df_n


# ─── Pipeline components ───────────────────────────────────────────────────────

class VarFilter:
    def fit(self, X):
        self.mask_ = np.var(X, axis=0) > VAR_THRESH
        return self
    def transform(self, X):        return X[:, self.mask_]
    def fit_transform(self, X):    return self.fit(X).transform(X)


class CorrFilter:
    def fit(self, X, y):
        n  = X.shape[1]
        cc = np.corrcoef(X.T)
        classes = np.unique(y)
        fs = np.zeros(n)
        for j in range(n):
            try:
                fs[j] = f_oneway(*[X[y == c, j] for c in classes])[0]
            except Exception:
                pass
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if keep[j] and abs(cc[i, j]) > CORR_THRESH:
                    if fs[i] >= fs[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        self.idx_ = np.where(keep)[0]
        return self
    def transform(self, X):           return X[:, self.idx_]
    def fit_transform(self, X, y):    return self.fit(X, y).transform(X)


class SHAPSelector:
    def fit(self, X, y):
        m = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE, verbosity=0
        )
        m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
        sv = shap.TreeExplainer(m).shap_values(X)
        if isinstance(sv, list):
            # Older SHAP: list of 2D arrays (one per class)
            imp = np.mean([np.abs(v).mean(0) for v in sv], 0)
        else:
            sv = np.array(sv)
            if sv.ndim == 3:
                # Newer SHAP: (n_samples, n_features, n_classes) — average over samples and classes
                imp = np.abs(sv).mean(axis=(0, 2))
            else:
                imp = np.abs(sv).mean(0)
        imp = np.asarray(imp).flatten()   # guarantee 1D regardless of SHAP version
        self.idx_ = np.argsort(imp)[::-1][:min(SHAP_TOP_K, X.shape[1])]
        return self
    def transform(self, X):           return X[:, self.idx_]
    def fit_transform(self, X, y):    return self.fit(X, y).transform(X)


def run_fold(Xtr, ytr, Xte):
    """Single outer-fold: impute → scale → var_filter → corr_filter → shap_topK → XGB"""
    imp = SimpleImputer(strategy='median')
    Xtr, Xte = imp.fit_transform(Xtr), imp.transform(Xte)

    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

    vf = VarFilter()
    Xtr, Xte = vf.fit_transform(Xtr), vf.transform(Xte)

    cf = CorrFilter()
    Xtr, Xte = cf.fit_transform(Xtr, ytr), cf.transform(Xte)

    sel = SHAPSelector()
    Xtr, Xte = sel.fit_transform(Xtr, ytr), sel.transform(Xte)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, verbosity=0
    )
    model.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    return model.predict_proba(Xte), model.predict(Xte)


def nested_cv(X, y, tag):
    cv    = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n_cls = len(np.unique(y))
    prob  = np.zeros((len(y), n_cls))
    pred  = np.zeros(len(y), dtype=int)

    for k, (tr, te) in enumerate(cv.split(X, y)):
        print(f'  [{tag}] fold {k+1}/{OUTER_SPLITS} …', end=' ', flush=True)
        p, q           = run_fold(X[tr], y[tr], X[te])
        prob[te], pred[te] = p, q
        fold_auc = roc_auc_score(y[te], p, multi_class='ovr', average='macro')
        print(f'AUC={fold_auc:.3f}')

    auc = roc_auc_score(y, prob, multi_class='ovr', average='macro')
    f1  = f1_score(y, pred, average='macro')
    acc = accuracy_score(y, pred)
    return auc, f1, acc


# ─── Load data ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='HU normalization sensitivity analysis (liver-reference subtraction)'
    )
    ap.add_argument('--csv', default=CSV_PATH,
                    help='Path to combined feature CSV (default: CSV_PATH at top of file)')
    ns = ap.parse_args()

    print('Loading CSV …')
    df = pd.read_csv(ns.csv)
    print(f'  Shape: {df.shape}')

    # Auto-detect label column
    for cand in ['cancer_type', 'label', 'Label', 'class', 'Class', 'diagnosis', 'tumor_type']:
        if cand in df.columns:
            LABEL_COL = cand
            break
    else:
        print('First 30 columns:', df.columns[:30].tolist())
        raise ValueError('Label column not found — set LABEL_COL manually above')

    print(f"  Label column: '{LABEL_COL}'")
    print(df[LABEL_COL].value_counts().to_string())

    meta      = [LABEL_COL] + [c for c in ['patient_id', 'pid', 'ID'] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta]
    X_df      = df[feat_cols].select_dtypes(include=[np.number])
    feat_cols = list(X_df.columns)
    y         = pd.Categorical(df[LABEL_COL]).codes.astype(int)
    print(f'  Numeric features: {len(feat_cols)}, N={len(y)}')

# ─── Feature classification ────────────────────────────────────────────────────
    to_normalize, to_remove = classify_features(feat_cols)
    print(f'\nFeature classification:')
    print(f'  Location HU features to normalize   : {len(to_normalize)}')
    print(f'  Liver-ref location cols to remove    : {len(to_remove)}')
    print(f'  Normalized feature matrix columns    : {len(feat_cols) - len(to_remove)}')
    print(f'\n  Removed columns : {to_remove}')
    print(f'\n  First 15 normalized columns:')
    for c in to_normalize[:15]:
        print(f'    {c}')

    print('\n' + '='*60)
    print('ORIGINAL MODEL  (absolute HU features)')
    print('='*60)
    X_orig = X_df.values.astype(np.float64)
    auc_o, f1_o, acc_o = nested_cv(X_orig, y, 'ORIG')
    print(f'\n  >> ORIGINAL  AUC={auc_o:.4f}  F1={f1_o:.4f}  Acc={acc_o:.4f}')

    print('\n' + '='*60)
    print('NORMALIZED MODEL  (liver-reference subtraction)')
    print('='*60)
    df_norm = normalize_features(X_df, to_normalize, to_remove)
    X_norm  = df_norm.select_dtypes(include=[np.number]).values.astype(np.float64)
    print(f'  Normalized matrix: {X_norm.shape}')
    auc_n, f1_n, acc_n = nested_cv(X_norm, y, 'NORM')
    print(f'\n  >> NORMALIZED  AUC={auc_n:.4f}  F1={f1_n:.4f}  Acc={acc_n:.4f}')

    print('\n' + '='*60)
    print('SENSITIVITY ANALYSIS — RESULTS')
    print('='*60)
    print(f'{"Model":<32} {"AUC":>7} {"F1_macro":>10} {"Acc":>8}')
    print('-'*60)
    print(f'{"Original (absolute HU)":<32} {auc_o:>7.4f} {f1_o:>10.4f} {acc_o:>8.4f}')
    print(f'{"Normalized (liver-ref)":<32} {auc_n:>7.4f} {f1_n:>10.4f} {acc_n:>8.4f}')
    print(f'{"Delta (Norm - Orig)":<32} {auc_n-auc_o:>7.4f} {f1_n-f1_o:>10.4f} {acc_n-acc_o:>8.4f}')
    print('='*60)

    d = auc_o - auc_n
    if d < 0.02:
        msg = 'ΔAUC < 0.02 → relative enhancement patterns carry the signal; absolute HU offset is not the primary driver'
    elif d < 0.06:
        msg = 'ΔAUC 0.02–0.06 → mixed signal: biological parenchymal differences and acquisition-related HU both contribute'
    else:
        msg = 'ΔAUC > 0.06 → absolute HU contributes substantially; strengthen limitation statement'
    print(f'\nINTERPRETATION: {msg}')
    print('='*60)


if __name__ == '__main__':
    main()

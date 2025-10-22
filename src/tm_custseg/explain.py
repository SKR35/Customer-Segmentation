import sqlite3, csv, warnings
from typing import Tuple, List
import numpy as np

from .segment import FEATURE_TABLE, SEGMENTS_TABLE, _zscore
from .preproc import preprocess

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import balanced_accuracy_score
except Exception:
    RandomForestClassifier = None

def _load_Xy(conn: sqlite3.Connection, method: str, k: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
    q = f"""
    SELECT s.segment, {', '.join('f.'+c for c in cols)}
    FROM {SEGMENTS_TABLE} s
    JOIN {FEATURE_TABLE} f ON f.customer_id = s.customer_id
    WHERE s.method=? AND s.k=? AND s.segment >= 0        -- only active-labeled rows
    ORDER BY s.customer_id
    """
    y, X = [], []
    for row in conn.execute(q, (method, k)):
        y.append(int(row[0]))
        X.append([float(v) for v in row[1:]])
    return np.array(X, float), np.array(y, int), cols

def _fit_rf(seed: int, n_estimators: int):
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced"
    )

def _score(clf, X, y):
    # balanced accuracy is robust to imbalance
    return balanced_accuracy_score(y, clf.predict(X))

def _auto_groups(Xz: np.ndarray, thr: float) -> List[List[int]]:
    """Greedy grouping by absolute correlation >= thr (on z-scored features)."""
    corr = np.corrcoef(Xz, rowvar=False)
    n = corr.shape[0]
    used = set()
    groups = []
    for i in range(n):
        if i in used: continue
        grp = [i]
        for j in range(i+1, n):
            if j in used: continue
            if abs(corr[i, j]) >= thr:
                grp.append(j); used.add(j)
        used.update(grp)
        groups.append(grp)
    return groups

def _pi_grouped_cv(Xz, y, n_repeats, seed, n_estimators, cv, corr_thr):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    p = Xz.shape[1]
    drops_per_group = []

    for fold, (tri, tei) in enumerate(skf.split(Xz, y), 1):
        clf = _fit_rf(seed + fold, n_estimators).fit(Xz[tri], y[tri])
        base = _score(clf, Xz[tei], y[tei])

        groups = _auto_groups(Xz[tri], corr_thr)  # derive groups on train
        fold_drops = np.zeros((len(groups), n_repeats))
        for g_idx, cols in enumerate(groups):
            for r in range(n_repeats):
                Xperm = Xz[tei].copy()
                # shuffle all columns in the group with the same permutation
                perm = rng.permutation(Xperm.shape[0])
                for c in cols:
                    Xperm[:, c] = Xperm[perm, c]
                fold_drops[g_idx, r] = base - _score(clf, Xperm, y[tei])
        drops_per_group.append(fold_drops.mean(axis=1))  # mean over repeats per group

    drops = np.vstack(drops_per_group)  # (cv, n_groups)
    return drops.mean(axis=0), drops.std(axis=0, ddof=1), groups  # mean/std over folds

def _dropcol_cv(Xz, y, seed, n_estimators, cv):
    """Drop-column importance with CV: baseline score minus score when a feature is removed."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    p = Xz.shape[1]
    drops = []

    for fold, (tri, tei) in enumerate(skf.split(Xz, y), 1):
        # baseline
        clf0 = _fit_rf(seed + fold, n_estimators).fit(Xz[tri], y[tri])
        base = _score(clf0, Xz[tei], y[tei])

        fold_drops = np.zeros(p)
        for j in range(p):
            keep = np.array([c for c in range(p) if c != j])
            clf = _fit_rf(seed + fold + 1000, n_estimators).fit(Xz[tri][:, keep], y[tri])
            s = _score(clf, Xz[tei][:, keep], y[tei])
            fold_drops[j] = base - s
        drops.append(fold_drops)

    drops = np.vstack(drops)  # (cv, p)
    return drops.mean(axis=0), drops.std(axis=0, ddof=1)

def explain_importance(db_path: str, k: int, method: str, out_csv: str,
                       seed: int = 42, test_size: float = 0.25,
                       n_estimators: int = 300, n_repeats: int = 20,
                       scoring: str = "balanced_accuracy",
                       force_permutation: bool = False,
                       cv: int = 1, mode: str = "pi_cv", corr_threshold: float = 0.85):
    """
    mode:
      - 'pi'      : single split permutation_importance (with safe fallback)
      - 'pi_cv'   : CV permutation_importance aggregated across folds  (recommended)
      - 'grouped' : CV grouped-permutation (handles collinearity; groups auto by |corr|>=thr)
      - 'dropcol' : CV drop-column retraining importance (slowest, most robust)
    """
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    conn = sqlite3.connect(db_path)
    try:
        X, y, feat_cols = _load_Xy(conn, method, k)
        # ---- NEW: drop near-constant columns before z-scoring / modeling ----
        MIN_STD = 1e-9
        std = X.std(axis=0)
        keep_cols = std > MIN_STD
        X, feat_cols = preprocess(X, feat_cols)
        if not np.all(keep_cols):
            dropped = [f for f, k in zip(feat_cols, keep_cols) if not k]
            print("dropping near-constant features:", ", ".join(dropped))
            X = X[:, keep_cols]
            feat_cols = [f for f, k in zip(feat_cols, keep_cols) if k]
        # ---------------------------------------------------------------------        
        if X.size == 0:
            raise RuntimeError("No rows found. Run 'extract-features' and 'segment' first.")
        Xz, _, _ = _zscore(X)

        source = mode
        used_train_set = False

        if mode == "dropcol":
            mean, std = _dropcol_cv(Xz, y, seed, n_estimators, cv=max(2, cv))
            import_mean, import_std = mean, std

        elif mode == "grouped":
            mean_g, std_g, groups = _pi_grouped_cv(Xz, y, n_repeats, seed, n_estimators, cv=max(2, cv), corr_thr=corr_threshold)
            # explode group scores back to features (same score for all members)
            import_mean = np.zeros(Xz.shape[1]); import_std = np.zeros(Xz.shape[1])
            for g_idx, cols in enumerate(groups):
                import_mean[cols] = mean_g[g_idx]
                import_std[cols]  = std_g[g_idx]

        elif mode == "pi_cv":
            # standard CV permutation
            skf = StratifiedKFold(n_splits=max(2, cv), shuffle=True, random_state=seed)
            imps = []
            for fold, (tri, tei) in enumerate(skf.split(Xz, y), 1):
                clf = _fit_rf(seed + fold, n_estimators).fit(Xz[tri], y[tri])
                pi = permutation_importance(clf, Xz[tei], y[tei],
                                            n_repeats=n_repeats, random_state=seed + fold,
                                            n_jobs=-1, scoring="balanced_accuracy")
                imps.append(pi.importances_mean)
            arr = np.vstack(imps)
            import_mean = arr.mean(axis=0)
            import_std  = arr.std(axis=0, ddof=1)

        else:  # 'pi' single split with fallback
            classes, counts = np.unique(y, return_counts=True)
            if counts.min() <= 1:
                used_train_set = True
                Xtr, ytr, Xte, yte = Xz, y, Xz, y
                warnings.warn("Using training set for permutation importance (tiny/imbalanced segments).")
            else:
                Xtr, Xte, ytr, yte = train_test_split(Xz, y, test_size=test_size, random_state=seed, stratify=y)

            clf = _fit_rf(seed, n_estimators).fit(Xtr, ytr)
            try:
                pi = permutation_importance(clf, Xte, yte, n_repeats=n_repeats,
                                            random_state=seed, n_jobs=-1, scoring="balanced_accuracy")
                import_mean = pi.importances_mean
                import_std  = pi.importances_std
            except Exception as e:
                warnings.warn(f"permutation_importance failed: {e}")
                source = "rf_importance"
                import_mean = clf.feature_importances_
                import_std  = np.zeros_like(import_mean)

            if (not np.any(np.isfinite(import_mean)) or np.allclose(import_mean, 0.0)) and not force_permutation:
                source = "rf_importance"
                import_mean = clf.feature_importances_
                import_std  = np.zeros_like(import_mean)

        order = np.argsort(-import_mean)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "importance_mean", "importance_std", "rank"])
            for rank, idx in enumerate(order, start=1):
                w.writerow([feat_cols[idx], float(import_mean[idx]), float(import_std[idx]), rank])

        top5 = [(feat_cols[i], float(import_mean[i])) for i in order[:5]]
        return {"k": k, "method": method, "path": out_csv, "top5": top5, "source": source, "used_train_set": used_train_set}
    finally:
        conn.close()
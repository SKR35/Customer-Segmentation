import sqlite3, math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .segment import FEATURE_TABLE, SEGMENTS_TABLE, _zscore
from .preproc import preprocess

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
except Exception:
    RandomForestClassifier = None


def _fit_rf(seed: int, n_estimators: int):
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced"
    )


def _score(clf, X, y):
    return balanced_accuracy_score(y, clf.predict(X))


def _load_Xy(db_path: str, method: str, k: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features and active-only labels (segment >= 0), then drop near-constant columns."""
    conn = sqlite3.connect(db_path)
    try:
        feat_cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
        # features for all customers
        rows = conn.execute(
            f"SELECT customer_id, {', '.join(feat_cols)} FROM {FEATURE_TABLE} ORDER BY customer_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        X   = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float)

        # labels for active customers only (segment >= 0)
        lab = dict(conn.execute(
            f"SELECT customer_id, segment FROM {SEGMENTS_TABLE} WHERE method=? AND k=? AND segment >= 0",
            (method, k)
        ).fetchall())

        y = np.array([lab.get(cid, -1) for cid in ids], dtype=int)

        # keep active only
        keep_rows = y >= 0
        X, feat_cols = preprocess(X, feat_cols)
        X = X[keep_rows]; y = y[keep_rows]

        # ---- drop near-constant columns (avoid zero-variance artifacts) ----
        MIN_STD = 1e-9  # tweak to 1e-6 for stricter filtering
        std = X.std(axis=0)
        keep_cols = std > MIN_STD
        if not np.all(keep_cols):
            dropped = [f for f, k in zip(feat_cols, keep_cols) if not k]
            print("dropping near-constant features:", ", ".join(dropped))
            X = X[:, keep_cols]
            feat_cols = [f for f, k in zip(feat_cols, keep_cols) if k]
        # ------------------------------------------------------------------------

        return X, y, feat_cols
    finally:
        conn.close()

def _auto_groups(Xz: np.ndarray, thr: float) -> List[List[int]]:
    """Group features that are |correlation| >= thr (greedy)."""
    corr = np.corrcoef(Xz, rowvar=False)
    n = corr.shape[0]
    used = set()
    groups = []
    for i in range(n):
        if i in used: 
            continue
        grp = [i]
        for j in range(i + 1, n):
            if j in used:
                continue
            if abs(corr[i, j]) >= thr:
                grp.append(j)
                used.add(j)
        used.update(grp)
        groups.append(grp)
    return groups


def grouped_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    cv: int = 5,
    n_estimators: int = 400,
    n_repeats: int = 30,
    corr_threshold: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Correlation-aware permutation importance:
    - z-score features
    - build groups on the TRAIN fold via |corr| >= threshold
    - for each group, compute drop in balanced accuracy when all columns
      in the group are shuffled together
    - aggregate mean/std across CV folds
    Returns (mean_importance_per_feature, std_importance_per_feature, groups)
    """
    Xz, _, _ = _zscore(X)

    skf = StratifiedKFold(n_splits=max(2, cv), shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    per_fold_feat_scores: List[np.ndarray] = []

    for fold_idx, (tri, tei) in enumerate(skf.split(Xz, y), 1):
        Xtr, ytr = Xz[tri], y[tri]
        Xte, yte = Xz[tei], y[tei]

        # groups learned on TRAIN
        groups = _auto_groups(Xtr, corr_threshold)

        clf = _fit_rf(seed + fold_idx, n_estimators).fit(Xtr, ytr)
        base = _score(clf, Xte, yte)

        # group-level drops (mean over repeats)
        g_drops = np.zeros(len(groups), dtype=float)
        for g_idx, cols in enumerate(groups):
            reps = []
            for r in range(n_repeats):
                Xperm = Xte.copy()
                perm = rng.permutation(Xperm.shape[0])
                for c in cols:
                    Xperm[:, c] = Xperm[perm, c]
                reps.append(base - _score(clf, Xperm, yte))
            g_drops[g_idx] = float(np.mean(reps))

        # explode group scores back to features (same score for all group members)
        f_scores = np.zeros(Xz.shape[1], dtype=float)
        for g_idx, cols in enumerate(groups):
            for c in cols:
                f_scores[c] = g_drops[g_idx]

        per_fold_feat_scores.append(f_scores)

    arr = np.vstack(per_fold_feat_scores)  # (cv, p)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1), groups


def plot_importance_bar(
    names: List[str], means: np.ndarray, stds: np.ndarray, out_png: str,
    title: str, top_n: int | None = None
):
    order = np.argsort(-means)
    if top_n:
        order = order[:top_n]

    names_s = [names[i] for i in order]
    means_s = means[order]
    stds_s  = stds[order]

    fig = plt.figure(figsize=(9, 5))
    y = np.arange(len(names_s))
    plt.barh(y, means_s, xerr=stds_s, align="center")
    plt.yticks(y, names_s)
    plt.gca().invert_yaxis()
    plt.xlabel("importance (mean drop in balanced accuracy)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def viz_grouped_importance(
    db_path: str, k: int, method: str, out_png: str,
    seed: int = 42, cv: int = 5, n_estimators: int = 400, n_repeats: int = 30,
    corr_threshold: float = 0.85, top_n: int | None = None
):
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    X, y, feat_cols = _load_Xy(db_path, method, k)
    if X.size == 0 or len(np.unique(y)) < 2:
        raise RuntimeError("Need active customers and at least 2 segments. Run 'segment' first.")

    means, stds, groups = grouped_permutation_importance(
        X, y, seed=seed, cv=cv, n_estimators=n_estimators, n_repeats=n_repeats, corr_threshold=corr_threshold
    )

    title = f"Grouped permutation importance (k={k}, cv={cv}, corr≥{corr_threshold})"
    plot_importance_bar(feat_cols, means, stds, out_png, title, top_n=top_n)

    return {
        "features": feat_cols,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "groups": groups,
        "path": out_png,
    }
import sqlite3, csv
from typing import List, Tuple
import numpy as np

from .segment import FEATURE_TABLE, _zscore
from .preproc import preprocess

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
except Exception:
    KMeans = None


def _load_X(db_path: str, method: str, k_active: int) -> Tuple[np.ndarray, List[str]]:
    """Load ACTIVE customers (segment>=0) for an existing (method,k_active),
       then apply the same preprocessing used elsewhere.
    """
    conn = sqlite3.connect(db_path)
    try:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
        sql = f"""
        SELECT {', '.join('f.'+c for c in cols)}
        FROM {FEATURE_TABLE} f
        JOIN customer_segments s ON s.customer_id = f.customer_id
        WHERE s.method=? AND s.k=? AND s.segment >= 0
        ORDER BY f.customer_id
        """
        rows = conn.execute(sql, (method, k_active)).fetchall()
        X = np.array([[float(v) for v in r] for r in rows], dtype=float)
        Xp, _ = preprocess(X, cols)
        return Xp, cols
    finally:
        conn.close()

def sweep_k(db_path: str,
            method: str = "kmeans",
            k_active: int = 4,
            kmin: int = 2,
            kmax: int = 12,
            seed: int = 42,
            out_csv: str | None = None):
    """Sweep k on the ACTIVE, preprocessed feature set defined by (method,k_active)."""
    if KMeans is None:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    X, cols = _load_X(db_path, method, k_active)
    if X.size == 0:
        raise RuntimeError("No active rows. Run 'segment' first (with --min-tx 1).")

    Xz, _, _ = _zscore(X)

    rows: list[list[float]] = []
    for k in range(kmin, kmax + 1):
        if k < 2 or k >= len(Xz):
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(Xz)

        sil = silhouette_score(Xz, labels) if k > 1 else float("nan")
        ch  = calinski_harabasz_score(Xz, labels)
        db  = davies_bouldin_score(Xz, labels)

        counts = np.bincount(labels, minlength=k)
        min_share = counts.min() / counts.sum()
        max_share = counts.max() / counts.sum()

        rows.append([k, float(km.inertia_), float(sil), float(ch), float(db),
                     int(counts.min()), int(counts.max()),
                     float(min_share), float(max_share)])

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k","inertia","silhouette","calinski_harabasz","davies_bouldin",
                        "min_cluster_size","max_cluster_size","min_share","max_share"])
            w.writerows(rows)

    # simple rank aggregation: high sil & CH, low DB
    def rank(vals, reverse=False):
        order = sorted(vals, reverse=reverse)
        pos = {v: i for i, v in enumerate(order)}
        return [pos[v] for v in vals]

    ks   = [r[0] for r in rows]
    sils = [r[2] for r in rows]
    chs  = [r[3] for r in rows]
    dbs  = [r[4] for r in rows]

    r_sil = rank(sils, reverse=True)
    r_ch  = rank(chs,  reverse=True)
    r_db  = rank(dbs,  reverse=False)

    score = [r_sil[i] + r_ch[i] + r_db[i] for i in range(len(rows))]
    best  = min(range(len(rows)), key=lambda i: score[i])

    suggestion = {
        "k": int(ks[best]),
        "silhouette": float(sils[best]),
        "calinski_harabasz": float(chs[best]),
        "davies_bouldin": float(dbs[best]),
    }
    return {"rows": rows, "suggestion": suggestion}
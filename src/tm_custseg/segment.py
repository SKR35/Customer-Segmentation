import sqlite3, csv, json, math
from datetime import datetime
from typing import List, Tuple
import numpy as np
from .preproc import preprocess

try:
    from sklearn.cluster import KMeans
except Exception as e:
    KMeans = None

FEATURE_TABLE = "customer_features"
SEGMENTS_TABLE = "customer_segments"
CENTERS_TABLE  = "segment_centers"
    
def _load_features(conn: sqlite3.Connection):
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
    cur  = conn.execute(f"SELECT customer_id, {', '.join(cols)} FROM {FEATURE_TABLE} ORDER BY customer_id")
    ids, X = [], []
    for row in cur.fetchall():
        ids.append(row[0]); X.append([float(v) for v in row[1:]])
    return ids, np.array(X, float), cols

def _ensure_segment_tables(conn: sqlite3.Connection):
    conn.execute(f"""
      CREATE TABLE IF NOT EXISTS {SEGMENTS_TABLE} (
        customer_id TEXT NOT NULL,
        method TEXT NOT NULL,
        k INTEGER NOT NULL,
        segment INTEGER NOT NULL,
        distance REAL,
        created_ts_utc TEXT NOT NULL,
        PRIMARY KEY (customer_id, method, k)
      );
    """)
    conn.execute(f"""
      CREATE TABLE IF NOT EXISTS {CENTERS_TABLE} (
        method TEXT NOT NULL,
        k INTEGER NOT NULL,
        segment INTEGER NOT NULL,
        center_json TEXT NOT NULL,
        created_ts_utc TEXT NOT NULL,
        PRIMARY KEY (method, k, segment)
      );
    """)

def _zscore(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd

def run_kmeans(db_path: str, k: int = 6, seed: int = 42, overwrite: bool = False, min_tx: int = 1):
    if KMeans is None:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")

    conn = sqlite3.connect(db_path)
    try:
        _ensure_segment_tables(conn)

        ids, X, feat_cols = _load_features(conn)
        if X.size == 0:
            raise RuntimeError("customer_features is empty. Run 'extract-features' first.")

        # --- filter: active only ---
        try:
            ntx_idx = feat_cols.index("n_tx")
        except ValueError:
            raise RuntimeError("Feature 'n_tx' not found; re-run 'extract-features'.")
        active_mask = X[:, ntx_idx] >= float(min_tx)
        inactive_ids = [cid for cid, ok in zip(ids, active_mask) if not ok]
        ids_act = [cid for cid, ok in zip(ids, active_mask) if ok]
        X_act   = X[active_mask]

        if len(ids_act) < max(2, k):
            raise RuntimeError(f"Not enough active customers for k={k}. Active={len(ids_act)}. "
                               "Lower --k or regenerate data.")

        Xp, feat_cols = preprocess(X_act, feat_cols)
        Xz, mu, sd = _zscore(Xp)
        
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(Xz)
        centers = km.cluster_centers_
        d = np.linalg.norm(Xz - centers[labels], axis=1)
        created = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        method = "kmeans"

        if overwrite:
            conn.execute(f"DELETE FROM {SEGMENTS_TABLE} WHERE method=? AND k=?", (method, k))
            conn.execute(f"DELETE FROM {CENTERS_TABLE}  WHERE method=? AND k=?", (method, k))

        # write active
        conn.executemany(
            f"INSERT OR REPLACE INTO {SEGMENTS_TABLE} (customer_id, method, k, segment, distance, created_ts_utc) "
            f"VALUES (?,?,?,?,?,?)",
            [(cid, method, k, int(seg), float(dist), created) for cid, seg, dist in zip(ids_act, labels, d)]
        )

        # write inactive as segment = -1 (no distance)
        conn.executemany(
            f"INSERT OR REPLACE INTO {SEGMENTS_TABLE} (customer_id, method, k, segment, distance, created_ts_utc) "
            f"VALUES (?,?,?,?,?,?)",
            [(cid, method, k, -1, None, created) for cid in inactive_ids]
        )

        # centers: store in original space for readability
        centers_orig = centers * sd + mu
        conn.executemany(
            f"INSERT OR REPLACE INTO {CENTERS_TABLE} (method, k, segment, center_json, created_ts_utc) VALUES (?,?,?,?,?)",
            [(method, k, int(i),
              json.dumps({f: float(v) for f, v in zip(feat_cols, centers_orig[i])}),
              created)
             for i in range(k)]
        )

        conn.commit()
        return {"customers": len(ids), "active": len(ids_act), "inactive": len(inactive_ids),
                "k": k, "method": method}
    finally:
        conn.close()

def profile_segments(db_path: str, k: int = 6, method: str = "kmeans"):
    conn = sqlite3.connect(db_path)
    try:
        # counts
        print("segment counts:")
        for seg, cnt in conn.execute(
            f"SELECT segment, COUNT(*) FROM {SEGMENTS_TABLE} WHERE method=? AND k=? GROUP BY segment ORDER BY segment",
            (method, k)
        ).fetchall():
            print(f"  segment {seg}: {cnt}")

        # mean features per segment
        # join features
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
        sel = ", ".join([f"AVG(f.{c}) AS {c}" for c in cols])
        q = f"""
        SELECT s.segment, {sel}
        FROM {SEGMENTS_TABLE} s
        JOIN {FEATURE_TABLE} f ON f.customer_id = s.customer_id
        WHERE s.method=? AND s.k=?
        GROUP BY s.segment
        ORDER BY s.segment;
        """
        print("\nsegment means:")
        for row in conn.execute(q, (method, k)):
            seg = row[0]
            vals = ", ".join(f"{col}={row[i+1]:.2f}" for i, col in enumerate(cols))
            print(f"  [{seg}] {vals}")
    finally:
        conn.close()

def export_segments(db_path: str, k: int, method: str, out_csv: str):
    conn = sqlite3.connect(db_path)
    try:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()]
        header = ["customer_id","method","k","segment","distance"] + cols[1:]
        q = f"""
        SELECT s.customer_id, s.method, s.k, s.segment, s.distance, {', '.join('f.'+c for c in cols[1:])}
        FROM {SEGMENTS_TABLE} s
        JOIN {FEATURE_TABLE} f ON f.customer_id = s.customer_id
        WHERE s.method=? AND s.k=?
        ORDER BY s.customer_id;
        """
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in conn.execute(q, (method, k)):
                w.writerow(row)
    finally:
        conn.close()
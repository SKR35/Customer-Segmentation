import sqlite3, csv, json, math
from typing import Tuple, List
import numpy as np

from .segment import FEATURE_TABLE, SEGMENTS_TABLE, _zscore
from .preproc import preprocess

try:
    from sklearn.decomposition import PCA
except Exception as e:
    PCA = None

def _load_feats_and_labels(conn, method, k):
    # features
    feat_cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
    cur = conn.execute(f"SELECT customer_id, {', '.join(feat_cols)} FROM {FEATURE_TABLE} ORDER BY customer_id")
    ids, X = [], []
    for row in cur.fetchall():
        ids.append(row[0]); X.append([float(v) for v in row[1:]])
    X = np.array(X, float)
    lab = dict(conn.execute(
        f"SELECT customer_id, segment FROM {SEGMENTS_TABLE} WHERE method=? AND k=? AND segment>=0",
        (method, k)
    ).fetchall())
    y = np.array([lab.get(cid, -1) for cid in ids], int)
    return ids, X, y, feat_cols
        
def pca_2d_csv(db_path: str, k: int, method: str, out_csv: str):    
    """Compute 2D PCA on z-scored features and export coordinates + segment label."""
    if PCA is None:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")
    conn = sqlite3.connect(db_path)
    try:
        # load all customers + labels (labels may be -1 for inactive)
        feat_cols = [c[1] for c in conn.execute(f"PRAGMA table_info({FEATURE_TABLE})").fetchall()][1:]
        rows = conn.execute(
            f"SELECT customer_id, {', '.join(feat_cols)} FROM {FEATURE_TABLE} ORDER BY customer_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        X   = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float)

        lab = dict(conn.execute(
            f"SELECT customer_id, segment FROM {SEGMENTS_TABLE} WHERE method=? AND k=?",
            (method, k)
        ).fetchall())
        y = np.array([lab.get(cid, -1) for cid in ids], dtype=int)

        # ---- filter to active customers FIRST ----
        valid = (y >= 0)
        if valid.sum() == 0:
            raise RuntimeError("No active customers for this (method,k). "
                               "Run 'segment --min-tx 1' for that k, or choose a k that exists.")
        ids = np.array(ids)[valid].tolist()
        y   = y[valid]
        X   = X[valid, :]

        # consistent preprocessing + z-score on the filtered matrix
        X, _ = preprocess(X, feat_cols)
        Xz, _, _ = _zscore(X)

        # PCA on active-only rows
        pca = PCA(n_components=2, random_state=0)
        pts = pca.fit_transform(Xz)
        var = pca.explained_variance_ratio_

        # write CSV: customer_id, segment, PC1, PC2
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["customer_id", "segment", "pc1", "pc2"])
            for cid, seg, (x1, x2) in zip(ids, y.tolist(), pts):
                w.writerow([cid, int(seg), float(x1), float(x2)])

        return var
    finally:
        conn.close()

def feature_importance_z(db_path: str, k: int, method: str) -> Tuple[List[str], List[float]]:
    """Simple cluster-importance heuristic per feature (unsupervised):
    stddev of cluster centers across segments in z-space (higher => separates clusters more)."""
    conn = sqlite3.connect(db_path)
    try:
        # load z-scored features and labels
        ids, X, y, feat_cols = _load_feats_and_labels(conn, method, k)
        if X.size == 0:
            return feat_cols, [0.0]*len(feat_cols)
        Xz, mu, sd = _zscore(X)
        valid = y >= 0
        Xz = Xz[valid]; y = y[valid]
        if Xz.size == 0:
            return feat_cols, [0.0]*len(feat_cols)

        centers = []
        for seg in sorted(set(y)):
            centers.append(Xz[y==seg].mean(axis=0))
        C = np.vstack(centers)  # k x p
        scores = C.std(axis=0)  # std across segments per feature (z-space)
        return feat_cols, scores.tolist()
    finally:
        conn.close()

def html_report(db_path: str, k: int, method: str, out_html: str,
                pca_points_csv: str | None = None):
    """Generate a tiny HTML (no deps) with summary, top features, and an inline SVG PCA scatter."""
    # Recompute PCA internally for self-contained report
    var_ratio = pca_2d_csv(db_path, k, method, pca_points_csv or "_pca_tmp.csv")

    # Load PCA points
    pts = []
    with open(pca_points_csv or "_pca_tmp.csv", "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            cid, seg, pc1, pc2 = line.strip().split(",")
            pts.append((cid, int(seg), float(pc1), float(pc2)))

    # Importance ranking
    feat_cols, scores = feature_importance_z(db_path, k, method)
    pairs = sorted(zip(feat_cols, scores), key=lambda x: x[1], reverse=True)
    top = pairs[:10]

    # bounds for SVG
    xs = [p[2] for p in pts]; ys = [p[3] for p in pts]
    if not xs:
        xs=[0.0]; ys=[0.0]
    xmin,xmax = min(xs), max(xs); ymin,ymax = min(ys), max(ys)
    def scale(v, a,b, A,B): 
        return A if a==b else (A + (v-a)*(B-A)/(b-a))
    W,H, pad = 900, 520, 40

    # color palette (10 distinct)
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    # build SVG circles
    circles = []
    for cid, seg, x, y in pts:
        cx = scale(x, xmin,xmax, pad, W-pad)
        cy = scale(y, ymin,ymax, H-pad, pad)  # invert y
        color = palette[seg % len(palette)] if seg >= 0 else "#bbbbbb"
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}" opacity="0.75"><title>{cid} | seg {seg}</title></circle>')

    # legend
    segs = sorted(set([p[1] for p in pts if p[1] >= 0]))
    legend_items = "".join(
        f'<div style="display:inline-block;margin-right:12px">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{palette[s%len(palette)]};margin-right:6px"></span>'
        f'Segment {s}</div>'
        for s in segs
    )

    # top features table rows
    feat_rows = "".join(f"<tr><td>{i+1}</td><td>{name}</td><td>{score:.3f}</td></tr>"
                        for i,(name,score) in enumerate(top))

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Customer Segmentation Report (k={k}, method={method})</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
 h1 {{ margin: 0 0 8px 0; }}
 .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #f1f3f5; }}
 .legend {{ margin-top: 8px; }}
 .muted {{ color: #6b7280; }}
</style>
</head>
<body>
<h1>Customer Segmentation Report</h1>
<div class="muted">Method: <b>{method}</b> &nbsp; k=<b>{k}</b> &nbsp; PCA var ratio: PC1={var_ratio[0]:.2f}, PC2={var_ratio[1]:.2f}</div>

<div class="card">
  <h3>PCA Scatter (PC1 vs PC2)</h3>
  <div class="legend">{legend_items}</div>
  <svg width="{W}" height="{H}" style="border:1px solid #e5e7eb;border-radius:8px;margin-top:8px;background:#fff">
    <!-- axes -->
    <line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#e5e7eb"/>
    <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#e5e7eb"/>
    {"".join(circles)}
  </svg>
  <div class="muted" style="margin-top:6px">Hover points to see customer_id & segment.</div>
</div>

<div class="card">
  <h3>Top features separating segments (z-space std across centers)</h3>
  <table>
    <thead><tr><th>#</th><th>feature</th><th>score</th></tr></thead>
    <tbody>
      {feat_rows}
    </tbody>
  </table>
  <div class="muted">Heuristic (unsupervised): higher score ⇒ feature varies more between segments in standardized space.</div>
</div>

</body></html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
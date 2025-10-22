import argparse
from .features import extract_and_store
from .segment import run_kmeans, profile_segments, export_segments
from .report import pca_2d_csv, html_report
from .explain import explain_importance
from .kselect import sweep_k
from .viz import viz_grouped_importance

def main():
    p = argparse.ArgumentParser(
        prog="tm-custseg",
        description="Customer segmentation on TM SQLite data (features + KMeans + PCA/report + explain)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) feature extraction
    p_feat = sub.add_parser("extract-features", help="Compute per-customer features and store in DB (and optional CSV).")
    p_feat.add_argument("--db", required=True, help="Path to SQLite DB (from fcc-synthetic-tm).")
    p_feat.add_argument("--out-csv", default=None, help="Optional CSV path (e.g., data/customer_features.csv).")

    # 2) segmentation
    p_seg = sub.add_parser("segment", help="Run KMeans on stored features, write segments to DB.")
    p_seg.add_argument("--db", required=True)
    p_seg.add_argument("--k", type=int, default=6, help="Number of clusters.")
    p_seg.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_seg.add_argument("--overwrite", action="store_true", help="Overwrite existing segments for same (method,k).")
    p_seg.add_argument("--min-tx", type=int, default=1, help="Minimum n_tx to include in clustering (default: 1)")  # NEW

    # 3) profile segments
    p_prof = sub.add_parser("profile", help="Print per-segment counts and mean of features.")
    p_prof.add_argument("--db", required=True)
    p_prof.add_argument("--k", type=int, default=6)
    p_prof.add_argument("--method", default="kmeans")

    # 4) export segments
    p_exp = sub.add_parser("export", help="Export segments joined with features to CSV.")
    p_exp.add_argument("--db", required=True)
    p_exp.add_argument("--k", type=int, default=6)
    p_exp.add_argument("--method", default="kmeans")
    p_exp.add_argument("--out-csv", required=True)

    # 5) PCA 2D export
    p_pca = sub.add_parser("pca-2d", help="Export PCA(2D) coordinates with segment labels to CSV.")
    p_pca.add_argument("--db", required=True)
    p_pca.add_argument("--k", type=int, default=6)
    p_pca.add_argument("--method", default="kmeans")
    p_pca.add_argument("--out-csv", required=True)

    # 6) HTML report
    p_rep = sub.add_parser("report", help="Generate tiny HTML report with PCA scatter and top features.")
    p_rep.add_argument("--db", required=True)
    p_rep.add_argument("--k", type=int, default=6)
    p_rep.add_argument("--method", default="kmeans")
    p_rep.add_argument("--out-html", required=True)
    p_rep.add_argument("--pca-csv", default=None, help="Optional path to also save PCA coordinates CSV.")

    # 7) Explain (RF + permutation importance)
    p_ex = sub.add_parser("explain", help="Train a small RF to predict segments, export permutation importances.")
    p_ex.add_argument("--db", required=True)
    p_ex.add_argument("--k", type=int, default=6)
    p_ex.add_argument("--method", default="kmeans")
    p_ex.add_argument("--out-csv", required=True)
    p_ex.add_argument("--seed", type=int, default=42)
    p_ex.add_argument("--test-size", type=float, default=0.25)
    p_ex.add_argument("--n-estimators", type=int, default=300)
    p_ex.add_argument("--n-repeats", type=int, default=20)
    p_ex.add_argument("--force-permutation", action="store_true")
    p_ex.add_argument("--cv", type=int, default=5, help="CV folds for importance (>=2).")
    p_ex.add_argument("--mode", choices=["pi","pi_cv","grouped","dropcol"], default="pi_cv",
    help="Importance method (pi_cv recommended; dropcol/grouped handle collinearity).")
    p_ex.add_argument("--corr-threshold", type=float, default=0.85, help="Correlation threshold for grouped permutation.")
    
    # 8) Suggest k
    p_ks = sub.add_parser("suggest-k", help="Sweep k on ACTIVE customers using existing (method,k) to define the active set.")
    p_ks.add_argument("--db", required=True)
    p_ks.add_argument("--method", default="kmeans", help="Segmentation method used to tag actives.")
    p_ks.add_argument("--k", type=int, default=4, help="Existing k used ONLY to define 'active' rows (segment>=0).")
    p_ks.add_argument("--kmin", type=int, default=2)
    p_ks.add_argument("--kmax", type=int, default=12)
    p_ks.add_argument("--out-csv", default="data\\k_sweep.csv")
    
    # 9) Viz
    p_viz = sub.add_parser("viz-importance", help="Chart correlation-aware (grouped) permutation importance.")
    p_viz.add_argument("--db", required=True)
    p_viz.add_argument("--k", type=int, default=6)
    p_viz.add_argument("--method", default="kmeans")
    p_viz.add_argument("--out-png", required=True)
    p_viz.add_argument("--cv", type=int, default=5)
    p_viz.add_argument("--n-repeats", type=int, default=30)
    p_viz.add_argument("--n-estimators", type=int, default=400)
    p_viz.add_argument("--corr-threshold", type=float, default=0.85)
    p_viz.add_argument("--top-n", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "extract-features":
        n = extract_and_store(args.db, out_csv=args.out_csv)
        print(f"features written: {n}")
        
    elif args.cmd == "segment":
        stats = run_kmeans(args.db, k=args.k, seed=args.seed, overwrite=args.overwrite, min_tx=args.min_tx)
        print("segmentation done: customers={customers} active={active} inactive={inactive} k={k} method={method}".format(**stats))

    elif args.cmd == "profile":
        profile_segments(args.db, k=args.k, method=args.method)

    elif args.cmd == "export":
        export_segments(args.db, k=args.k, method=args.method, out_csv=args.out_csv)
        print(f"exported: {args.out_csv}")

    elif args.cmd == "pca-2d":
        var = pca_2d_csv(args.db, k=args.k, method=args.method, out_csv=args.out_csv)
        print(f"pca written: {args.out_csv} (var ratio: PC1={var[0]:.2f}, PC2={var[1]:.2f})")

    elif args.cmd == "report":
        html_report(args.db, k=args.k, method=args.method, out_html=args.out_html, pca_points_csv=args.pca_csv)
        print(f"report written: {args.out_html}")

    elif args.cmd == "explain":
        stats = explain_importance(
        db_path=args.db, k=args.k, method=args.method, out_csv=args.out_csv,
        seed=args.seed, test_size=args.test_size,
        n_estimators=args.n_estimators, n_repeats=args.n_repeats)
        print(f"importance written: {stats['path']} | top5: {stats['top5']}" + (" | (train PI)" if stats.get("used_train_set") else ""))
        
    elif args.cmd == "explain":
        stats = explain_importance(
        db_path=args.db, k=args.k, method=args.method, out_csv=args.out_csv,
        seed=args.seed, test_size=args.test_size, n_estimators=args.n_estimators,
        n_repeats=args.n_repeats, force_permutation=args.force_permutation,
        cv=args.cv, mode=args.mode, corr_threshold=args.corr_threshold)
        print(f"importance written: {stats['path']} | source={stats['source']} | top5: {stats['top5']}" + (" | (train PI)" if stats.get('used_train_set') else ""))
        
    elif args.cmd == "suggest-k":
        res = sweep_k(args.db, method=args.method, k_active=args.k,
        kmin=args.kmin, kmax=args.kmax, out_csv=args.out_csv)
        s = res["suggestion"]
        print(f"k sweep written: {args.out_csv} | suggested k={s['k']} "
        f"(sil={s['silhouette']:.2f}, CH={s['calinski_harabasz']:.0f}, DB={s['davies_bouldin']:.2f})")
        
    elif args.cmd == "viz-importance":
        res = viz_grouped_importance(
        db_path=args.db, k=args.k, method=args.method, out_png=args.out_png,
        cv=args.cv, n_repeats=args.n_repeats, n_estimators=args.n_estimators,
        corr_threshold=args.corr_threshold, top_n=args.top_n)
        print(f"chart written: {res['path']}")
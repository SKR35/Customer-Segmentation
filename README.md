# TM Customer Segmentation

Customer segmentation on **TM SQLite** data (from <a href="https://github.com/SKR35/FCC-Synthetic-TM" target="_blank" rel="noopener noreferrer">**fcc-synthetic-tm**</a>).  
It computes per-customer behavioral features and clusters customers with **KMeans**.

---

## Quickstart (Conda on Windows)

```bat
conda create -n tmcseg python=3.11 -y
conda activate tmcseg
python --version

pip install -e .```

---

## Steps

1) Extract features

python -m tm_custseg extract-features --db data\fcc_tm.sqlite --out-csv data\customer_features.csv

2) Pick k + then segment with the suggested k

python -m tm_custseg suggest-k --db data\fcc_tm.sqlite --method kmeans --k 4 --kmin 2 --kmax 10 --out-csv data\k_sweep_active.csv

3) Run segmentation

python -m tm_custseg segment --db data\fcc_tm.sqlite --k 4 --overwrite --min-tx 1

4) Inspect profiles

python -m tm_custseg profile --db data\fcc_tm.sqlite --k 4

5) Export segments + features (CSV)

python -m tm_custseg export --db data\fcc_tm.sqlite --k 4 --out-csv data\segments_k4.csv

6) PCA scatter CSV
python -m tm_custseg pca-2d --db data\fcc_tm.sqlite --k 4 --out-csv data\pca2d_k4.csv

7) Tiny HTML report
python -m tm_custseg report --db data\fcc_tm.sqlite --k 4 --out-html data\seg_report_k4.html --pca-csv data\pca2d_k4.csv

8) Explain clusters with RF + permutation importance
python -m tm_custseg explain --db data\fcc_tm.sqlite --k 4 --out-csv data\importance_k4_cv5.csv --mode pi_cv --cv 5

9) Viz importance
python -m tm_custseg viz-importance --db data\fcc_tm.sqlite --k 4 --out-png data\importance_k4_grouped.png --cv 5

## Features

- Counts & amounts: n_tx, sum_amt_major, avg_amt_major, max_amt_major

- Composition: pct_card, pct_atm, pct_cash, pct_in

- Activity / network: active_days, daily_rate, unique_cp, cp_ratio

- Stored in table: customer_features.

## Outputs

customer_segments: (customer_id, method='kmeans', k, segment, distance, created_ts_utc)

segment_centers: cluster centers per segment in original feature space (JSON)

Optional CSV exports (features/segments)

---

## How to run (conda, one-liners)

```bat
# from repo root
conda create -n tmcseg python=3.11 -y
conda activate tmcseg
pip install -e .

python -m tm_custseg extract-features --db data\fcc_tm.sqlite --out-csv data\customer_features.csv
python -m tm_custseg segment --db data\fcc_tm.sqlite --k 4 --overwrite --min-tx 1
python -m tm_custseg suggest-k --db data\fcc_tm.sqlite --method kmeans --k 4 --kmin 2 --kmax 10 --out-csv data\k_sweep_active.csv

python -m tm_custseg profile --db data\fcc_tm.sqlite --k 4
python -m tm_custseg export --db data\fcc_tm.sqlite --k 4 --out-csv data\segments_k4.csv
python -m tm_custseg pca-2d --db data\fcc_tm.sqlite --k 4 --out-csv data\pca2d_k4.csv
python -m tm_custseg report --db data\fcc_tm.sqlite --k 4 --out-html data\seg_report_k4.html --pca-csv data\pca2d_k4.csv
python -m tm_custseg explain --db data\fcc_tm.sqlite --k 4 --out-csv data\importance_k4_cv5.csv --mode pi_cv --cv 5
python -m tm_custseg viz-importance --db data\fcc_tm.sqlite --k 4 --out-png data\importance_k4_grouped.png --cv 5
```
import sqlite3, csv, math
from typing import List, Tuple

FEATURE_TABLE = "customer_features"

FEATURE_COLUMNS = [
    # raw counts / amounts
    "n_tx", "sum_amt_major", "avg_amt_major", "max_amt_major",
    # composition
    "pct_card", "pct_atm", "pct_cash", "pct_in",
    # activity / network
    "active_days", "daily_rate", "unique_cp", "cp_ratio"
]

def _ensure_feature_table(conn: sqlite3.Connection):
    cols = ", ".join(f"{c} REAL" for c in FEATURE_COLUMNS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {FEATURE_TABLE} (
            customer_id TEXT PRIMARY KEY,
            {cols}
        );
    """)

def _rows_from_db(conn: sqlite3.Connection) -> List[Tuple]:
    # Internal customers only; aggregate tx by customer_id (owner of account)
    q = """
    WITH agg AS (
      SELECT
        t.customer_id                            AS cid,
        COUNT(*)                                 AS n_tx,
        SUM(t.amount_minor)                      AS sum_amt,
        AVG(t.amount_minor)                      AS avg_amt,
        MAX(t.amount_minor)                      AS max_amt,
        SUM(CASE WHEN t.channel='CARD' THEN 1 ELSE 0 END) AS n_card,
        SUM(CASE WHEN t.channel='ATM'  THEN 1 ELSE 0 END) AS n_atm,
        SUM(CASE WHEN t.channel='CASH' THEN 1 ELSE 0 END) AS n_cash,
        SUM(CASE WHEN t.direction='IN'  THEN 1 ELSE 0 END) AS n_in,
        SUM(CASE WHEN t.direction='OUT' THEN 1 ELSE 0 END) AS n_out,
        COUNT(DISTINCT DATE(t.ts_utc))           AS active_days,
        COUNT(DISTINCT t.counterparty_customer_id) AS unique_cp
      FROM cash_transactions t
      GROUP BY t.customer_id
    )
    SELECT c.customer_id,
           COALESCE(a.n_tx,0),
           COALESCE(a.sum_amt,0),
           COALESCE(a.avg_amt,0),
           COALESCE(a.max_amt,0),
           COALESCE(a.n_card,0),
           COALESCE(a.n_atm,0),
           COALESCE(a.n_cash,0),
           COALESCE(a.n_in,0),
           COALESCE(a.n_out,0),
           COALESCE(a.active_days,0),
           COALESCE(a.unique_cp,0)
    FROM customers c
    LEFT JOIN agg a ON a.cid = c.customer_id
    WHERE c.is_internal = 1
    ORDER BY c.customer_id;
    """
    return conn.execute(q).fetchall()

def _derive_features(row) -> Tuple[str, list]:
    (cid, n_tx, sum_amt_mn, avg_amt_mn, max_amt_mn,
     n_card, n_atm, n_cash, n_in, n_out, active_days, unique_cp) = row

    n_tx = float(n_tx)
    sum_major = float(sum_amt_mn) / 100.0
    avg_major = float(avg_amt_mn) / 100.0
    max_major = float(max_amt_mn) / 100.0

    pct_card = (n_card / n_tx) if n_tx > 0 else 0.0
    pct_atm  = (n_atm  / n_tx) if n_tx > 0 else 0.0
    pct_cash = (n_cash / n_tx) if n_tx > 0 else 0.0
    pct_in   = (n_in   / n_tx) if n_tx > 0 else 0.0

    active_days = float(active_days)
    daily_rate = (n_tx / active_days) if active_days > 0 else 0.0

    unique_cp = float(unique_cp)
    cp_ratio = (unique_cp / n_tx) if n_tx > 0 else 0.0

    feats = [
        n_tx, sum_major, avg_major, max_major,
        pct_card, pct_atm, pct_cash, pct_in,
        active_days, daily_rate, unique_cp, cp_ratio
    ]
    return cid, feats

def extract_and_store(db_path: str, out_csv: str | None = None) -> int:
    conn = sqlite3.connect(db_path)
    try:
        _ensure_feature_table(conn)
        rows = _rows_from_db(conn)
        data = []
        for r in rows:
            cid, feats = _derive_features(r)
            data.append((cid, *feats))

        # store in DB
        placeholders = ",".join(["?"] * (1 + len(FEATURE_COLUMNS)))
        conn.executemany(
            f"INSERT OR REPLACE INTO {FEATURE_TABLE} (customer_id, {', '.join(FEATURE_COLUMNS)}) VALUES ({placeholders})",
            data
        )
        conn.commit()

        # optional CSV
        if out_csv:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["customer_id"] + FEATURE_COLUMNS)
                w.writerows(data)

        return len(data)
    finally:
        conn.close()
import numpy as np
from typing import List, Iterable, Tuple

DEFAULT_LOG1P = ["n_tx", "sum_amt_major", "avg_amt_major",
                 "max_amt_major", "unique_cp", "daily_rate"]

def preprocess(X: np.ndarray, feat_cols: List[str],
               log1p_cols: Iterable[str] = DEFAULT_LOG1P,
               clip_q: float = 0.99) -> Tuple[np.ndarray, List[str]]:
    """Return a copy of X with gentle de-skew and outlier clipping.
       - log1p on heavy-tailed count/amount features
       - per-column winsorization at clip_q (default 99th)
    """
    Xp = X.copy()
    col2idx = {c:i for i,c in enumerate(feat_cols)}

    # log1p on selected columns (clip to >=0 first)
    for c in log1p_cols:
        if c in col2idx:
            j = col2idx[c]
            Xp[:, j] = np.log1p(np.clip(Xp[:, j], a_min=0.0, a_max=None))

    # light winsorization to tame extreme tails
    lo = np.quantile(Xp, 1.0 - clip_q, axis=0)  # usually tiny/near 0
    hi = np.quantile(Xp, clip_q, axis=0)
    Xp = np.clip(Xp, lo, hi)

    return Xp, feat_cols
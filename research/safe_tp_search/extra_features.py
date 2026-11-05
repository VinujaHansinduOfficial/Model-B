"""Additional causal features (bars <= t only) and an ORACLE diagnostic helper."""
import numpy as np
import pandas as pd


def build(m):
    c = m.close.astype(float); h = m.high.astype(float); l = m.low.astype(float)
    lr = np.log(c).diff()
    out = {}
    atr = m.atr_pct.astype(float)
    for w in (4, 16, 48, 96, 384, 1536):
        rv = lr.rolling(w, min_periods=max(4, w // 2)).std() * 100
        out[f"rv_{w}_over_atr"] = rv / atr
        out[f"range_{w}_atr"] = (100 * (h.rolling(w).max() - l.rolling(w).min()) / c) / atr
        out[f"pos_in_range_{w}"] = (c - l.rolling(w).min()) / (h.rolling(w).max() - l.rolling(w).min())
    for w in (96, 384, 1536):
        out[f"atr_pct_rel_{w}"] = atr / atr.rolling(w, min_periods=w // 2).median()
    out["log_atr_pct"] = np.log(atr)
    # realised swing sizes of the recent past at several depths (causal fractal legs via rolling extremes)
    for w in (12, 24, 48):
        hh = h.rolling(2 * w + 1).max(); ll = l.rolling(2 * w + 1).min()
        out[f"swing_amp_{w}_atr"] = (100 * (hh - ll) / c) / atr
    # lagged versions of a few features
    for lag in (4, 16):
        out[f"ret_lag{lag}_atr"] = (100 * np.log(c / c.shift(lag))) / atr
    out["hour_sin"] = np.sin(2 * np.pi * m.time.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * m.time.dt.hour / 24)
    out["dow"] = m.time.dt.dayofweek.astype(float)
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)


def oracle_bars_since_true_pivot(m):
    """HINDSIGHT: bars since the previous true endpoint pivot. Diagnostic only."""
    pv = m.tp_pivot_time.dt.tz_convert("UTC").astype("int64").to_numpy()
    uniq = np.unique(pv)
    k = np.searchsorted(uniq, pv)
    start = np.where(k > 0, uniq[np.maximum(k - 1, 0)], -1)
    t = m.time.dt.tz_convert("UTC").astype("int64").to_numpy()
    return np.where(start >= 0, (t - start) / (15 * 60 * 1e9), np.nan).astype(np.float32)

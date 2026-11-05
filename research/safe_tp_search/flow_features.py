"""Volume / taker-flow and multi-timeframe causal features from the raw CSV (bars <= t only)."""
import sys
import numpy as np
import pandas as pd

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv, usecols=["time", "open", "high", "low", "close", "volume", "quote_volume", "num_trades",
                                "taker_base_volume"])
c, h, l, v = d.close, d.high, d.low, d.volume
tb = d.taker_base_volume
lr = np.log(c).diff()
f = {}
imb = (2 * tb - v) / v.replace(0, np.nan)            # taker buy minus sell share, in [-1, 1]
for w in (4, 16, 48, 96, 384):
    f[f"taker_imb_{w}"] = (imb * v).rolling(w).sum() / v.rolling(w).sum()
    f[f"vol_z_{w}"] = (np.log(v + 1) - np.log(v + 1).rolling(w * 4, min_periods=w).mean()) / \
                      np.log(v + 1).rolling(w * 4, min_periods=w).std()
    f[f"trades_rel_{w}"] = d.num_trades.rolling(w).mean() / d.num_trades.rolling(w * 8, min_periods=w).mean()
    f[f"signed_vol_ret_{w}"] = (np.sign(lr) * v).rolling(w).sum() / v.rolling(w).sum()
f["avg_trade_size_rel"] = (d.quote_volume / d.num_trades) / (d.quote_volume / d.num_trades).rolling(384, min_periods=96).median()
# multi-timeframe returns and range positions from 1h/4h/1d windows built causally on 15m bars
for name, w in (("1h", 4), ("4h", 16), ("1d", 96), ("3d", 288), ("1w", 672)):
    hh, ll = h.rolling(w).max(), l.rolling(w).min()
    f[f"mtf_ret_{name}"] = np.log(c / c.shift(w))
    f[f"mtf_pos_{name}"] = (c - ll) / (hh - ll)
    f[f"mtf_rng_{name}"] = np.log(hh / ll)
    f[f"mtf_rv_{name}"] = lr.rolling(w, min_periods=max(2, w // 2)).std()
for a, b in (("1h", "1d"), ("4h", "1w")):
    f[f"rv_ratio_{a}_{b}"] = f[f"mtf_rv_{a}"] / f[f"mtf_rv_{b}"]
F = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan).astype(np.float32)
F.to_parquet(out)
print(F.shape)

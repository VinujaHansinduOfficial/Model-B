"""Strict-accuracy experiment harness (CPU, LightGBM). See PROTOCOL.md."""
import numpy as np
import pandas as pd
import lightgbm as lgb

W = np.log(4 / 3)  # log width of the strict window
LOWER = 0.75
BAR = pd.Timedelta(minutes=15)
EMBARGO = 128

FOLDS = {
    "dev_1": ("2025-09-01", "2025-10-31"),
    "dev_2": ("2026-01-01", "2026-03-02"),
    "dev_3": ("2026-04-01", "2026-05-31"),
    "outer": ("2026-07-01", "2026-10-15"),
}
HOLDOUT = ("2026-10-15", "2026-11-02")


def load(feat_dir):
    X = pd.read_parquet(f"{feat_dir}/features.parquet")
    m = pd.read_parquet(f"{feat_dir}/meta.parquet")
    return X, m


def strict_hits(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return np.where(y == 0, yhat == 0, (yhat >= LOWER * y) & (yhat <= y))


def fold_indices(m, start, end, max_train=None):
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    ok = m.finite.to_numpy() & np.isfinite(m.z.to_numpy())
    t = m.time
    test = np.flatnonzero(ok & (t >= s).to_numpy() & (t < e).to_numpy())
    train = np.flatnonzero(ok & (t < s - EMBARGO * BAR).to_numpy() & (m.label_available_at < s).to_numpy())
    cut = int(len(train) * 0.9)
    fit, es = train[:cut], train[cut:]
    es = es[es > fit[-1] + EMBARGO]
    fit = fit[(m.label_available_at.iloc[fit] < m.time.iloc[es[0]]).to_numpy()]
    if max_train and len(fit) > max_train:
        fit = fit[np.linspace(0, len(fit) - 1, max_train).astype(int)]
    return fit, es, test, train


def best_constant_log(lz):
    """Constant c (log) maximizing share of lz in [c, c+W]."""
    s = np.sort(lz[np.isfinite(lz)])
    j = np.searchsorted(s, s + W, side="right")
    i = int(np.argmax(j - np.arange(len(s))))
    # centre the window over the densest run: start at s[i]
    return s[i]


def window_decode(probs, edges):
    """probs [N,K] over log bins with edges [K+1]; piecewise-uniform density.
    Returns log start u maximizing mass in [u, u+W], and that mass."""
    N, K = probs.shape
    cdf = np.concatenate([np.zeros((N, 1)), np.cumsum(probs, axis=1)], axis=1)
    cand = np.unique(np.concatenate([edges, edges - W]))
    cand = cand[(cand >= edges[0]) & (cand <= edges[-1] - W)]
    best_u = np.full(N, cand[0]); best_m = np.full(N, -1.0)
    for u in cand:
        ku = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, K - 1)
        kv = np.clip(np.searchsorted(edges, u + W, side="right") - 1, 0, K - 1)
        fu = cdf[:, ku] + probs[:, ku] * (u - edges[ku]) / (edges[ku + 1] - edges[ku])
        fv = cdf[:, kv] + probs[:, kv] * (u + W - edges[kv]) / (edges[kv + 1] - edges[kv])
        mass = fv - fu
        better = mass > best_m
        best_u[better], best_m[better] = u, mass[better]
    return best_u, best_m


def make_edges(lz_fit, K):
    q = np.quantile(lz_fit, np.linspace(0.002, 0.998, K + 1))
    q = np.maximum.accumulate(q + np.arange(K + 1) * 1e-9)
    return q


def to_class(lz, edges):
    K = len(edges) - 1
    return np.clip(np.searchsorted(edges, np.clip(lz, edges[0], edges[-1]), side="right") - 1, 0, K - 1)


BASE = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=7)


def train(params, xf, yf, xe, ye, rounds=2000, patience=50):
    d = lgb.Dataset(xf, yf, free_raw_data=True)
    v = lgb.Dataset(xe, ye, reference=d)
    return lgb.train({**BASE, **params}, d, num_boost_round=rounds, valid_sets=[v],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def leg_block_ci(hits, legs, reps=2000, seed=1):
    d = pd.DataFrame({"b": legs, "c": hits.astype(int)}).groupby("b").c.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    s = d[rng.integers(len(d), size=(reps, len(d)))].sum(axis=1)
    r = s[:, 0] / s[:, 1]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

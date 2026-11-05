"""Safe-TP architecture search (CPU). See PROTOCOL_SAFE_TP.md.
Usage: python -I safe_search.py FEAT_DIR OUT_CSV cand1 [cand2 ...]
All models output a per-candle base score b (log TP %) and optionally a scale s; TP = exp(b + k*s),
with k set so that 98% of candles have log y >= b + k*s (calibration slice, or test window for ranking value).
"""
import sys, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd, lightgbm as lgb
from harness import FOLDS, HOLDOUT, BAR, EMBARGO, load, BASE
import extra_features as xfm

GOAL, COST = 0.98, 0.11
feat_dir, out_csv, names = sys.argv[1], sys.argv[2], sys.argv[3:]
X, m = load(feat_dir)
ZZ = pd.read_parquet(f"{feat_dir}/zz_multi.parquet")
EXTRA = xfm.build(m)
FLOW = pd.read_parquet(f"{feat_dir}/flow.parquet")
y = m.tp_pct.to_numpy(); atr = m.atr_pct.to_numpy(); legs = m.leg_id.to_numpy()
ly = np.log(np.where(y > 0, y, np.nan)); la = np.log(atr); lz = ly - la
FEATS = {"base": X, "extra": pd.concat([X, EXTRA], axis=1), "zz": pd.concat([X, ZZ], axis=1),
         "all": pd.concat([X, EXTRA, ZZ], axis=1),
         "zzflow": pd.concat([X, ZZ, FLOW], axis=1), "allflow": pd.concat([X, EXTRA, ZZ, FLOW], axis=1)}
FEATS = {k: v.to_numpy(np.float32) for k, v in FEATS.items()}


def split3(start, end):
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    ok = m.finite.to_numpy() & np.isfinite(y)
    t, la_ = m.time, m.label_available_at
    test = np.flatnonzero(ok & (t >= s).to_numpy() & (t < e).to_numpy())
    tr = np.flatnonzero(ok & (t < s - EMBARGO * BAR).to_numpy() & (la_ < s).to_numpy())
    def cut(idx, frac):
        k = int(len(idx) * (1 - frac)); a, b = idx[:k], idx[k:]
        b = b[b > a[-1] + EMBARGO]
        a = a[(la_.iloc[a] < t.iloc[b[0]]).to_numpy()]
        return a, b
    rest, cal = cut(tr, 0.10)
    core, es = cut(rest, 0.05 / 0.90)
    return core, es, cal, test


def fit_lgb(params, xc, yc, xe, ye, patience=50, rounds=3000):
    d = lgb.Dataset(xc, yc); v = lgb.Dataset(xe, ye, reference=d)
    return lgb.train({**BASE, **params}, d, rounds, valid_sets=[v],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def pos(idx):
    return idx[np.isfinite(ly[idx])]


def tgt_arr(unit):
    return {"z": lz, "y": ly}[unit]


def offset(unit, idx):
    return la[idx] if unit == "z" else np.zeros(len(idx))


# ------------------------------------------------------------------ model families
def fam_const(c, core, es, cal, test):
    off = la if c.get("unit", "z") == "z" else np.zeros(len(la))
    return (off[cal], None), (off[test], None), {}


def fam_quantile(c, core, es, cal, test):
    x, u = FEATS[c.get("feats", "base")], c.get("unit", "z")
    T = tgt_arr(u); pc, pe = pos(core), pos(es)
    if c.get("thin"):
        pc = pc[np.linspace(0, len(pc) - 1, c["thin"]).astype(int)]
    mdls = [fit_lgb(dict(objective="quantile", alpha=c["alpha"], metric="quantile",
                         **{**c.get("params", {}), "seed": 7 + sd}), x[pc], T[pc], x[pe], T[pe])
            for sd in range(c.get("bag", 1))]
    f = lambda i: np.mean([mm.predict(x[i], num_iteration=mm.best_iteration) for mm in mdls], 0) + offset(u, i)
    return (f(cal), None), (f(test), None), {"trees": float(np.mean([mm.best_iteration for mm in mdls]))}


def crossfit_resid(x, T, pc, pe, params):
    """Out-of-fold residuals on core from 2 chronological halves (each model early-stopped on es)."""
    h = len(pc) // 2; a, b = pc[:h], pc[h:]
    r = np.empty(len(pc))
    for tr_, te_, sl in ((a, b, slice(h, None)), (b, a, slice(None, h))):
        mm = fit_lgb(dict(objective="regression", metric="l2", **params), x[tr_], T[tr_], x[pe], T[pe])
        r[sl] = T[te_] - mm.predict(x[te_], num_iteration=mm.best_iteration)
    return r


def fam_locscale(c, core, es, cal, test):
    """mu: L2 regression of log target; s: model of |OOF residual|; TP = mu + k*s (normalised conformal)."""
    x, u = FEATS[c.get("feats", "base")], c.get("unit", "z")
    T = tgt_arr(u); pc, pe = pos(core), pos(es)
    p = c.get("params", {})
    mu = fit_lgb(dict(objective="regression", metric="l2", **p), x[pc], T[pc], x[pe], T[pe])
    r = crossfit_resid(x, T, pc, pe, p)
    # scale model on |r| (L1 -> conditional median of |r|), early-stop on es residuals
    re = T[pe] - mu.predict(x[pe], num_iteration=mu.best_iteration)
    sobj = c.get("scale_obj", "l1")
    if sobj == "lowq":   # model the lower-tail residual directly: conditional 5% quantile of r
        sm = fit_lgb(dict(objective="quantile", alpha=0.05, metric="quantile", **p), x[pc], r, x[pe], re)
        sfun = lambda i: np.maximum(-sm.predict(x[i], num_iteration=sm.best_iteration), 0.05)
    else:
        sm = fit_lgb(dict(objective="regression_l1", metric="l1", **p), x[pc], np.abs(r), x[pe], np.abs(re))
        sfun = lambda i: np.maximum(sm.predict(x[i], num_iteration=sm.best_iteration), 0.05)
    f = lambda i: mu.predict(x[i], num_iteration=mu.best_iteration) + offset(u, i)
    return (f(cal), sfun(cal)), (f(test), sfun(test)), {"trees": mu.best_iteration, "s_trees": sm.best_iteration}


def fam_dist(c, core, es, cal, test):
    x, u = FEATS[c.get("feats", "base")], c.get("unit", "z")
    T = tgt_arr(u); pc, pe = pos(core), pos(es)
    if c.get("thin"):
        pc = pc[np.linspace(0, len(pc) - 1, c["thin"]).astype(int)]
    K = c.get("K", 32)
    edges = np.quantile(T[pc], np.linspace(0, 1, K + 1)); edges[0] -= 1e-6; edges[-1] += 1e-6
    edges = np.maximum.accumulate(edges + np.arange(K + 1) * 1e-9)
    cls = lambda i: np.clip(np.searchsorted(edges, T[i], side="right") - 1, 0, K - 1)
    mdl = fit_lgb(dict(objective="multiclass", num_class=K, metric="multi_logloss", **c.get("params", {})),
                  x[pc], cls(pc), x[pe], cls(pe))
    q = c.get("q", 0.05)
    def f(i):
        P = mdl.predict(x[i], num_iteration=mdl.best_iteration)
        cdf = np.concatenate([np.zeros((len(i), 1)), np.cumsum(P, 1)], 1)
        k = np.clip((cdf < q).sum(1) - 1, 0, K - 1)
        lo, hi = cdf[np.arange(len(i)), k], cdf[np.arange(len(i)), k + 1]
        frac = np.clip((q - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
        return edges[k] + frac * (edges[k + 1] - edges[k]) + offset(u, i)
    return (f(cal), None), (f(test), None), {"trees": mdl.best_iteration}


def fam_ladder(c, core, es, cal, test):
    """Binary GBDTs for P(target >= t_j); base = largest t_j with survival >= p."""
    x, u = FEATS[c.get("feats", "base")], c.get("unit", "z")
    T = tgt_arr(u); pc, pe = pos(core), pos(es)
    levels = c.get("levels", (0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50))
    ts = np.quantile(T[pc], levels)
    S_cal, S_test = [], []
    for t in ts:
        mm = fit_lgb(dict(objective="binary", metric="binary_logloss", **c.get("params", {})),
                     x[pc], (T[pc] >= t).astype(int), x[pe], (T[pe] >= t).astype(int))
        S_cal.append(mm.predict(x[cal], num_iteration=mm.best_iteration))
        S_test.append(mm.predict(x[test], num_iteration=mm.best_iteration))
    p = c.get("p", 0.9)
    def base(S, i):
        S = np.minimum.accumulate(np.stack(S, 1), axis=1)       # survival must fall with t
        # interpolate in t where S crosses p
        out = np.full(len(i), ts[0])
        for j in range(1, len(ts)):
            a, b = S[:, j - 1], S[:, j]
            cross = (a >= p) & (b < p)
            w = np.clip((a - p) / np.maximum(a - b, 1e-9), 0, 1)
            out = np.where(cross, ts[j - 1] + w * (ts[j] - ts[j - 1]), out)
            out = np.where(b >= p, ts[j], out)
        return out + offset(u, i)
    return (base(S_cal, cal), None), (base(S_test, test), None), {}


def fam_mlp(c, core, es, cal, test):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    x, u = FEATS[c.get("feats", "base")], c.get("unit", "z")
    T = tgt_arr(u); pc = pos(core)
    sc = StandardScaler().fit(x[pc]); clipx = lambda a: np.clip(np.nan_to_num(sc.transform(a)), -6, 6)
    mm = MLPRegressor(hidden_layer_sizes=c.get("hidden", (128, 64)), alpha=c.get("l2", 1e-3), batch_size=512,
                      learning_rate_init=1e-3, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8,
                      max_iter=200, random_state=7)
    mm.fit(clipx(x[pc]), T[pc])
    f = lambda i: mm.predict(clipx(x[i])) + offset(u, i)
    return (f(cal), None), (f(test), None), {"iters": mm.n_iter_}


def _qt(x, pc):
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=7)
    qt.fit(np.nan_to_num(x[pc]))
    return lambda i: np.clip(qt.transform(np.nan_to_num(x[i])), -5, 5).astype(np.float32)


def fam_mlpq(c, core, es, cal, test):
    """Quantile (pinball) MLP on quantile-normalised inputs; optional seed ensemble."""
    from np_mlp import QuantileMLP
    x, u = FEATS[c.get("feats", "base")], c.get("unit", "z")
    T = tgt_arr(u); pc, pe = pos(core), pos(es)
    tr = _qt(x, pc); xc, xe = tr(pc), tr(pe)
    outs_cal, outs_test, eps = [], [], []
    for sd in range(c.get("seeds", 1)):
        mm = QuantileMLP(hidden=c.get("hidden", (256, 128)), alpha=c.get("alpha", 0.05), dropout=c.get("dropout", 0.1),
                         wd=c.get("wd", 1e-4), lr=c.get("lr", 1e-3), seed=7 + sd).fit(xc, T[pc], xe, T[pe])
        outs_cal.append(mm.predict(tr(cal))); outs_test.append(mm.predict(tr(test))); eps.append(mm.epochs_run)
    return ((np.mean(outs_cal, 0) + offset(u, cal), None), (np.mean(outs_test, 0) + offset(u, test), None),
            {"epochs": float(np.mean(eps))})


def fam_blend(c, core, es, cal, test):
    """Average (in log space) of member bases; members are candidate names."""
    bc, bt = [], []
    for name, w in c["members"]:
        mc = get(name)
        (a, _), (b, _), _ = FAM[mc["fam"]](mc, core, es, cal, test)
        bc.append(w * a); bt.append(w * b)
    return (np.sum(bc, 0), None), (np.sum(bt, 0), None), {}


FAM = {"mlpq": fam_mlpq, "blend": fam_blend, "const": fam_const, "quantile": fam_quantile, "locscale": fam_locscale, "dist": fam_dist,
       "ladder": fam_ladder, "mlp": fam_mlp}

CANDS = {
    "fixed_atr": dict(fam="const", unit="z"),
    "fixed_tp": dict(fam="const", unit="y"),
    "q02": dict(fam="quantile", alpha=0.02),
    "q05": dict(fam="quantile", alpha=0.05),
    "q10": dict(fam="quantile", alpha=0.10),
    "q25": dict(fam="quantile", alpha=0.25),
    "q50": dict(fam="quantile", alpha=0.50),
    "q05_y": dict(fam="quantile", alpha=0.05, unit="y"),
    "ls_l1": dict(fam="locscale"),
    "ls_lowq": dict(fam="locscale", scale_obj="lowq"),
    "dist32_q05": dict(fam="dist", K=32, q=0.05, thin=40000),
    "dist32_q10": dict(fam="dist", K=32, q=0.10, thin=40000),
    "ladder_p90": dict(fam="ladder", p=0.90),
    "ladder_p80": dict(fam="ladder", p=0.80),
    "mlp": dict(fam="mlp"),
    "hp_slow": dict(fam="quantile", alpha=0.05, feats="zz", params=dict(learning_rate=0.02)),
    "hp_reg": dict(fam="quantile", alpha=0.05, feats="zz", params=dict(num_leaves=15, min_data_in_leaf=800, lambda_l2=10.0)),
    "hp_deep": dict(fam="quantile", alpha=0.05, feats="zz", params=dict(num_leaves=63, min_data_in_leaf=100)),
    "hp_ff05": dict(fam="quantile", alpha=0.05, feats="zz", params=dict(feature_fraction=0.5)),
    "hp_min800": dict(fam="quantile", alpha=0.05, feats="zz", params=dict(min_data_in_leaf=800)),
    "hp_leaf7": dict(fam="quantile", alpha=0.05, feats="zz", params=dict(num_leaves=7, learning_rate=0.03)),
    "bag5": dict(fam="quantile", alpha=0.05, feats="zz", bag=5),
    "q03zz": dict(fam="quantile", alpha=0.03, feats="zz"),
    "q07zz": dict(fam="quantile", alpha=0.07, feats="zz"),
    "mlpq": dict(fam="mlpq"),
    "mlpq_wide": dict(fam="mlpq", hidden=(512, 256, 128), dropout=0.2, wd=3e-4),
    "mlpq_small": dict(fam="mlpq", hidden=(64, 32), dropout=0.0, wd=1e-3),
    "mlpq_ens5": dict(fam="mlpq", seeds=5),
    "blend_gbdt_mlp": dict(fam="blend", members=[("q05@zz", 0.5), ("mlpq@zz", 0.5)]),
    "blend_gbdt_mlp_73": dict(fam="blend", members=[("q05@zz", 0.7), ("mlpq@zz", 0.3)]),
}
# feature-set variants of every LGB family, generated on demand: "<cand>@<feats>"
def get(name):
    if "@" in name:
        b, f = name.split("@"); return {**CANDS[b], "feats": f}
    return CANDS[name]


def score(bc, sc, bt, st, cal, test):
    """Calibrated TP and ranking value. Shift k: normalised if scale given."""
    def k_for(b, s, i, target):
        r = ly[i] - b
        if s is not None:
            r = r / s
        r = np.where(np.isfinite(r), r, -np.inf)
        n = len(r); kk = int(np.floor((1 - target) * (n + 1))) - 1
        return np.sort(r)[max(kk, 0)]
    def tp(b, s, k):
        return np.exp(b + k * (s if s is not None else 1.0))
    k_cal = k_for(bc, sc, cal, GOAL)
    tp_cal = tp(bt, st, k_cal)
    hits = np.isfinite(ly[test]) & (tp_cal <= y[test])
    k_or = k_for(bt, st, test, GOAL)
    tp_or = tp(bt, st, k_or)
    return {"cal_acc": float(hits.mean()), "cal_mean_tp": float(tp_cal.mean()), "cal_median_tp": float(np.median(tp_cal)),
            "cal_below_cost": float((tp_cal < COST).mean()),
            "rank_mean_tp": float(tp_or.mean()), "rank_below_cost": float((tp_or < COST).mean())}


if __name__ == "__main__":
    rows = []
    for name in names:
        c = get(name)
        for fold, (s, e) in FOLDS.items():
            t0 = time.time()
            core, es, cal, test = split3(s, e)
            (bc, sc), (bt, st), info = FAM[c["fam"]](c, core, es, cal, test)
            r = {"cand": name, "fold": fold, "n": len(test), **score(bc, sc, bt, st, cal, test), **info,
                 "secs": round(time.time() - t0, 1)}
            rows.append(r); print(json.dumps(r), flush=True)
    df = pd.DataFrame(rows)
    try:
        old = pd.read_csv(out_csv); df = pd.concat([old[~old.cand.isin(names)], df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(out_csv, index=False)
    g = df.groupby("cand")
    summ = pd.DataFrame({"rank_mean_tp": g.rank_mean_tp.mean(), "cal_mean_tp": g.cal_mean_tp.mean(),
                         "cal_min_acc": g.cal_acc.min(), "cal_mean_acc": g.cal_acc.mean(),
                         "cal_below_cost": g.cal_below_cost.mean(),
                         **{f"rank_{f}": df[df.fold == f].set_index("cand").rank_mean_tp for f in FOLDS}})
    print(summ.sort_values("rank_mean_tp", ascending=False).round(4).to_string())

"""Run named experiments over the four tuning folds. Appends rows to results.csv.
Usage: python -I run_exp.py FEAT_DIR OUT_CSV exp1 [exp2 ...]
"""
import sys, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from harness import *
import extra_features as xf_mod

feat_dir, out_csv, names = sys.argv[1], sys.argv[2], sys.argv[3:]
X, m = load(feat_dir)
z = m.z.to_numpy()
lz = np.log(np.where(z > 0, z, np.nan))
atr = m.atr_pct.to_numpy()
y = m.tp_pct.to_numpy()
legs = m.leg_id.to_numpy()

EXTRA = None
def features(kind):
    global EXTRA
    base = X.to_numpy(np.float32)
    if kind == "base":
        return base
    if EXTRA is None:
        EXTRA = xf_mod.build(m).to_numpy(np.float32)
    if kind == "extra":
        return np.concatenate([base, EXTRA], 1)
    if kind == "oracle_dir":
        return np.concatenate([base, m.label_side.to_numpy(np.float32)[:, None]], 1)
    if kind == "oracle_dir_pos":
        since = xf_mod.oracle_bars_since_true_pivot(m)
        return np.concatenate([base, m.label_side.to_numpy(np.float32)[:, None], since[:, None]], 1)
    if kind == "oracle_remaining":
        rem = ((m.tp_pivot_time - m.time).dt.total_seconds()/900).to_numpy(np.float32)
        return np.concatenate([base, m.label_side.to_numpy(np.float32)[:, None], np.log1p(rem)[:, None]], 1)
    raise KeyError(kind)

EXPS = {
    "fixed_atr": dict(kind="const_z"),
    "fixed_tp": dict(kind="const_y"),
    "reg_logz": dict(kind="reg", feats="base", params={}),
    "dist32": dict(kind="dist", feats="base", K=32, params={}, max_train=40000),
    "dist32_full": dict(kind="dist", feats="base", K=32, params={}, max_train=None),
    "dist48": dict(kind="dist", feats="base", K=48, params={}, max_train=40000),
    "dist32_small": dict(kind="dist", feats="base", K=32, max_train=40000,
                         params=dict(num_leaves=15, min_data_in_leaf=800, learning_rate=0.03, lambda_l2=10.0)),
    "dist32_big": dict(kind="dist", feats="base", K=32, max_train=40000,
                       params=dict(num_leaves=63, min_data_in_leaf=100)),
    "dist32_extra": dict(kind="dist", feats="extra", K=32, params={}, max_train=40000),
    "reg_logz_extra": dict(kind="reg", feats="extra", params={}),
    "dist32_ytarget": dict(kind="dist", feats="base", K=32, params={}, max_train=40000, target="y"),
    "ORACLE_remaining_reg": dict(kind="reg", feats="oracle_remaining", params={}),
    "ORACLE_remaining_dist32": dict(kind="dist", feats="oracle_remaining", K=32, params={}, max_train=40000),
    "ORACLE_dir_dist32": dict(kind="dist", feats="oracle_dir", K=32, params={}, max_train=40000),
    "ORACLE_dirpos_dist32": dict(kind="dist", feats="oracle_dir_pos", K=32, params={}, max_train=40000),
}

rows = []
for name in names:
    e = EXPS[name]
    for fold, (s, t_end) in FOLDS.items():
        t0 = time.time()
        fit, es, test, tr_idx = fold_indices(m, s, t_end, e.get("max_train"))
        info = {}
        if e["kind"] == "const_z":
            c = best_constant_log(lz[tr_idx])
            pred = np.exp(c) * atr[test]
            info["const"] = float(np.exp(c))
        elif e["kind"] == "const_y":
            c = best_constant_log(np.log(np.where(y[tr_idx] > 0, y[tr_idx], np.nan)))
            pred = np.full(len(test), np.exp(c))
        else:
            x = features(e["feats"])
            tgt = np.log(np.where(y > 0, y, np.nan)) if e.get("target") == "y" else lz
            scale = 1.0 if e.get("target") == "y" else atr
            pf, pe = fit[np.isfinite(tgt[fit])], es[np.isfinite(tgt[es])]
            if e["kind"] == "reg":
                mdl = train(dict(objective="regression", metric="l2", **e["params"]), x[pf], tgt[pf], x[pe], tgt[pe])
                pe_pred = mdl.predict(x[pe], num_iteration=mdl.best_iteration)
                # offset placing the strict window optimally, fitted on the early-stopping set
                r = tgt[pe] - pe_pred
                off = best_constant_log(r)
                lp = mdl.predict(x[test], num_iteration=mdl.best_iteration) + off
                info.update(trees=mdl.best_iteration, offset=float(off),
                            r2_test=float(1 - np.nanmean((tgt[test] - lp + off) ** 2) / np.nanvar(tgt[test])))
            else:
                edges = make_edges(tgt[pf], e["K"])
                mdl = train(dict(objective="multiclass", num_class=e["K"], metric="multi_logloss", **e["params"]),
                            x[pf], to_class(tgt[pf], edges), x[pe], to_class(tgt[pe], edges))
                probs = mdl.predict(x[test], num_iteration=mdl.best_iteration)
                lp, mass = window_decode(probs, edges)
                info.update(trees=mdl.best_iteration, mean_mass=float(mass.mean()))
            pred = np.exp(lp) * (scale[test] if np.ndim(scale) else 1.0)
        hits = strict_hits(y[test], pred)
        rows.append(dict(exp=name, fold=fold, n=len(test), hits=int(hits.sum()), acc=float(hits.mean()),
                         secs=round(time.time() - t0, 1), **info))
        print(json.dumps(rows[-1]), flush=True)

df = pd.DataFrame(rows)
try:
    old = pd.read_csv(out_csv)
    df = pd.concat([old[~old.exp.isin(names)], df], ignore_index=True)
except FileNotFoundError:
    pass
df.to_csv(out_csv, index=False)
summ = df.groupby("exp").apply(lambda g: pd.Series({"pooled": g.hits.sum() / g.n.sum(),
       **{f: g.loc[g.fold == f, "acc"].iloc[0] for f in FOLDS if (g.fold == f).any()}}), include_groups=False)
print(summ.sort_values("pooled", ascending=False).round(4).to_string())

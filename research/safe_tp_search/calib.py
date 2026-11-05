"""Compare calibration schemes for the selected model (GBDT q05 + multi-depth ZigZag).
Rolling schemes recompute the shift at the start of each UTC day from the most recent N candles whose
labels had matured (label_available_at < day start) — strictly causal."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
CAND = sys.argv[2] if len(sys.argv) > 2 else "q05@zz"
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null", CAND]
import numpy as np, pandas as pd
import safe_search as S

cand = S.get(CAND)
rows, perday = [], []
for fold, (s, e) in S.FOLDS.items():
    core, es, cal, test = S.split3(s, e)
    (bc, _), (bt, _), _ = S.FAM[cand["fam"]](cand, core, es, cal, test)
    pool = np.concatenate([cal, test]); bpool = np.concatenate([bc, bt])
    r_pool = S.ly[pool] - bpool
    r_pool = np.where(np.isfinite(r_pool), r_pool, -np.inf)
    ns = lambda col: col.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    avail = ns(S.m.label_available_at)[pool]
    t_test = ns(S.m.time)[test]
    days = pd.DatetimeIndex(t_test).floor("D").to_numpy()
    def k_of(r, target):
        n = len(r); kk = int(np.floor((1 - target) * (n + 1))) - 1
        return np.sort(r)[max(kk, 0)]
    schemes = {}
    for target in (0.98, 0.985, 0.99):
        rc = S.ly[cal] - bc
        schemes[f"static_{target}"] = np.full(len(test), k_of(np.where(np.isfinite(rc), rc, -np.inf), target))
        for N in (1920, 2880, 5760, 8640):
            k = np.empty(len(test))
            for d in np.unique(days):
                sel = days == d
                ok = avail < d
                rr = r_pool[ok][-N:]
                k[sel] = k_of(rr, target)
            schemes[f"roll{N}_{target}"] = k
    for name, k in schemes.items():
        tp = np.exp(bt + k)
        hit = np.isfinite(S.ly[test]) & (tp <= S.y[test])
        wk = pd.Series(hit, index=pd.DatetimeIndex(t_test)).resample("W").mean()
        rows.append(dict(scheme=name, fold=fold, acc=hit.mean(), mean_tp=tp.mean(), below_cost=(tp < S.COST).mean(),
                         worst_week=wk.min()))
df = pd.DataFrame(rows)
df.to_csv(f"calib_{CAND}.csv", index=False)
g = df.groupby("scheme")
out = pd.DataFrame({"min_window_acc": g.acc.min(), "mean_acc": g.acc.mean(), "mean_tp": g.mean_tp.mean(),
                    "below_cost": g.below_cost.mean(), "worst_week": g.worst_week.min(),
                    **{f: df[df.fold == f].set_index("scheme").acc for f in S.FOLDS}})
print(out.sort_values("mean_tp", ascending=False).round(4).to_string())

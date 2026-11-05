"""Score the selected config (dist48) and fixed ATR ONCE on the untouched holdout."""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from harness import *

X, m = load(sys.argv[1])
z = m.z.to_numpy(); lz = np.log(np.where(z > 0, z, np.nan))
atr, y, legs = m.atr_pct.to_numpy(), m.tp_pct.to_numpy(), m.leg_id.to_numpy()
fit, es, test, tr = fold_indices(m, *HOLDOUT, max_train=40000)
x = X.to_numpy(np.float32)
pf, pe = fit[np.isfinite(lz[fit])], es[np.isfinite(lz[es])]
edges = make_edges(lz[pf], 48)
mdl = train(dict(objective="multiclass", num_class=48, metric="multi_logloss"),
            x[pf], to_class(lz[pf], edges), x[pe], to_class(lz[pe], edges))
lp, mass = window_decode(mdl.predict(x[test], num_iteration=mdl.best_iteration), edges)
h_model = strict_hits(y[test], np.exp(lp) * atr[test])
c = best_constant_log(lz[tr])
h_fixed = strict_hits(y[test], np.exp(c) * atr[test])
diff = h_model.astype(int) - h_fixed.astype(int)
res = {"holdout": HOLDOUT, "candles": len(test), "legs": int(len(np.unique(legs[test]))),
       "train_rows": len(tr), "trees": mdl.best_iteration,
       "model_dist48": {"acc": float(h_model.mean()), "ci95_leg": leg_block_ci(h_model, legs[test])},
       "fixed_atr": {"multiple": float(np.exp(c)), "acc": float(h_fixed.mean()), "ci95_leg": leg_block_ci(h_fixed, legs[test])},
       "lift_pp": float(100 * diff.mean())}
print(json.dumps(res, indent=1))
json.dump(res, open("holdout_result.json", "w"), indent=1)

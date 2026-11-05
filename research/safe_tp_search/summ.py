import sys, pandas as pd
df = pd.read_csv(sys.argv[1]); F = ["dev_1","dev_2","dev_3","outer"]
g = df.groupby("cand")
s = pd.DataFrame({"rank_tp": g.rank_mean_tp.mean(), "cal_tp": g.cal_mean_tp.mean(), "cal_min_acc": g.cal_acc.min(),
                  "below_cost": g.cal_below_cost.mean(), **{f: df[df.fold==f].set_index("cand").rank_mean_tp for f in F}})
base = s.loc["q05"]
s["vs_q05_%"] = 100*(s.rank_tp/base.rank_tp-1)
s["folds_worse"] = sum((s[f] < base[f]-1e-9).astype(int) for f in F)
print(s.sort_values("rank_tp", ascending=False).round(4).to_string())

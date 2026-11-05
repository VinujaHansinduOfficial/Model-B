"""Causal ZigZag leg-state features at extra depths (3, 5, 8, 24), using the notebook's causal_zigzag_state."""
import json, sys, types
from unittest import mock
nb_path, feat_dir = sys.argv[1], sys.argv[2]
torch = types.ModuleType("torch"); torch.nn = mock.MagicMock(); torch.cuda = mock.MagicMock()
torch.backends = mock.MagicMock()
for n, m_ in {"torch": torch, "torch.nn": torch.nn, "torch.nn.functional": mock.MagicMock(),
              "torch.utils": mock.MagicMock(), "torch.utils.data": mock.MagicMock(),
              "torch.utils.checkpoint": mock.MagicMock()}.items():
    sys.modules[n] = m_
nbmod = types.ModuleType("nbf"); sys.modules["nbf"] = nbmod
nb = json.load(open(nb_path, encoding="utf-8"))
for i in (5, 11, 13):
    exec(compile("".join(nb["cells"][i]["source"]), f"cell{i}", "exec"), nbmod.__dict__)
import numpy as np, pandas as pd
m = pd.read_parquet(f"{feat_dir}/meta.parquet")
h, l, c = (m[k].to_numpy(float) for k in ("high", "low", "close"))
atr = m.atr_pct.to_numpy() * c / 100
cols = {}
for d in (3, 5, 8, 24):
    st, side, _ = nbmod.causal_zigzag_state(h, l, c, atr, d, 0.0)
    for j in range(12):
        cols[f"zz{d}_{j}"] = st[:, j]
pd.DataFrame(cols).to_parquet(f"{feat_dir}/zz_multi.parquet")
print("zz_multi", len(cols))

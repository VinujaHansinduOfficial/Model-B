"""Extract the notebook's feature pipeline (cells 5, 11, 13) and build the 97 causal features on CPU.

torch is not installed here; the config cell only imports it, so a stub module stands in.
Usage: python -I build_features.py NOTEBOOK CSV OUT_PARQUET_DIR
"""
import json
import sys
import types
from unittest import mock

nb_path, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

# Stub torch so the config cell's imports succeed.
torch = types.ModuleType("torch")
torch.nn = mock.MagicMock()
torch.cuda = mock.MagicMock()
torch.cuda.is_available = lambda: False
torch.backends = mock.MagicMock()
torch.manual_seed = lambda s: None
for name, mod in {
    "torch": torch,
    "torch.nn": torch.nn,
    "torch.nn.functional": mock.MagicMock(),
    "torch.utils": mock.MagicMock(),
    "torch.utils.data": mock.MagicMock(),
    "torch.utils.checkpoint": mock.MagicMock(),
}.items():
    sys.modules[name] = mod

nb = json.load(open(nb_path, encoding="utf-8"))
nbmod = types.ModuleType("nbfeatures")
sys.modules["nbfeatures"] = nbmod
ns = nbmod.__dict__
for i in (5, 11, 13):
    src = "".join(nb["cells"][i]["source"])
    exec(compile(src, f"cell{i}", "exec"), ns)

import numpy as np
import pandas as pd
from pathlib import Path

cfg = ns["Config"]()
meta = {"confirmation_timestamp_is_bar_open": True, "maturity_verified": False, "live_side_verified": False}
assets = ns["prepare_assets"]({"BTCUSDT": csv_path}, {"BTCUSDT": meta}, cfg)
a = assets["BTCUSDT"]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
feat = a.features.copy()
feat.to_parquet(out / "features.parquet")
raw = a.raw
keep = ["time", "open", "high", "low", "close", "tp_pct", "tp_pivot_time", "tp_confirm_time",
        "label_available_at", "label_side", "leg_id", "signal_live"]
meta_df = raw[[c for c in keep if c in raw]].copy()
meta_df["atr_pct"] = a.atr_pct
meta_df["z"] = a.z
meta_df["finite"] = a.finite_features
meta_df["zz_side"] = a.side
meta_df.to_parquet(out / "meta.parquet")
print(json.dumps({k: v for k, v in a.info.items() if k != "metadata"}, default=str, indent=1))
print("features", feat.shape)

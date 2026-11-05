"""Sanity tests: decoder vs brute force, constant baseline, strict hits."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
from harness import window_decode, best_constant_log, strict_hits, W

rng = np.random.default_rng(0)
K = 20
edges = np.sort(rng.normal(1, 0.8, K + 1))
probs = rng.dirichlet(np.ones(K), size=50)
u, m = window_decode(probs, edges)
# brute force on a fine grid
grid = np.linspace(edges[0], edges[-1] - W, 4000)
cdf = np.concatenate([np.zeros((50, 1)), np.cumsum(probs, 1)], 1)
def F(x):
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, K - 1)
    return cdf[:, k] + probs[:, k] * (x - edges[k]) / (edges[k + 1] - edges[k])
bf = np.max(np.stack([F(g + W) - F(g) for g in grid]), axis=0)
assert np.all(m >= bf - 1e-9), (m - bf).min()
assert np.all(m <= bf + 1e-3)
# constant baseline on lognormal: ~0.288/(0.86*2.507)
lz = rng.normal(1, 0.86, 200000)
c = best_constant_log(lz)
acc = np.mean((lz >= c) & (lz <= c + W))
assert abs(acc - 0.1334) < 0.01, acc
# strict hits
assert strict_hits([1.0, 1.0, 1.0, 0.0], [0.75, 1.0, 1.01, 0.0]).tolist() == [True, True, False, True]
print("ok; lognormal constant ceiling", round(acc, 4))

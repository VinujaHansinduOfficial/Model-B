"""Small NumPy MLP with pinball (quantile) loss, Adam, dropout, weight decay, early stopping."""
import numpy as np


class QuantileMLP:
    def __init__(self, hidden=(256, 128), alpha=0.05, lr=1e-3, wd=1e-4, dropout=0.1, batch=512,
                 epochs=60, patience=6, seed=7):
        self.h, self.alpha, self.lr, self.wd, self.p = hidden, alpha, lr, wd, dropout
        self.batch, self.epochs, self.patience = batch, epochs, patience
        self.rng = np.random.default_rng(seed)

    def _init(self, d):
        sizes = [d, *self.h, 1]
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.m = [np.zeros_like(w) for w in self.W + self.b]
        self.v = [np.zeros_like(w) for w in self.W + self.b]
        self.t = 0

    def _forward(self, x, train):
        acts, masks = [x], []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
                if train and self.p > 0:
                    mk = (self.rng.random(h.shape) > self.p).astype(np.float32) / (1 - self.p)
                    h = h * mk
                else:
                    mk = None
                masks.append(mk)
            acts.append(h)
        return acts, masks

    def predict(self, x):
        out = []
        for i in range(0, len(x), 8192):
            out.append(self._forward(x[i:i + 8192], False)[0][-1][:, 0])
        return np.concatenate(out) + self.y0

    def _loss(self, x, y):
        r = y - self.predict(x)
        return float(np.mean(np.maximum(self.alpha * r, (self.alpha - 1) * r)))

    def fit(self, x, y, xe, ye):
        x, xe = x.astype(np.float32), xe.astype(np.float32)
        self.y0 = float(np.quantile(y, self.alpha))
        yc = (y - self.y0).astype(np.float32)
        self._init(x.shape[1])
        best, best_state, bad = np.inf, None, 0
        n = len(x)
        for ep in range(self.epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                acts, masks = self._forward(x[idx], True)
                pred = acts[-1][:, 0]
                r = yc[idx] - pred
                g = np.where(r > 0, -self.alpha, 1 - self.alpha).astype(np.float32)[:, None] / len(idx)
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ g + self.wd * self.W[i]
                    gb[i] = g.sum(0)
                    if i > 0:
                        g = g @ self.W[i].T
                        g = g * (acts[i] > 0)
                        if masks[i - 1] is not None:
                            g = g * (masks[i - 1] > 0) / (1 - self.p)
                self.t += 1
                params, grads = self.W + self.b, gW + gb
                for k, (pp, gg) in enumerate(zip(params, grads)):
                    self.m[k] = 0.9 * self.m[k] + 0.1 * gg
                    self.v[k] = 0.999 * self.v[k] + 0.001 * gg * gg
                    mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
                    pp -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            le = self._loss(xe, ye)
            if le < best - 1e-5:
                best, bad = le, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.W, self.b = best_state
        self.epochs_run = ep + 1
        return self

# Architecture search for `tp_pct` (Nov 2026)

CPU-only search for the best model for Model B's TP, run on `data_raw_with_indicators_tp_20261102_cleaned_continuous.csv` (sha256 `b5819fdb…a903`, 99,416 BTCUSDT 15m candles, 2024-01-01 → 2026-11-01). The rules were written before training: see `PROTOCOL.md` (strict accuracy) and `PROTOCOL_SAFE_TP.md` (safe TP).

## Result

**Best architecture: LightGBM 5% quantile regression on log Z (Z = tp_pct / ATR14%), on the 97 notebook features plus causal ZigZag leg state at depths 3, 5, 8 and 24 (48 features), with causal rolling recalibration.**

```
OHLCV ─► 97 notebook features ─┐
       └► causal ZigZag state at depths 3/5/8/24 (12 × 4) ─┤
                                                           ▼
                         LightGBM quantile, alpha = 0.05, target log Z
                         (lr 0.05, 31 leaves, min 200/leaf, ff/bagging 0.8, L2 1, early stop 50)
                                                           ▼
                         shift k = conformal order statistic of (log tp − base) over the
                         last 20 days of candles whose labels have matured (recomputed daily)
                                                           ▼
                         TP% = exp(base + k) × ATR%
```

### Safe TP (0 < TP ≤ actual), mean TP at 98% accuracy, four walk-forward test periods

The ranking value sets the shift on each test window. It measures model quality only and can't be deployed.

| Model | Mean TP | vs original | Folds better |
|---|---|---|---|
| **GBDT q05 + multi-depth ZigZag** | **0.184%** | **+2.8%** | **4/4** |
| GBDT q05 + all packs (extra, ZigZag, flow) | 0.186% | +3.7% | 4/4 |
| GBDT q05, original 97 features (notebook) | 0.179% | — | — |
| Location–scale GBDT (mean + lower-tail scale) | 0.179% | −0.2% | 2/4 |
| 32-bin distribution GBDT, 5% quantile | 0.175% | −2.3% | 0/4 |
| Threshold ladder (10 binary GBDTs, survival 0.9) | 0.173% | −3.6% | 0/4 |
| GBDT + quantile-MLP blend (70/30) | 0.182% | +1.8% | 3/4 |
| Quantile MLP (NumPy, pinball loss), best size | 0.175% | −2.2% | 1/4 |
| Plain MLP (average-size target) | 0.100% | −44% | 0/4 |
| Fixed ATR | 0.126% | −29% | 0/4 |

Hyperparameters (slower learning, more regularisation, deeper or shallower trees, feature fraction, 5-seed bagging) and quantile levels 3–7% all moved the result by under 2%. That's noise, so the defaults stay.

### Calibration (the selected model, target ≥ 98% in every window)

| Scheme | Worst window | Mean TP | Below 0.11% |
|---|---|---|---|
| Static shift from the training tail, 98% | 97.5% | 0.184% | 30% |
| Rolling 60-day, 98% | 97.9% | 0.184% | 29% |
| **Rolling 20-day, 98.5%** | **98.4%** | **0.169%** | **34%** |
| Original model, rolling 20-day, 98.5% | 98.3% | 0.164% | 35% |

### Untouched holdout (2026-10-15 → 2026-11-01, 1,721 candles, scored once)

| Model, rolling 20-day 98.5% | Accuracy | Mean TP | Below 0.11% |
|---|---|---|---|
| **Selected model** | 98.4% | **0.130%** | 45% |
| Original notebook GBDT | 98.4% | 0.125% | 48% |
| Fixed ATR | 98.4% | 0.085% | 78% |

## Strict accuracy (0.75 × actual ≤ predicted ≤ actual)

Every model scored 13–15%, the same as fixed ATR (14.5% across the test periods). That covers regression, 32/48-bin distributions with an exact window decoder, hyperparameter variants and extra features. Holdout: 13.1% vs 12.0%, with overlapping 95% intervals. Hindsight diagnostics:

- true direction: 14.5%
- true direction + position in the leg: 14.4%
- true direction + exact bars until the leg ends: 19.9%

50% would need ≈ 94% of the log-Z variance explained; the best model explains 2–8%.

## What this means

- **Trees beat neural nets here.** With ~8.6k effectively independent samples (candles inside a leg share one endpoint), gradient-boosted trees beat every neural variant, including the V3 TCN–Transformer.
- **More capacity won't help.** The models stop after tens to a few hundred trees. A GPU doesn't help this architecture; it trains in seconds on CPU.
- **More labelled data is the likely next lever.** It was the biggest confirmed gain for Model A: run the ZigZag++ generator on more symbols (ETH, SOL, …) and on longer BTC history, then pool.
- **Execution reach depends on direction, not TP size.** It's still the open trading question.

## Reproduce

```bash
python -I build_features.py ../../model_b_v3_directional_multitask_endpoint.ipynb DATA.csv feat
python -I build_zz_multi.py ../../model_b_v3_directional_multitask_endpoint.ipynb feat
python -I flow_features.py DATA.csv feat/flow.parquet
python -I safe_search.py feat safe.csv fixed_atr q05 q05@zz      # family / feature search
python -I calib.py feat q05@zz                                    # calibration schemes
python -I holdout_safe.py feat q05@zz                             # holdout (score once)
python -I run_exp.py feat results.csv fixed_atr dist48            # strict-accuracy search
```

Needs `numpy pandas lightgbm scikit-learn pyarrow`; no GPU. `build_features.py` runs the notebook's feature cells directly, so features match the notebook exactly. Result files are in `results/`.

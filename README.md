# Model B — take-profit endpoint model for BTCUSDT 15m

Model B predicts a take-profit (TP) for **every** 15-minute BTCUSDT candle: how far price will run in the current ZigZag leg before the next pivot. It is trained in Google Colab on labels from a Python port of the TradingView ZigZag++ indicator (depth 12, deviation 5, backstep 2), Binance Futures BTCUSDT 15m data from 2024-01-01.

The notebook `model_b_v3_directional_multitask_endpoint.ipynb` contains the V3 model and every study used to judge it.

## Target and acceptance rules

| | Definition |
|---|---|
| Target | `tp_pct`: best price reached in the leg's direction after the candle, before the leg's endpoint pivot. Normalised as `Z = tp_pct / ATR14%` |
| Strict accuracy | prediction lands in `[0.75 × actual, actual]` |
| Safe-TP accuracy (Option A) | `0 < predicted ≤ actual` |
| Execution reach | the TP is actually hit on the price path after entry, spread included, within 192 bars |
| Coverage | 100%: one prediction per candle, no confidence filtering |

## Results so far

| Study | Result |
|---|---|
| V2 network, strict accuracy (Jul–Oct 2026) | 14.10%, vs 14.24% for a fixed 2.1246 × ATR. The predictions were effectively constant |
| V3 safe-TP at the 98% goal (window `dev_1_2025-09-01`, mean TP) | fixed ATR 0.095% · fixed TP 0.069% · **GBDT quantile 0.145%** · V3 network 0.096% |
| V3 network training curve (same window) | by epoch 9, train Spearman 0.52 vs validation 0.18; validation strict accuracy fell from 0.153 to 0.130 |
| Live direction (causal ZigZag, depth 12) | agrees with the label direction on 49% of candles |
| Cross-asset inputs (ETH, SOL, BNB, XRP, DOGE) | no gain (strict 14.3% vs 14.0%) |

The leakage checks pass: prefix replay, future perturbation, label-input exclusion and model prefix tests.

### Architecture search (Nov 2026)

A CPU search over model families, features, hyperparameters and calibration is in `research/safe_tp_search/`. Best: **GBDT 5% quantile on log Z + causal ZigZag state at depths 3/5/8/24, with 20-day causal rolling recalibration at 98.5%**. It gives a mean TP of 0.169% with every window ≥ 98.4%, against 0.164% for the original GBDT under the same calibration. On the untouched 15 Oct–1 Nov 2026 holdout: 0.130% vs 0.125% (original) vs 0.085% (fixed ATR). Quantile MLPs, distribution heads, threshold ladders and blends did not beat it. Strict accuracy stays at 13–15% for every model.

### What this means

- **The limit is information, not the network.** The network matches a fixed ATR multiple. The gradient-boosted model gets a ~50% larger TP at the same 98% accuracy.
- **The training set is small in effect.** Candles inside one leg share most of their information (intra-leg correlation of log Z ≈ 0.55). The 86k training candles behave like ~8.6k independent samples, against ~3.8M network parameters.
- **The causal direction is late.** A depth-12 ZigZag confirms a pivot 12 bars after it happens, while the median leg lasts ~14 bars. By confirmation the leg is usually almost over.
- **80% strict accuracy is not a realistic target from OHLCV alone.** It asks for the size of a future swing to within a 25% band; a constant scores ~14%.
- **98% safe-TP accuracy is measured in the hindsight direction.** It is not a trading result. Execution reach in a direction known at the time of the trade, and expected value per trade including the losing trades, are the numbers that matter. With ~0.11% round-trip costs, a 0.145% mean TP leaves a few basis points per winning trade, so the size of the ~2% of losers decides profitability.

## Running the notebook

1. Open the notebook in Google Colab with a GPU runtime (the safe-TP and information studies run on CPU).
2. Put the labelled CSV on Google Drive, or set the `MODEL_B_DATA` environment variable to its path.
3. Run the cells in order. Switches in the configuration cell decide which studies run.

Default switches:

| Switch | Default | Why |
|---|---|---|
| `RUN_CAUSALITY_AUDITS` | on | cheap leakage checks |
| `RUN_SAFE_TP_STUDY` | on | the GBDT safe-TP model is the current best |
| `RUN_SAFE_TP_NEURAL` | off | matched fixed ATR |
| `RUN_V3_SELECTION`, `RUN_V3_REFIT_EVALUATION` | off | expensive, and the network shows no lift |
| `EXECUTION_TEST` | off | set on for a quick code check with a tiny network; its scores mean nothing |

Outputs go to `/content/drive/MyDrive/model_b_runs/<RUN_NAME>/`.

## Required CSV columns

`time, open, high, low, close, volume, quote_volume, num_trades, taker_base_volume, taker_quote_volume, tp_pct, tp_price, tp_pivot_time, tp_confirm_time, signal`, plus optional `signal_live` and `close_time`. Bars must be continuous 15-minute candles in UTC.

## Open items

- **Label timing:** the label-finality audit still uses `tp_confirm_time` as an unverified estimate. Pass the ZigZag++ generator to `audit_label_prefixes` and match `causal_zigzag_state` to it.
- **Direction:** feed a verified live LONG/SHORT column (`direction_source='column'`, `live_side_verified=True`) instead of the late causal ZigZag.
- **Trading evaluation:** report execution reach and expected value per trade in that direction, including the size of the losing trades.

## Requirements

See `requirements.txt`. Colab already includes everything except LightGBM, which the notebook installs when needed.

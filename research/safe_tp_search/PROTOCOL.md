# Strict-accuracy search — protocol (written before any model was trained)

Data: data_raw_with_indicators_tp_20261102_cleaned_continuous.csv, sha256 b5819fdb…a903, 99,416 rows, 2024-01-01 → 2026-11-01.
Features: the notebook's 97 causal features (build_features, causal_zigzag depth 12), plus any additions tested below. Hindsight columns (signal, tp_*, zigzag, label_side) are never inputs, except in variants explicitly named ORACLE, which are diagnostics only.

Metric: strict hit = 0.75 × actual ≤ predicted ≤ actual (zero target accepts only zero). One prediction per candle, 100% coverage.

Tuning folds (walk-forward; train on rows with time < start − 128 bars AND label available before start; last 10% of training is the early-stopping set):
- dev_1: 2025-09-01 → 2025-10-31 (5,760 candles, 335 legs)
- dev_2: 2026-01-01 → 2026-03-02 (5,760 candles, 313 legs)
- dev_3: 2026-04-01 → 2026-05-31 (5,760 candles, 332 legs)
- outer: 2026-07-01 → 2026-10-15 (10,176 candles, 612 legs; inspected before, used as a 4th tuning fold)

Selection rule: highest pooled strict accuracy over the four folds; a change is kept only if it gains ≥ 0.5 pp pooled and does not lose in more than one fold.

Holdout (untouched, scored ONCE with the final selected configuration, no changes afterwards):
- 2026-10-15 → 2026-11-01 (1,721 candles, 99 legs). Small, so the leg-block 95% CI is reported.

Baselines: fixed ATR multiple (best constant Z on training rows) and fixed TP %.

Goal: 50% strict accuracy. Expected ceiling from the target spread: log Z std ≈ 0.86; a constant scores ≈ 13–14%; 50% needs residual log error ≈ 0.21, i.e. ≈ 94% of log-Z variance explained.

# Architecture search for tp_pct — Option A safe TP (written before any search model was trained)

Target rule (user's Option A): a TP is correct when 0 < TP ≤ actual tp_pct. One TP per candle, 100% coverage.
Goal: label accuracy ≥ 98% in every window with the largest mean TP; current best (notebook GBDT quantile + conformal shift) is 0.177% mean TP; user's target 0.20–0.25%, and fewer TPs below the 0.11% cost floor.

Two numbers per candidate, both on the four tuning folds of PROTOCOL.md (dev_1, dev_2, dev_3, outer):
1. **Ranking value (primary for architecture choice):** mean TP % at exactly 98% accuracy on each test window when one multiplicative shift is set on that window, averaged over windows. This measures how well the model orders candles, independent of calibration noise. It is a diagnostic of model quality, not a deployable number.
2. **Calibrated result (deployable):** the shift is fitted on the most recent 10% of each fold's training data (never used for fitting), then applied unchanged. Report mean TP, per-window accuracy and share below 0.11%.

Training split per fold: core | embargo | early-stop (5%) | embargo | calibration (10%), chronological, labels resolving before the next part starts.

Selection: a family/feature change is kept if it raises the pooled ranking value by ≥ 2% relative and does not lower it in more than one fold. Calibration methods are compared on the calibrated result for the chosen model.

Holdout: 2026-10-15 → 2026-11-01 (1,721 candles). It was scored once for strict accuracy (48-bin GBDT) but never for safe TP; it is scored once for safe TP with the final configuration. No changes afterwards.

Hindsight columns are never inputs. Direction for execution reach is not part of this search (TP architecture only); reach is reported where computable.

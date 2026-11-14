# Polymarket FAA Outage Attribution Forecaster (v1)

Forecasts, for every market of a case, the next 240 minutes of Yes prices
as five quantiles (5%, 25%, 50%, 75%, 95%). Scored by mean pinball loss.

## Model

5-leg ensemble (weights 0.28 / 0.28 / 0.18 / 0.06 / 0.20), a fresh
reimplementation of the validated NVIDIA-CEO v18 ensemble, adapted for
the FAA family:

1. EWMA-sigma fan (per-price-level half-life, log-log level-fit shrink,
   gated sum correction, spike reversion).
2. Logit ensemble (0.7 logit-space fan + 0.3 price-space fan).
3. Pooled empirical h-step quantiles (18h window), vol-adaptive spread,
   gated sum drift, net-jump median correction, short-window reversion.
4. Gaussian reshape of leg 1.
5. Regime-adaptive sigma fan (26-min half-life EWMA, level-fit shrink,
   regime calming, gated sum correction, fatness adjustment).

## FAA adaptation (the critical part)

The FAA markets are four INDEPENDENT attribution questions. Their Yes
prices do NOT sum to 1.0 (live sum ~0.11). Every sum correction in
every leg is therefore GATED to the mutually-exclusive regime
0.8 <= cross-market sum <= 1.2; outside it the correction is zero.
Without this gate the ensemble scores 3.73 on real FAA data (vs 0.40
gated, vs 0.55 house baseline) — the sum pull toward 1.0 destroys
every median. FAA markets are also young, so every lookback window is
capped at the available history.

## Validation (local, honest)

- Kit practice (NVIDIA instance, 6 cases): loss 0.2349, median error
  0.0067, 90% coverage 0.90. House baseline 0.3636. (v18 itself: 0.2341.)
- Real FAA data (4 pseudo-cases, checkpoints 2026-11-14 00:00-02:30 UTC,
  CLOB prices-history): loss 0.3954, median error 0.0127, coverage 0.91.
  House baseline 0.5526. Calm-regime cases (02:00, 02:30 UTC): 0.20, 0.13.
- Deterministic: two runs byte-identical. Stdlib only, no network.

## Usage

`python3 main.py` reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else
`/input`) and writes one `*.out` per case to `$MAKEFUN_OUTPUT_DIR`
(else `/output`): `{"forecasts": [{"id", "q"}]}` with `q` = 240 rows of
5 finite numbers in [0, 1], non-decreasing per row.

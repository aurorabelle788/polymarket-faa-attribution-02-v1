"""Polymarket FAA outage attribution forecaster (v1).

Forecasts, for every market of a case, the next 240 minutes of Yes prices
as five quantiles (5%, 25%, 50%, 75%, 95%). Scored by mean pinball loss.
Offline, no network, stdlib only, deterministic.

Model: 5-leg ensemble (weights 0.28 / 0.28 / 0.18 / 0.06 / 0.20), a fresh
reimplementation of the validated NVIDIA-CEO v18 ensemble spec, adapted
for the FAA family:

* The FAA markets are four INDEPENDENT attribution questions (Iran, a
  foreign actor, a hack, an AI sandbox escape). Their Yes prices do NOT
  sum to 1.0 (live sum ~0.11). The NVIDIA ensemble's sum correction
  (pull medians toward a total of 1.0) would badly bias every median
  here, so every sum correction in every leg is GATED to the
  mutually-exclusive regime 0.8 <= cross-market sum <= 1.2 (the same
  gate leg 2B already had). Outside that range the correction is zero
  and the model forecasts each market on its own dynamics.
* FAA markets are young (created 2026-11-13); a final case may hold
  only ~1000 minutes of history. Every lookback window is capped at the
  available length, with safe fallbacks for very short series.

Legs:
  1 (0.28) EWMA-sigma fan, per-price-level half-life, log-log level-fit
     shrink, gated sum correction, spike reversion.
  2 (0.28) Logit ensemble: 0.7 logit-space fan + 0.3 price-space fan.
  3 (0.18) Pooled empirical h-step quantiles (18h window), vol-adaptive
     spread, gated sum drift, net-jump median correction, short-window
     reversion.
  4 (0.06) Gaussian reshape of leg 1.
  5 (0.20) Regime-adaptive sigma fan (26-min half-life EWMA, level-fit
     shrink, regime calming, gated sum correction, fatness adjustment).

Each leg rounds rows to 6 decimals first; the ensemble averages the
rounded rows and rounds once more. Clamp [0, 1], enforce non-decreasing
quantiles.
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZGAUSS = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)

# --- leg 1 constants ---
L1_MULT = (-1.8130133458944, -0.9521483940509615, -0.27,
           0.3914563862643396, 1.8623702217199494)
L1_SHRINK = 0.4
L1_LVL_SHRINK = 0.6
L1_FLOOR = 0.0002
L1_CAP = 0.005  # v18 validated value; a 0.02 cap was tried and REVERTED:
# it helped FAA crash cases slightly (0.395->0.382) but hurt practice
# badly (0.235->0.248, p02 +0.05) — the cap encodes "don't trust a
# high recent sigma", and that holds on FAA too.
L1_GSCALE = 0.87
L1_BETA = 0.47
L1_SUM_K = 1.15
L1_SUM_POW = 0.7

# --- leg 5 constants ---
FAN_MULT = (-2.007, -0.946, -0.403, 0.275, 1.854)
FAN_BANDS = ((30, 1.2399), (120, 1.1139), (10 ** 9, 0.9215))

# --- ensemble weights ---
W = (0.28, 0.28, 0.18, 0.06, 0.20)

# Sum correction applies only when the markets look mutually exclusive
# (cross-market sum near 1). The FAA family's sum is ~0.11: independent
# questions, no sum anchor, so the correction must stay off.
SUM_LO, SUM_HI = 0.8, 1.2


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def r6(v):
    return round(v, 6)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def quantile(sorted_xs, q):
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(sorted_xs)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def clean_rows(rows):
    """Clamp [0,1], enforce non-decreasing quantiles, round 6dp."""
    out = []
    for row in rows:
        prev = 0.0
        clean = []
        for v in row:
            v = clamp(v, 0.0, 1.0)
            if v < prev:
                v = prev
            v = r6(v)
            clean.append(v)
            prev = v
        out.append(clean)
    return out


def returns(prices):
    return [prices[i] - prices[i - 1] for i in range(1, len(prices))]


def ewma_rms(rets, half_life):
    """EWMA-weighted RMS of returns; most recent return weight 1."""
    if not rets:
        return 0.0025
    d = 0.5 ** (1.0 / half_life)
    wsum = 0.0
    vsum = 0.0
    w = 1.0
    for r in reversed(rets):
        vsum += w * r * r
        wsum += w
        w *= d
    return math.sqrt(vsum / wsum) if wsum > 0 else 0.0025


def level_fit_shrink(raw, lasts):
    """Log-log OLS of sigma on x(1-x); returns (level_fit, median)."""
    n = len(raw)
    if n == 0:
        return [], 0.0025
    lx = [math.log(max(1e-6, x * (1.0 - x))) for x in lasts]
    ly = [math.log(max(1e-9, s)) for s in raw]
    mx = mean(lx)
    my = mean(ly)
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    b1 = sxy / sxx if sxx > 0 else 0.0
    b0 = my - b1 * mx
    fit = [math.exp(b0 + b1 * v) for v in lx]
    return fit, median(raw)


def hl_for(x):
    """Per-price-level EWMA half-life (leg 1)."""
    if x < 0.10:
        return 8.0
    if x < 0.20:
        return 10.0
    if x <= 0.30:
        return 15.0
    return 22.0


def pv_mult(x):
    """Price-level sigma multiplier (leg 1)."""
    if x < 0.10:
        return 0.90
    if x <= 0.30:
        return 1.15
    return 0.80


def spike_corr(prices, h):
    """Spike reversion: full give-back of a >=0.03 one-minute move,
    ramping over ~90 min. Returns the correction to add to all quantiles."""
    if len(prices) < 2:
        return 0.0
    d = prices[-1] - prices[-2]
    if abs(d) < 0.03:
        return 0.0
    return -1.0 * d * (1.0 - math.exp(-h / 90.0))


# ---------------------------------------------------------------- leg 1
def leg1(case, lasts, rets_all, raw_all, med_raw):
    n = len(case["markets"])
    fit, _ = level_fit_shrink(raw_all, lasts)
    sigs = []
    plains = []
    for i, x in enumerate(lasts):
        raw = raw_all[i]
        lv = fit[i]
        t = lv * (1.0 - L1_SHRINK) + med_raw * L1_SHRINK
        s = raw * (1.0 - L1_LVL_SHRINK) + t * L1_LVL_SHRINK
        sig = clamp(s, L1_FLOOR, L1_CAP) * L1_GSCALE * pv_mult(x)
        sigs.append(sig)
        plains.append(clamp(raw * 0.6 + med_raw * 0.4, L1_FLOOR, L1_CAP) * L1_GSCALE)
    s_now = sum(lasts)
    shifts = [0.0] * n
    if SUM_LO <= s_now <= SUM_HI:
        gap = (1.0 - s_now) * L1_SUM_K
        tot = sum(plains) or 1.0
        shifts = [gap * p / tot for p in plains]
    out = []
    for i, x in enumerate(lasts):
        prices = case["markets"][i]["prices"]
        rows = []
        for h in range(1, 241):
            band = 1.08 if h <= 30 else 1.05 if h <= 120 else 0.95
            width = (h ** L1_BETA) * band
            ramp = min(1.0, h / 240.0) ** L1_SUM_POW
            c = x + shifts[i] * ramp
            row = [c + L1_MULT[j] * sigs[i] * width for j in range(5)]
            if x < 0.05:
                row[4] = c + L1_MULT[4] * sigs[i] * width * 1.5
            sc = spike_corr(prices, h)
            if sc:
                row = [v + sc for v in row]
            rows.append(row)
        out.append(clean_rows(rows))
    return out


# ---------------------------------------------------------------- leg 2
def _logit(p):
    p = clamp(p, 0.0005, 0.9995)
    return math.log(p / (1.0 - p))


def _inv_logit(v):
    if v > 30:
        return 1.0
    if v < -30:
        return 0.0
    e = math.exp(v)
    return e / (1.0 + e)


def leg2(case, lasts):
    n = len(case["markets"])
    s_now = sum(lasts)
    out = []
    for i, x in enumerate(lasts):
        prices = case["markets"][i]["prices"]
        # --- Leg A (logit space) ---
        lp = [_logit(p) for p in prices]
        lmoves = [lp[k] - lp[k - 1] for k in range(1, len(lp))]

        def robust_sigma(moves):
            if len(moves) < 10:
                return 0.0025
            am = [abs(v) for v in moves]
            sig = mean(am) / 0.7978845608028654
            med = median(am) / 0.6744897501960817
            if med > 0:
                sig = min(sig, med)
            return clamp(sig, 0.002, 0.10)

        sig_a = 0.5 * robust_sigma(lmoves[-1440:]) + 0.5 * robust_sigma(lmoves[-60:])
        big = x >= 0.10
        lv_a = (-2.0, -1.0, 0.0, 0.4, 1.30) if big else (-1.80, -0.85, 0.0, 0.5, 1.50)
        r60 = x - prices[-61] if len(prices) >= 61 else 0.0
        # --- Leg B (price space) ---
        moves = returns(prices)
        w = moves[-720:]
        w = [clamp(v, -0.05, 0.05) for v in w]
        sig_b = clamp(mean([abs(v) for v in w]) * 0.75, 0.0002, 0.01) if w else 0.0025
        gated = SUM_LO <= s_now <= SUM_HI
        rows = []
        for h in range(1, 241):
            # median in price space (rise given back, falls not continued)
            med_p = x - (1.0 - math.exp(-h / 120.0)) * r60 if r60 > 0 else x
            lm = _logit(med_p)
            row_a = []
            for j in range(5):
                pw = 0.45 if j in (0, 4) else 0.40
                widen = 1.30 if j in (0, 4) else 1.0
                row_a.append(_inv_logit(lm + lv_a[j] * sig_a * (h ** pw) * widen))
            # leg B median pulled toward x/S when gated
            med_b = x
            if gated and s_now > 0:
                med_b = x + (x / s_now - x) * (1.0 - math.exp(-h / 120.0))
            lv_b = (-2.0, -0.5, 0.0, 0.5, 2.0)
            row_b = [med_b + lv_b[j] * sig_b * math.sqrt(h) for j in range(5)]
            row = [0.7 * a + 0.3 * b for a, b in zip(row_a, row_b)]
            rows.append(row)
        out.append(clean_rows(rows))
    return out


# ---------------------------------------------------------------- leg 3
GRID_H = (1, 2, 3, 5, 7, 11, 16, 24, 36, 54, 81, 121, 180, 240)


def _interp_table(table, h):
    """Log-interpolate pooled quantiles at horizon h. table: {grid_h: [5]}."""
    if h in table:
        return table[h]
    lo = max([g for g in GRID_H if g < h], default=None)
    hi = min([g for g in GRID_H if g > h], default=None)
    if lo is None:
        return table[hi]
    if hi is None:
        return table[lo]
    f = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return [table[lo][j] * (1 - f) + table[hi][j] * f for j in range(5)]


def leg3(case, lasts):
    n = len(case["markets"])
    s_now = sum(lasts)
    # pooled empirical h-step change quantiles over last 1080 min
    table = {}
    for gh in GRID_H:
        diffs = []
        for m in case["markets"]:
            pr = m["prices"]
            w = pr[-1080:] if len(pr) > 1080 else pr
            for k in range(len(w) - gh):
                diffs.append(w[k + gh] - w[k])
        diffs.sort()
        table[gh] = [quantile(diffs, q) for q in TAUS]
    # vol-adaptive spread
    pool = []
    for m in case["markets"]:
        pr = m["prices"]
        w = returns(pr[-720:]) if len(pr) > 1 else []
        pool.extend(clamp(v, -0.02, 0.02) for v in w)
    pvol = stdev(pool) or 1e-6
    out = []
    for i, x in enumerate(lasts):
        prices = case["markets"][i]["prices"]
        mv = stdev(returns(prices[-120:])) if len(prices) > 2 else pvol
        vf = clamp((mv / pvol) ** 0.65, 0.4, 2.0) if pvol > 0 else 1.0
        # net-jump median correction (last 60 min, |move| >= 0.05)
        jn = 0.0
        for v in returns(prices[-60:]):
            if abs(v) >= 0.05:
                jn += v
        mean30 = mean(prices[-30:])
        dev = x - mean30
        rows = []
        for h in range(1, 241):
            q = _interp_table(table, h)
            drift = 0.0
            if SUM_LO <= s_now <= SUM_HI and s_now > 0:
                drift = (x / s_now - x) * (h / 240.0)
            row = [
                x + q[0] * 0.90 * vf + drift,
                x + q[1] * math.sqrt(0.85) * vf + drift,
                x + drift,
                x + q[3] * math.sqrt(0.85) * vf + drift,
                x + q[4] * 0.80 * vf + drift,
            ]
            if jn > 0:
                corr = -0.5 * jn * (1.0 - math.exp(-h / 15.0))
                row = [v + corr for v in row]
            elif jn < 0:
                corr = 0.3 * jn * (1.0 - math.exp(-h / 30.0))
                row = [v + corr for v in row]
            if dev > 0:
                corr = -0.5 * dev * (1.0 - math.exp(-h / 60.0))
                row = [v + corr for v in row]
            rows.append(row)
        out.append(clean_rows(rows))
    return out


# ---------------------------------------------------------------- leg 4
def leg4(leg1_rows):
    out = []
    span = L1_MULT[4] - L1_MULT[0]
    for rows in leg1_rows:
        g_rows = []
        for row in rows:
            w = (row[4] - row[0]) / span if span else 0.0
            c = row[2] - L1_MULT[2] * w
            g = [c + ZGAUSS[j] * w * (1.30 if j in (0, 4) else 1.0) for j in range(5)]
            g_rows.append(g)
        out.append(clean_rows(g_rows))
    return out


# ---------------------------------------------------------------- leg 5
def leg5(case, lasts, rets_all, raw_all, med_raw):
    n = len(case["markets"])
    fit, _ = level_fit_shrink(raw_all, lasts)
    sigs = []
    plains = []
    acts = []
    fat = []
    ratios = []
    for i, x in enumerate(lasts):
        raw = raw_all[i]
        lv = fit[i]
        s = 0.4 * raw + 0.6 * (0.6 * lv + 0.4 * med_raw)
        sigs.append(clamp(s, 0.0001, 0.005) * 0.9)
        plains.append(clamp(raw * 0.6 + med_raw * 0.4, 0.0001, 0.005) * 0.9)
        rets = rets_all[i][-120:]
        acts.append(max(0.05, sum(1 for v in rets if v != 0) / 120.0))
        mabs = mean([abs(v) for v in rets]) if rets else 1e-6
        ratios.append((raw / mabs) if mabs > 1e-9 else 1.0)
    med_ratio = median(ratios) or 1.0
    for r in ratios:
        fat.append(clamp(((r / med_ratio) ** 0.6), 0.8, 1.6))
    R = max(med_raw, 1e-9)
    width_factor = clamp((R / 0.0003) ** -1.8, 0.6, 1.6)
    sum_strength = clamp((R / 0.0003) ** -1.0, 0.1, 3.0)
    s_now = sum(lasts)
    shifts = [0.0] * n
    if SUM_LO <= s_now <= SUM_HI:
        gap = (1.0 - s_now) * 0.84 * sum_strength
        weights = [plains[i] * (acts[i] ** 0.65) for i in range(n)]
        tot = sum(weights) or 1.0
        shifts = [gap * w / tot for w in weights]
    out = []
    for i, x in enumerate(lasts):
        prices = case["markets"][i]["prices"]
        rows = []
        for h in range(1, 241):
            band = 1.2399 if h <= 30 else 1.1139 if h <= 120 else 0.9215
            width = (h ** 0.46) * width_factor * band
            ramp = (h / 240.0) ** 0.46
            c = x + shifts[i] * ramp
            row = [c + FAN_MULT[j] * sigs[i] * width for j in range(5)]
            row[0] *= 1.0  # fatness applies to outer quantiles only
            row[0] = c + FAN_MULT[0] * sigs[i] * width * fat[i]
            row[4] = c + FAN_MULT[4] * sigs[i] * width * fat[i]
            sc = spike_corr(prices, h)
            if sc:
                row = [v + sc for v in row]
            rows.append(row)
        out.append(clean_rows(rows))
    return out


# ---------------------------------------------------------------- driver
def forecast_case(case):
    markets = case["markets"]
    lasts = [m["prices"][-1] for m in markets]
    rets_all = []
    raw_all = []
    for i, m in enumerate(markets):
        pr = m["prices"]
        rets = returns(pr)
        rets_all.append(rets)
        raw_all.append(ewma_rms(rets[-120:], hl_for(lasts[i])))
    med_raw = median(raw_all) if raw_all else 0.0025

    l1 = leg1(case, lasts, rets_all, raw_all, med_raw)
    l2 = leg2(case, lasts)
    l3 = leg3(case, lasts)
    l4 = leg4(l1)
    l5 = leg5(case, lasts, rets_all, raw_all, med_raw)

    legs = (l1, l2, l3, l4, l5)
    forecasts = []
    for i, m in enumerate(markets):
        rows = []
        for h in range(240):
            row = [0.0] * 5
            for li in range(5):
                w = W[li]
                for j in range(5):
                    row[j] += w * legs[li][i][h][j]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": clean_rows(rows)})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(INPUT_DIR, name)) as f:
            case = json.load(f)
        out = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()

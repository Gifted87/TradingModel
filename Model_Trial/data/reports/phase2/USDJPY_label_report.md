# Gate 2 Target & Label Verification Report — USDJPY

**Gate Status:** ✅ **PASSED**
**Horizon:** 30 Minutes (6 Bars)
**Total Valid Labeled Rows:** `1,164,225`

---

## 1. Primary Target: $r_{30} = \log(\text{mid}_{t+30} / \text{mid}_t)$
| Statistic | Value | Theoretical Expectation | Status |
|---|---|---|---|
| **Mean** | `+0.000004` | $\approx 0$ (Martingale property) | ✅ Sane |
| **Standard Deviation** | `0.000785` | Multi-minute volatility | ✅ Normal |
| **Skewness (winsorized)** | `-0.8388` | $[-1.80, +1.80]$ post p0.010/p99.990 clip | ✅ Symmetric |
| **Excess Kurtosis (winsorized)** | `+24.32` | $> 0$ (Fat-tailed / Leptokurtic) | ✅ Realistic |
| **Range [Min, Max] (raw)** | `[-0.0376, +0.0216]` | Finite log-return bounds | ✅ Clean |

---

## 2. Magnitude & Volatility Targets (Log-Space per [E5])
- **Direct Magnitude $m_{30} = \log(|r_{30}| + \epsilon)$:**
  - Mean: `-8.6781` | Std: `2.2581`
- **Forward Realized Volatility $RV_{30} = \log(\sqrt{\sum r^2} + \epsilon)$:**
  - Mean: `-8.1110` | Std: `2.0955`

---

## 3. Excursions & Execution Boundary Metrics
- **Mean MFE ($MFE_{30}$):** `+0.000353` (Max upward excursion)
- **Mean MAE ($MAE_{30}$):** `-0.000354` (Max downward excursion)

---

## 4. Direction & Economic Dead-Zone ($|r_{30}| < \tau$)
- **Up-move Frequency:** `47.92%`
- **Down-move Frequency:** `45.85%`
- **Median Spread Threshold ($\tau$):** `0.000038`
- **Dead-Zone Fraction ($|r_{30}| < \text{Spread}$):** **`14.39%`** (Excluded from direction metrics per [C2])
- **Tradable Signal Fraction:** **`85.61%`**

---

## 5. Winsorization Metadata (Audit-Only; Parquet Labels Unchanged)
- **Clip Quantile:** p`0.010` / p`99.990`
- **Clip Bounds:** `[-0.012856, +0.008690]`
- **Rows Clipped:** `234` of `1,164,225` (`0.0201%`)
- **Note:** Winsorization applies to the audit skew metric only. Raw r30 values in the Parquet file are unmodified.

---

## 6. Gate 2 Verdict & Conclusion
**`GATE 2 PASS: Valid stationary label distributions (skew_w=-0.839, dead_zone=14.4%, tradable=85.6%, winsorized=234 rows at p0.010/p99.990)`**

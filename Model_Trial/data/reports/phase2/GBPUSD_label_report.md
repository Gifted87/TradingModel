# Gate 2 Target & Label Verification Report — GBPUSD

**Gate Status:** ✅ **PASSED**
**Horizon:** 30 Minutes (6 Bars)
**Total Valid Labeled Rows:** `1,156,684`

---

## 1. Primary Target: $r_{30} = \log(\text{mid}_{t+30} / \text{mid}_t)$
| Statistic | Value | Theoretical Expectation | Status |
|---|---|---|---|
| **Mean** | `+0.000000` | $\approx 0$ (Martingale property) | ✅ Sane |
| **Standard Deviation** | `0.000761` | Multi-minute volatility | ✅ Normal |
| **Skewness (winsorized)** | `+0.0039` | $[-1.80, +1.80]$ post p0.010/p99.990 clip | ✅ Symmetric |
| **Excess Kurtosis (winsorized)** | `+16.41` | $> 0$ (Fat-tailed / Leptokurtic) | ✅ Realistic |
| **Range [Min, Max] (raw)** | `[-0.0549, +0.0374]` | Finite log-return bounds | ✅ Clean |

---

## 2. Magnitude & Volatility Targets (Log-Space per [E5])
- **Direct Magnitude $m_{30} = \log(|r_{30}| + \epsilon)$:**
  - Mean: `-8.7202` | Std: `2.2545`
- **Forward Realized Volatility $RV_{30} = \log(\sqrt{\sum r^2} + \epsilon)$:**
  - Mean: `-8.1414` | Std: `2.1018`

---

## 3. Excursions & Execution Boundary Metrics
- **Mean MFE ($MFE_{30}$):** `+0.000344` (Max upward excursion)
- **Mean MAE ($MAE_{30}$):** `-0.000344` (Max downward excursion)

---

## 4. Direction & Economic Dead-Zone ($|r_{30}| < \tau$)
- **Up-move Frequency:** `47.04%`
- **Down-move Frequency:** `46.77%`
- **Median Spread Threshold ($\tau$):** `0.000071`
- **Dead-Zone Fraction ($|r_{30}| < \text{Spread}$):** **`21.16%`** (Excluded from direction metrics per [C2])
- **Tradable Signal Fraction:** **`78.84%`**

---

## 5. Winsorization Metadata (Audit-Only; Parquet Labels Unchanged)
- **Clip Quantile:** p`0.010` / p`99.990`
- **Clip Bounds:** `[-0.009521, +0.008700]`
- **Rows Clipped:** `232` of `1,156,684` (`0.0201%`)
- **Note:** Winsorization applies to the audit skew metric only. Raw r30 values in the Parquet file are unmodified.

---

## 6. Gate 2 Verdict & Conclusion
**`GATE 2 PASS: Valid stationary label distributions (skew_w=+0.004, dead_zone=21.2%, tradable=78.8%, winsorized=232 rows at p0.010/p99.990)`**

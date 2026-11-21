# Gate 1 Quality Audit Report — GBPUSD

**Audit Status:** ✅ **PASSED**
**Audit Timestamp:** 2026-08-23 21:55:13 UTC

---

## 1. Dataset Dimensions
| Metric | Value |
|---|---|
| **Symbol** | `GBPUSD` |
| **Total Bars** | `1,160,916` |
| **Start Date** | `2012-01-11 01:35:00` |
| **End Date** | `2026-08-23 21:25:00` |

---

## 2. Integrity & Anomaly Summary
| Test Category | Detected Issues | Rate (%) | Status |
|---|---|---|---|
| **Intra-Session Trading Gaps (>5m)** | 921 | 0.0793% | ✅ Clean |
| **Bad-Print Spikes (>10x ATR)** | 13 | 0.0011% | ✅ Clean |
| **Inverted / Negative Spreads** | 0 | 0.0000% | ✅ Zero |
| **Spread Blowouts (>50x Median)** | 0 | 0.0000% | Info |
| **Zero Volume Bars** | 68,447 | 5.8959% | Info |

---

## 3. Spread Statistics
- **Median Spread:** `0.00010`
- **Maximum Recorded Spread:** `0.00400`

---

## 4. Gate 1 Evaluation & Kill Criteria
- **Total Anomaly Count:** `934`
- **Composite Anomaly Ratio:** **`0.0805%`** (Kill Threshold: `< 0.50%`)
- **Conclusion:** **GATE 1 PASS: 0.080% anomalous bars (< 0.5% threshold)**

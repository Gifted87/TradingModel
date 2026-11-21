# Gate 1 Quality Audit Report — XAUUSD

**Audit Status:** ✅ **PASSED**
**Audit Timestamp:** 2026-08-23 21:55:19 UTC

---

## 1. Dataset Dimensions
| Metric | Value |
|---|---|
| **Symbol** | `XAUUSD` |
| **Total Bars** | `1,680,380` |
| **Start Date** | `2003-05-05 00:00:00` |
| **End Date** | `2026-08-21 20:55:00` |

---

## 2. Integrity & Anomaly Summary
| Test Category | Detected Issues | Rate (%) | Status |
|---|---|---|---|
| **Intra-Session Trading Gaps (>5m)** | 5,220 | 0.3106% | ⚠️ High Gaps |
| **Bad-Print Spikes (>10x ATR)** | 105 | 0.0062% | ✅ Clean |
| **Inverted / Negative Spreads** | 0 | 0.0000% | ✅ Zero |
| **Spread Blowouts (>50x Median)** | 0 | 0.0000% | Info |
| **Zero Volume Bars** | 35,248 | 2.0976% | Info |

---

## 3. Spread Statistics
- **Median Spread:** `0.38000`
- **Maximum Recorded Spread:** `14.05400`

---

## 4. Gate 1 Evaluation & Kill Criteria
- **Total Anomaly Count:** `5,325`
- **Composite Anomaly Ratio:** **`0.3169%`** (Kill Threshold: `< 0.50%`)
- **Conclusion:** **GATE 1 PASS: 0.317% anomalous bars (< 0.5% threshold)**

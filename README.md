# Gold Cause-and-Effect Model

This project predicts the next 5-minute XAU/USD closing-price change from a configurable sequence of preceding 1-minute closing-price changes.

~~~text
Cause:  1-minute close changes available before the target candle
Effect: next 5-minute close minus the previous 5-minute close
~~~

For a target candle closing at 10:45, the default 60-minute configuration uses the 1-minute changes from 9:40 through 10:40. The target is the 10:45 5-minute close change. The cause window ends before the target candle begins, preventing future information from entering the input.

## Current status

The current checkpoint uses:

- Dukascopy XAU/USD spot-gold data.
- 5,941,407 one-minute bars from January 2010 through August 19, 2026.
- 1,104,407 valid cause/effect training examples.
- A two-layer Transformer with an auxiliary continuous-return head.
- Nine movement ranges derived from historical returns.
- Chronological training, validation, and unseen test periods.

The final model is a research candidate, not an automatic live-trading system. It must pass spread-aware, walk-forward, and independent-feed validation before deployment.

## Observed checkpoint performance

The current checkpoint was trained on 1,104,407 cause/effect examples. The final 15 percent of the time-ordered dataset was held out for evaluation.

Model results:

| Measure | Result |
| --- | ---: |
| Best chronological validation accuracy | 79.89% |
| Unseen test range accuracy | 58.76% |
| Unseen test top-3 range accuracy | 87.68% |
| Unseen test directional accuracy | 86.94% |
| Unseen test continuous-change MAE | 1.35 price units |
| Simple cause-sum directional baseline | 58.91% |
| Shuffled-input placebo directional accuracy | 50.00% |
| Target shifted backward one 5-minute bar | 51.65% |
| Target shifted forward one 5-minute bar | 50.95% |

Unseen directional accuracy by year was 94.48% in 2024, 88.65% in 2025, and 74.52% in 2026. The decline in the most recent period indicates regime sensitivity and is one reason rolling retraining is required.

The cost-aware exploratory backtest used 165,662 unseen examples and an assumed round-trip cost of 0.50 gold price units. At an expected-edge threshold of 0.50, it produced 83,128 trades, 109,626.73 aggregate price units, and 97.38% pre-cost directional accuracy. These figures are not live-profit claims: the cost is assumed rather than reconstructed from historical bid/ask quotes, and the result still requires walk-forward and independent-feed validation.

## Installation

~~~powershell
python -m pip install -r requirements.txt
~~~

Python 3.12 or newer is recommended. PyTorch is required for the advanced model.

## Configuration

The main configuration is configs/gold.yaml.

~~~yaml
symbol: XAUUSD
lookback_minutes: 60
gap_minutes: 5
label_quantiles: [0.0, 0.05, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85, 0.95, 1.0]
~~~

lookback_minutes controls the number of 1-minute changes in each cause window. It can be increased to 3600, but larger windows require substantially more memory and training time. Movement ranges come from gold's observed distribution; there is no arbitrary +20 or -20 limit.

## Data workflow

### Download 1-minute data

The downloader caches each calendar month as Parquet and resumes after interruption:

~~~powershell
python src/download_dukascopy.py --config configs/gold.yaml --timeframe 1m --start 2010-01-01
~~~

Combined output:

~~~text
data/raw/XAUUSD_DUKASCOPY_1m.parquet
~~~

Dukascopy provides spot gold, not COMEX futures. Do not mix this feed with GC=F during training or evaluation.

### Build the cause/effect dataset

~~~powershell
python src/build_cause_effect_dataset.py --config configs/gold.yaml
~~~

This loads the 1-minute close series, computes 1-minute changes, derives 5-minute closes from the same source, computes each effect, builds the preceding cause window, and assigns a data-derived movement range.

Outputs:

~~~text
data/processed/gold_1m_cause_5m_effect.parquet
data/processed/gold_1m_cause_5m_effect.bin_edges.npy
~~~

## Model training

The Transformer consumes the 1-minute cause sequence and produces:

- A probability distribution over movement ranges.
- A continuous estimate of the 5-minute price change.

Train chronologically, then refit the selected epoch count on every valid example:

~~~powershell
python src/train_advanced.py --config configs/gold.yaml --epochs 3
~~~

Generated artifacts:

~~~text
data/models/gold_transformer_full.pt
data/models/gold_transformer_preprocessing.npz
data/models/gold_transformer_metrics.json
data/models/gold_transformer_oos.pt
data/models/gold_transformer_oos_preprocessing.npz
~~~

The oos checkpoint is trained before the final test period. The full checkpoint is refit on the complete dataset.

## Verification

~~~powershell
python src/audit_transformer.py --config configs/gold.yaml
~~~

The audit measures overall and top-3 accuracy, yearly performance, a simple cause-sum baseline, shuffled-input placebo performance, forward/backward target shifts, and directional accuracy at multiple confidence thresholds.

Shuffled inputs and shifted targets should fall close to chance. Otherwise investigate alignment and leakage.

## Backtesting

~~~powershell
python src/backtest_transformer.py --config configs/gold.yaml --cost 0.50
~~~

The cost argument is assumed round-trip spread plus slippage in gold price units. It is not historical spread reconstruction. A production backtest should use Dukascopy bid and ask data, session-specific spreads, execution delay, and position sizing.

Report:

~~~text
data/reports/gold_transformer_backtest.json
~~~

## Offline RL policy

The RL layer treats the frozen Transformer as the market-state predictor. It receives the range probabilities, expected return, confidence, and cause-window statistics, then chooses one of seven actions: no trade, long at three sizes, or short at three sizes.

Train the offline contextual-bandit policy with:

~~~powershell
python src/train_rl_policy.py --config configs/gold.yaml --epochs 8 --cost 0.50
~~~

The reward is net price movement after the configured round-trip cost. The policy is trained chronologically and evaluated on the final unseen period. Outputs are written to data/models/gold_rl_policy.pt, data/models/gold_rl_preprocessing.npz, and data/models/gold_rl_metrics.json.

## Range prediction and guessing

The model produces a probability for every movement range. The guessing policy ranks those probabilities:

1. Highest probability is the first candidate.
2. Lower probabilities are subsequent candidates.
3. A confidence or expected-edge rule can reject all candidates and produce no trade.

Weights must not update during live prediction because the future effect is unknown. Weight updates happen during historical training after the true effect is available.

## Limitations

- Dukascopy is spot gold, not COMEX GC=F.
- Fixed costs do not replace historical bid/ask execution data.
- One held-out period does not establish live profitability.
- Quantile boundaries and preprocessing should be fitted on training data only for strict evaluation.
- The next validation stage is rolling walk-forward retraining across multiple regimes.
- Large datasets and model binaries are ignored by Git; the downloader and pipeline are versioned.

## Project layout

~~~text
configs/
  gold.yaml
src/
  download_dukascopy.py
  build_cause_effect_dataset.py
  train_advanced.py
  audit_transformer.py
  backtest_transformer.py
data/
  raw/
  processed/
  models/
  reports/
~~~

Current Git checkpoint: 8ed5740 (Checkpoint Dukascopy 1m cause 5m effect pipeline).

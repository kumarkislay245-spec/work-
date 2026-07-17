# Write-up: Trader Behavior vs. Market Sentiment

## Methodology

1. **Data & join.** Trade-level execution data (74,503 rows, Hyperliquid) was
   joined to the daily Bitcoin Fear & Greed Index on calendar date
   (`Timestamp IST` → `daily_date` ↔ `date`).
2. **Feature engineering.** For every `(Account, day)` pair we computed:
   Daily PnL (`Closed PnL` summed), win rate (% of trades with positive
   `Closed PnL`), average trade size (USD), number of trades, and the
   long/short ratio (share of `BUY` vs `SELL` orders). Each day was tagged
   with that day's Fear & Greed classification (Extreme Fear → Extreme
   Greed).
3. **Segmentation.**
   - **Trade Frequency Segment** — traders split into *Frequent* /
     *Infrequent* at the median of their own average daily trade count
     (median ≈ 44 trades/day).
   - **Performance Segment** — traders split into *Consistent Winner* /
     *Inconsistent Trader* based on whether their average daily PnL **and**
     average win rate both exceeded the cross-trader medians (median PnL ≈
     $6,106/day, median win rate ≈ 32.6%).
4. **Predictive model.** We labeled each trader-day with
   `next_day_profitable` (1 if that trader's PnL the *following* day was
   positive) and trained an XGBoost classifier on same-day features
   (PnL, win rate, trade size, trade count, long/short ratio, sentiment
   classification, frequency segment, performance segment), one-hot
   encoding the categorical features. Data was split 80/20 (stratified).
   We report a baseline model and one tuned via `RandomizedSearchCV`
   (50 candidates, 5-fold CV, optimizing ROC AUC).

## Insights

### Insight 1 — Fear days are the most profitable, not Greed days
| Sentiment | Avg Daily PnL (USD) | Avg Win Rate |
|---|---|---|
| **Extreme Fear** | **$11,569** | 33.6% |
| Fear | $7,791 | 35.9% |
| Extreme Greed | $6,681 | 40.3% |
| Neutral | $6,549 | 34.3% |
| Greed | $4,446 | 31.4% |

Average daily PnL is highest during Extreme Fear and lowest during Greed —
the opposite of what "buy the dip, sell the euphoria" retail intuition might
suggest for *win rate* (which actually peaks in Extreme Greed). Traders also
trade far more often in Extreme Fear (~120 trades/day) than in Extreme Greed
(~43 trades/day), and lean longer (60% long ratio) during Extreme Fear vs.
roughly balanced during Greed — consistent with contrarian/dip-buying
behavior that happened to pay off in this sample.

![Insight 1a](outputs/figures/insight1_pnl_by_sentiment.png)
![Insight 1b](outputs/figures/insight1_winrate_by_sentiment.png)

### Insight 2 — Frequent traders outperform on every axis
| Segment | Avg Daily PnL | Avg Win Rate | Avg # Trades/day | Avg Trade Size |
|---|---|---|---|---|
| **Frequent** | **$10,228** | **42.1%** | 114.0 | $7,179 |
| Infrequent | $2,850 | 28.4% | 25.6 | $11,046 |

Traders who trade above the median daily frequency earn ~3.6x the average
daily PnL and a 14-point higher win rate than infrequent traders — despite
placing smaller average trades. This looks like a skill/consistency signal
rather than "more trades = more luck": frequent traders are also more
accurate per trade.

![Insight 2](outputs/figures/insight2_frequency_segment.png)

### Insight 3 — "Consistent Winners" trade bigger and more often
| Segment | Avg Daily PnL | Avg Win Rate | Avg # Trades/day | Avg Trade Size |
|---|---|---|---|---|
| **Consistent Winner** | **$38,767** | 38.3% | 292.3 | **$18,032** |
| Inconsistent Trader | $4,669 | 34.9% | 56.5 | $8,655 |

Traders who beat the median on *both* PnL and win rate (only 3 of 13
accounts in this sample) are defined by a combination of high trade
frequency **and** larger position sizes, not just one or the other.

![Insight 3](outputs/figures/insight3_performance_segment.png)

### Predictive model — next-day profitability

| Model | ROC AUC | Accuracy |
|---|---|---|
| XGBoost (default params) | 0.670 | 0.61 |
| XGBoost (tuned, RandomizedSearchCV) | **0.683** | 0.63 |

The model has modest but real predictive power (AUC ≈ 0.68, well above the
0.50 coin-flip baseline) — same-day trading behavior and market sentiment do
carry *some* signal about whether a trader stays profitable the next day,
but the majority of next-day outcome variance is not explained by these
features alone. Feature importance and the confusion matrix/ROC curve are
in `outputs/figures/`.

![Confusion Matrix (tuned)](outputs/figures/model_confusion_matrix_tuned.png)
![ROC Curve (tuned)](outputs/figures/model_roc_curve_tuned.png)

## Strategy recommendations

1. **Lean into Extreme Fear periods — with sizing discipline.** Extreme Fear
   days show the highest average daily PnL in this sample, and pairing that
   with the Insight 2 finding (frequent trading correlates with higher PnL
   and win rate) suggests selectively *increasing* trade frequency during
   Extreme Fear rather than sitting out. Because "extreme fear" also implies
   elevated volatility and tail risk, this should be paired with strict
   per-trade risk limits (e.g., capped position size as % of capital, hard
   stop-losses) rather than simply scaling up blindly.

2. **Emulate "Consistent Winner" behavior: higher frequency + larger,
   calculated position sizes.** The Consistent Winner segment combines both
   frequent trading *and* above-average trade size — not one or the other.
   Traders aiming to move from "Inconsistent" toward "Consistent Winner"
   should scale up size gradually only once win rate and trade cadence are
   already above the trader's own historical median, to avoid over-sizing
   into a losing streak.

3. **Use sentiment + behavior as a soft signal, not a standalone trading
   rule.** The predictive model's AUC (~0.68) shows sentiment/behavior
   features have real but limited standalone predictive power for next-day
   profitability. These insights are best used as one input into a broader
   risk-management framework (e.g., adjusting position sizing bands by
   sentiment regime) rather than as a sole entry/exit trigger.

## Caveats

- Sample is small at the trader level (13 unique accounts), so segment-level
  averages (especially "Consistent Winner", n=3) are sensitive to individual
  outliers.
- Fear & Greed Index is a market-wide sentiment measure; it is applied
  uniformly to all traders/coins on a given day and does not capture
  asset-specific sentiment.
- Results describe **historical association**, not causation — profitable
  days coinciding with Extreme Fear does not by itself prove that trading
  more during Extreme Fear *causes* higher PnL.

# Stock Prediction, Strategy & Backtest System

An end-to-end system built around the Jane Street Real-Time Market Data
Forecasting notebook (`jane-street-baseline-lgb-xgb-and-catboost.ipynb`). The
notebook only trained a model and scored weighted R². This system adds the two
missing pieces — a **trading strategy** and a **walk-forward backtest** — and
ties everything into one reproducible pipeline whose objective is to **maximize
realized return**.

```
predict (responder_6)  ->  strategy (positions)  ->  backtest (PnL, risk)
```

## Layout

| File | Role |
|------|------|
| `config.py` | All paths, schema, and tunable parameters in one place |
| `src/data.py` | Loads real `train.parquet` if present; else generates a Jane-Street-like synthetic panel with real (noisy) signal |
| `src/features.py` | Feature matrix extraction, NaN fill, optional cross-sectional rank normalization |
| `src/model.py` | `EnsembleModel`: purged date-based CV over LGB (+XGB/CatBoost if installed), weighted-R² scoring, save/load |
| `src/strategy.py` | Maps predictions to positions: `threshold`, `tanh`, or `cross_sectional` (dollar-neutral) |
| `src/backtest.py` | Walk-forward PnL engine + metrics (Sharpe, Sortino, Calmar, max drawdown, hit rate, turnover) + equity-curve plot |
| `run_pipeline.py` | CLI that runs the whole thing and writes outputs |

## Quick start

```bash
pip install -r requirements.txt        # lightgbm + sklearn are the minimum
python run_pipeline.py --compare       # train, then backtest all strategies
```

With no Kaggle data present it runs on synthetic data automatically, so it works
out of the box. Point it at the real dataset by setting `JS_DATA_DIR`:

```bash
JS_DATA_DIR=/path/to/jane-street-real-time-market-data-forecasting \
  python run_pipeline.py --strategy cross_sectional
```

### Common commands

```bash
python run_pipeline.py                    # default strategy (cross_sectional)
python run_pipeline.py --strategy tanh    # smooth sizing
python run_pipeline.py --no-train         # reuse models/ensemble.joblib
python run_pipeline.py --cost 0.0005      # heavier transaction cost
```

## How it works

**Prediction.** `responder_6` is a clipped forward return. The ensemble trains
one set of models per fold, where fold `f` trains on dates with
`date_index % N_FOLD != f` and validates on the rest — disjoint by `date_id`,
so there is no intra-day leakage. Final prediction is the mean across all
fitted models. Scored with the competition's weighted R²
`1 - sum(w·(y-ŷ)²)/sum(w·y²)`.

**Strategy.** A larger predicted return should mean a larger position:
- `threshold` — go ±1 only when the signal clears `PRED_THRESHOLD`.
- `tanh` — smooth sizing `tanh(scale·pred)`, clipped to [-1, 1].
- `cross_sectional` — demean predictions within each `(date_id, time_id)` and
  size proportional to the demeaned value, normalized to fixed gross exposure.
  This is a market-neutral long/short book, the natural fit for a per-timestamp
  relative-value signal.

**Backtest.** Strictly walk-forward on the last `NUM_VALID_DATES` dates, which
the model never saw in training. A position earns `pos · responder_6`;
transaction cost is charged on per-symbol turnover between steps. PnL is
aggregated per timestamp into an equity curve, then summarized.

## Outputs (`outputs/`)

- `equity_<strategy>.png` — equity curve + drawdown
- `predictions.csv` — per-row prediction, position, and net PnL for the best strategy
- `report.json` — data source, parameters, and full metrics per strategy

## Notes & caveats

- Synthetic results demonstrate the machinery, not a real edge — the generator
  injects a known low signal-to-noise factor structure. Numbers are only
  meaningful on the real dataset.
- This is research/backtest tooling, not investment advice or a live trading
  system. Backtested performance does not guarantee future results.
- The original notebook is preserved unchanged for the Kaggle inference server;
  this system reuses its modeling idea but is independent of the Kaggle runtime.

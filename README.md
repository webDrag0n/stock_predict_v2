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
| `src/data/` | Data connectors (tushare) + factor engineering + panel loaders |
| `src/features.py` | Feature matrix extraction, NaN fill (Jane-Street row format) |
| `src/walkforward.py` | Rolling walk-forward engine + model objective dispatch |
| `src/ensemble.py` | `IncrementalEnsemble`: daily continued-training + rolling-IC weighting |
| `src/strategy.py` | Maps predictions to positions: `long_topk`, `tanh`, or `cross_sectional` (dollar-neutral) |
| `src/backtest.py` | Periodic-rebalance PnL engine + metrics (Sharpe, Sortino, Calmar, max drawdown, hit rate) |
| `src/backtest_daily.py` | Daily-replay engine: per-day decisions + next-day fill simulation |
| `predict_tomorrow.py` | CLI: train on data up to today, emit next-day executable orders |
| `run_webapp.py` | Dash front-end (session-based pipelines) |

> Removed in the phase-5 architecture cleanup: `src/model.py` (leaky purged-CV
> ensemble) and its CLI `run_pipeline.py`; the one-day-lookahead event engine
> `src/backtest_event.py`; and the inflated daily-overlap `run_backtest` in
> `src/backtest.py` (`run_backtest_periodic` is the only backtest path).

## Quick start

```bash
uv sync --extra dev                     # 安装依赖（含 pytest；torch 见下）
uv run python scripts/run_walkforward.py   # rolling walk-forward OOS backtest
uv run python predict_tomorrow.py          # next-day orders from current holdings
uv run pytest tests/ -v                    # test suite
```

Dependencies are declared in `pyproject.toml` (managed by uv; `uv sync` creates
`.venv` automatically). Optional extras: `--extra attn` for the attention model
leg (torch), `--extra full` for XGBoost/CatBoost, `--extra ui` for the webapp.

## How it works

**Prediction.** `responder_6` is a clipped forward return. Models are trained on
the named factor panel: the rolling walk-forward engine (`src/walkforward.py`,
research/validation) and the incremental multi-model ensemble
(`src/ensemble.py`, production daily path) share the same per-date
cross-sectional normalization and GBM hyperparameters (single source:
`walkforward.GBM_PARAMS`).

**Strategy.** A larger predicted return should mean a larger position:
- `long_topk` — long the top-K% predicted names, equal weight (the only
  practically tradeable strategy for A-share retail; the default).
- `tanh` — smooth sizing `tanh(scale·pred)`, clipped to [-1, 1].
- `cross_sectional` — demean predictions within each `(date_id, time_id)` and
  size proportional to the demeaned value, normalized to fixed gross exposure.
  This is a market-neutral long/short book (research reference only).

**Backtest.** `run_backtest_periodic` adopts the strategy weights on a rebalance
grid (every `REBALANCE_DAYS` dates), holds them in between, and earns the actual
daily return of held names — each return counted exactly once. The daily-replay
engine (`src/backtest_daily.py`) goes further: per-day orders via
`trading.decide_orders`, fills simulated against the next day's OHLC with
limit-up/down rules, T+1 and lot constraints.

## Outputs (`outputs/`)

- `orders_*.csv` — next-day executable orders (from `predict_tomorrow.py`)
- `walkforward_windows_*.csv` — per-window OOS metrics
- `walkforward_oos_curve.parquet` — stitched OOS daily curve (plot data)

All charts (walk-forward dashboards, multi-seed robustness, factor layers)
have moved to the Dash frontend (`run_webapp.py`): use the 滚动Walk-Forward /
多Seed稳健性 sessions and the backtest session's factor analysis step.

## Notes & caveats

- Synthetic results demonstrate the machinery, not a real edge — the generator
  injects a known low signal-to-noise factor structure. Numbers are only
  meaningful on the real dataset.
- This is research/backtest tooling, not investment advice or a live trading
  system. Backtested performance does not guarantee future results.
- The original notebook is preserved unchanged for the Kaggle inference server;
  this system reuses its modeling idea but is independent of the Kaggle runtime.

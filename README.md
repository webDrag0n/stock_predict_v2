# Stock Prediction, Strategy & Backtest System

[中文文档](README_CN.md)

An end-to-end A-share (China) stock prediction / strategy / backtest framework,
built on real market data from [tushare pro](https://tushare.pro). It grew out of
the Jane Street Real-Time Market Data Forecasting baseline notebook
(`jane-street-baseline-lgb-xgb-and-catboost.ipynb`), which only trained a model
and scored weighted R². This system adds the two missing pieces — a **trading
strategy** and a **leakage-proof walk-forward backtest** — and ties everything
into one reproducible pipeline whose objective is to **maximize realized
return**.

```
tushare daily bars (full market, ~5700 ex-ST)  →  technical + fundamental factors
    →  predict (forward return)  →  strategy (positions)  →  backtest (PnL, risk)
```

## Features

- **Incremental multi-model ensemble** (`src/ensemble.py`): gbm_l2 /
  gbm_ret_weighted / ic_linear legs, daily continued training (LightGBM
  `init_model`) with rolling OOS-IC weighting (EMA-smoothed).
- **Rolling walk-forward engine** (`src/walkforward.py`): embargo-isolated
  out-of-sample validation sharing the same GBM hyperparameters and
  cross-sectional normalization as the production ensemble (single source:
  `walkforward.GBM_PARAMS`).
- **Strategies** (`src/strategy.py`): `long_topk` (A-share default, long-only,
  the only practically tradeable one for retail), `tanh`, `cross_sectional`
  (market-neutral, research reference only).
- **Backtest engine** (single): daily replay with next-day fill simulation
  (`src/backtest_daily.py` + `src/trading.py`) — the only backtest engine after
  the vectorized periodic-rebalance engine was removed. A-share cost model
  (commission + stamp duty), limit-up/down rules, T+1 and lot constraints,
  participation cap; walk-forward OOS predictions are replayed through the same
  engine via a fingerprinted score cache.
- **Five-layer risk overlay** (`src/risk.py`): single implementation shared by
  backtest and live paths.
- **Factor evaluation gate** (`src/factor_eval.py`): IC / RankIC / ICIR / t-stats
  / layer monotonicity before any factor enters the model; YAML formula factor
  library in `factor_library/`.
- **Fundamental factors** (`ep`/`bp`/`sp`/`ln_mv` from akshare, point-in-time
  aligned via `merge_asof(backward)`).
- **Web UI** (`run_webapp.py`, Dash): session-based pipelines with per-step
  reports — research sessions (`walkforward` / `train` / `modeleval` /
  `factoreval`) launched from the Model Lab "new task" panel, and the
  production `nextday` order-generation session from the Trading Desk.
- **Live tooling** (`predict_tomorrow.py`): next-day executable orders, model
  gating (`models/staging/` promotion), adaptive retraining triggers,
  backtest-vs-live drift tracking.

## Quick start

Requires [uv](https://github.com/astral-sh/uv) and Python ≥ 3.10.

```bash
# 1. Install dependencies (creates .venv automatically)
uv sync --extra dev          # core + pytest
# optional extras: --extra attn (torch attention leg)
#                  --extra full (XGBoost / CatBoost)
#                  --extra ui   (Dash webapp)

# 2. Configure your tushare token in config_credentials.toml (gitignored):
#    [tushare]
#    token = "your_token"
#    (the TUSHARE_TOKEN environment variable takes precedence)

# 3. Run
uv run python run_webapp.py                  # web UI (recommended)
uv run python run_webapp.py --debug          # hot reload for development
                                             # (default mode: restart the process after code changes)
uv run python scripts/run_walkforward.py     # rolling walk-forward OOS backtest
uv run python predict_tomorrow.py            # next-day executable orders
uv run pytest                                # test suite
```

The default universe is the full market (~5700 ex-ST stocks, fetched per day).
First run downloads and caches daily bars into `data_cache/tushare/`; later runs
load in seconds.

All tunables live in `config.py` and can be overridden via environment
variables for every entry point:

```bash
START_DATE=2020-01-01 END_DATE=2024-12-31 FORWARD_WINDOW=5 \
  uv run python scripts/run_walkforward.py
```

**Prediction target.** `responder_6` is the forward `FORWARD_WINDOW`-day return
(clipped to ±0.5). With `MARKET_NEUTRAL_TARGET=1` (default) the same-day
cross-sectional mean is subtracted, so the model predicts cross-sectional
excess return — aligned with ranking strategies like `long_topk`. Turn it off
for directional strategies like `tanh`.

## Usage

```bash
# Rolling walk-forward OOS validation (the trusted benchmark)
uv run python scripts/run_walkforward.py --compare

# Recommended wide-cross-section config (300 screened names, 20-day window)
USE_SCREENING=1 UNIVERSE_SIZE=300 FORWARD_WINDOW=20 \
  uv run python scripts/run_walkforward.py --compare

# Next-day orders / commit fills back to holdings
uv run python predict_tomorrow.py --capital 1000000
uv run python predict_tomorrow.py --commit

# Model gating: staging models must beat the champion in dual replay
uv run python predict_tomorrow.py --no-gate          # skip gating
uv run python predict_tomorrow.py --promote <id>     # manually promote

# Adaptive retraining & drift tracking
uv run python predict_tomorrow.py --force-retrain
uv run python predict_tomorrow.py --tracking-report

# Single-factor evaluation / multi-seed robustness
# (the factoreval web session is an equivalent entry point for eval_factors)
uv run python scripts/eval_factors.py
uv run python scripts/research/multiseed_robust.py
```

Outputs land in `outputs/` (`orders_*.csv`, session reports). Walk-forward
results use the canonical `outputs/walkforward/` contract shared by the CLI
and webapp sessions: `oos_curve.parquet` (stitched OOS equity curve),
`window_metrics.parquet` (per-window OOS metrics), `oos_bench.parquet`
(equal-weight benchmark daily returns) and `meta.json` (summary metrics +
config snapshot). The Model Lab page renders the latest CLI walk-forward
result directly from this directory. Dashboards live in the web
UI; PNG chart output has been retired.

## Repository layout

```
config.py                  # all parameters + credential loading (env-overridable)
config_credentials.toml    # tushare token (gitignored, never commit)
predict_tomorrow.py        # CLI: next-day orders from current holdings
run_session.py             # session executor (spawned by the webapp)
run_webapp.py              # Dash front-end entry

src/                       # core library (top-level modules, src/ on sys.path)
  data/                    # connectors (tushare/akshare), factor engineering, panel loaders
    factors/               # formula factor engine (parser/registry/evaluator)
  walkforward.py           # rolling walk-forward engine + GBM_PARAMS
  ensemble.py              # incremental multi-model ensemble
  strategy.py              # prediction -> positions
  trading.py               # daily order decisions + fill simulation (backtest & live)
  risk.py                  # five-layer risk overlay (single implementation)
  backtest_daily.py        # daily-replay engine (the only backtest engine)
  factor_eval.py           # single-factor IC/ICIR/layering evaluation
  measure_alpha.py         # Newey-West alpha significance
  live.py / live_state.py / live_tracking.py   # live inference, state, drift tracking
  model_store.py / model_gate.py               # model artifacts + promotion gating

pipeline/                  # session pipeline framework (registry/context/steps)
webapp/                    # Dash multi-page front-end
scripts/                   # research CLIs (active vs archive: see scripts/README.md)
factor_library/            # YAML formula factor library
tests/                     # pytest suite
docs/system_architecture.html  # architecture diagram (open in browser)
```

New data sources: implement the 3 methods of `DataConnector` under `src/data/`
and register in `src/data/__init__.py`'s `_CONNECTORS`.

## Documentation

| Document | Contents |
|---|---|
| [README_CN.md](README_CN.md) | 中文文档（含实测结论与更多细节） |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Developer guide: tests, debugging, conventions, extension points |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CLAUDE.md](CLAUDE.md) | Full research conclusions & methodology rules (AI-assistant context, human-readable) |
| [factor_library/README.md](factor_library/README.md) | Formula factor language |
| [scripts/README.md](scripts/README.md) | Active vs archived scripts |
| [docs/system_architecture.html](docs/system_architecture.html) | Architecture diagram |

## Notes & caveats

- This is research/backtest tooling, not investment advice or a live trading
  system. Backtested performance does not guarantee future results.
- A-share T+1 is enforced in the fill layer (`src/trading.py`).
- Strategy and hyperparameters are fixed in `config.py`; all conclusions are
  drawn from stitched walk-forward OOS curves only, to avoid data-snooping.
- The original Kaggle notebook is preserved unchanged; this system reuses its
  modeling idea but runs independently of the Kaggle runtime.

## License

[MIT](LICENSE) — see the LICENSE file. Replace the copyright holder placeholder
with your name before publishing.

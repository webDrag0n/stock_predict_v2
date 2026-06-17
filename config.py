"""Central configuration for the A-share prediction + strategy + backtest system.

Targets the Chinese A-share market via tushare. Data flow:
  - fetch CSI 300 constituents' daily bars (forward-adjusted) + daily_basic
  - engineer technical + A-share factors (turnover rate, volume ratio, limit flags)
  - transform into the Jane Street row format:
      rows are (date_id, time_id, symbol_id) observations (time_id=0 for daily)
      feature_00..feature_78 are the engineered factors
      weight is the per-row scoring weight (inverse volatility + recency)
      responder_6 is the target: forward N-day return (clipped)
The model predicts responder_6; the strategy turns predictions into positions;
the backtester measures realized PnL with the goal of maximizing return.

A-share specifics:
  - retail investors cannot short single names -> the default tradeable strategy
    is long_topk (long the top-K%% predicted names, equal weight). cross_sectional
    market-neutral is kept as a RESEARCH reference only (not practically tradeable).
  - costs: 0.02%% commission per side + 0.1%% stamp duty on the SELL side.
"""
from __future__ import annotations

import os
import tomllib

# --- Paths -----------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
CACHE_DIR = os.path.join(ROOT, "data_cache")
CREDENTIALS_FILE = os.path.join(ROOT, "config_credentials.toml")


# --- Credentials -----------------------------------------------------------
def _load_credential(section: str, key: str, env_var: str) -> str:
    """Read a credential: env var takes precedence, else config_credentials.toml."""
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, "rb") as f:
                data = tomllib.load(f)
            return str(data.get(section, {}).get(key, "")).strip()
        except Exception:
            return ""
    return ""


TUSHARE_TOKEN = _load_credential("tushare", "token", "TUSHARE_TOKEN")

# --- Data source -----------------------------------------------------------
DATA_SOURCE = os.environ.get("DATA_SOURCE", "tushare")  # only "tushare" for now

# A-share index label for the universe (kept for reporting/compat). With a
# low-tier tushare token the index-constituent APIs are not accessible, so the
# universe is built by *whole-market market-cap screening* instead: at the
# backtest start date we rank all listed stocks by total market value and take
# the largest UNIVERSE_SIZE (a 沪深300-like large-cap pool), excluding ST names
# and recent IPOs. Selecting on the start date (not the end) avoids lookahead.
DEFAULT_INDEX = os.environ.get("DEFAULT_INDEX", "399300.SZ")
UNIVERSE_SIZE = int(os.environ.get("UNIVERSE_SIZE", "300"))

# Universe is always built by whole-market market-cap screening: rank all listed
# stocks by total_mv at the backtest start date, take the largest UNIVERSE_SIZE,
# excluding ST names and recent IPOs. Avoids survivorship bias and the narrow
# cross-section problem of a curated 49-stock list (IC≈0 on that set).
DEFAULT_SYMBOLS: list[str] = []

# Date range for historical data (None = auto: ~5 years back to today).
START_DATE = os.environ.get("START_DATE", None)  # YYYY-MM-DD
END_DATE = os.environ.get("END_DATE", None)      # YYYY-MM-DD

# Target prediction window (forward N trading days).
FORWARD_WINDOW = int(os.environ.get("FORWARD_WINDOW", "5"))

# Target type. True = cross-sectional excess forward return (aligns with the
# market-neutral / long_topk relative-ranking strategies; the model learns which
# names out-/under-perform the universe). False = raw forward return.
MARKET_NEUTRAL_TARGET = os.environ.get("MARKET_NEUTRAL_TARGET", "1") == "1"

# Fundamental (valuation) factors from tushare daily_basic: PE/PB/PS/dividend/
# market-cap, sampled monthly and forward-filled point-in-time. daily_basic is
# rate-limited to 1 call/min on the free tier, so the first build is slow
# (~1 snapshot/min) but fully cached afterwards. Set to "0" to skip.
USE_FUNDAMENTALS = os.environ.get("USE_FUNDAMENTALS", "1") == "1"
USE_FINANCIAL_FACTORS = os.environ.get("USE_FINANCIAL_FACTORS", "1") == "1"
FUND_SAMPLE_DAYS = int(os.environ.get("FUND_SAMPLE_DAYS", "21"))  # snapshot interval


# --- Schema ----------------------------------------------------------------
N_FEATURES = 79
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(N_FEATURES)]
TARGET = "responder_6"
WEIGHT = "weight"
DATE_COL = "date_id"
TIME_COL = "time_id"
SYMBOL_COL = "symbol_id"

# --- Data slicing (train / validation / test) ------------------------------
# Chronological 3-way split. The model trains on the train segment; strategies
# are SELECTED on the validation segment; the chosen strategy is REPORTED on the
# test segment. Selecting a strategy on the test set would be data snooping.
SKIP_DATES = 0          # drop early warm-up dates if needed
NUM_VAL_DATES = 60      # validation segment length (strategy selection)
NUM_TEST_DATES = 60     # test segment length (final out-of-sample report)
N_FOLD = 5              # purged date-based CV folds for training the ensemble

# --- Leakage control & walk-forward ----------------------------------------
# EMBARGO_DAYS: gap (in trading days) removed between any train segment and the
# out-of-sample segment that follows it. Must be >= FORWARD_WINDOW, otherwise
# the last FORWARD_WINDOW training labels (close.shift(-FORWARD_WINDOW)) peek
# into the OOS window's prices — a forward-information leak. Also applied
# between purged-CV folds.
EMBARGO_DAYS = int(os.environ.get("EMBARGO_DAYS", str(FORWARD_WINDOW)))

# Walk-forward (rolling) backtest: at each step, train on the trailing
# TRAIN_WINDOW days, embargo, then predict the next TEST_WINDOW days out of
# sample. Windows step by STEP (default = TEST_WINDOW => non-overlapping OOS).
TRAIN_WINDOW = int(os.environ.get("TRAIN_WINDOW", "750"))
TEST_WINDOW = int(os.environ.get("TEST_WINDOW", "60"))
WF_STEP = int(os.environ.get("WF_STEP", str(TEST_WINDOW)))

# Industry neutralization (P0a): when True, the target responder_6 is the
# industry-internal excess return (forward return minus the same-day industry
# mean) rather than the universe mean. Requires industry labels from stock_basic.
INDUSTRY_NEUTRAL = os.environ.get("INDUSTRY_NEUTRAL", "1") == "1"

# Rolling IC meta-features (Method C). Disabled by default: IC at test dates
# uses forward returns inside the test window (subtle look-ahead). Safe to re-
# enable only when computing IC from training dates exclusively (see walkforward).
REGIME_IC_FEATURES = os.environ.get("REGIME_IC_FEATURES", "0") == "1"

# --- Strategy --------------------------------------------------------------
# Default tradeable strategy for A-share retail: long the top-K% names.
STRATEGY = "long_topk"   # "long_topk" | "tanh" | "cross_sectional"(research-only)
TOPK_RATIO = 0.20        # fraction of the universe to go long each day (long_topk)
TANH_SCALE = 5.0         # steepness of tanh sizing
GROSS_LEVERAGE = 1.0     # total gross exposure per day

# --- A-share transaction costs ---------------------------------------------
COMMISSION_RATE = 0.0002   # brokerage commission, charged on BOTH buy and sell
STAMP_DUTY_RATE = 0.0010   # stamp duty, charged on the SELL side only

# --- Capital (for money-denominated PnL curve) -----------------------------
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000000"))  # ¥1,000,000

# --- Training --------------------------------------------------------------
RANDOM_STATE = 42

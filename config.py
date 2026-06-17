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

# Optional explicit symbol list (tushare ts_code, e.g. "600519.SH"). Non-empty
# => skip all universe-API calls entirely (recommended when your tushare token
# has limited quota — the free tier rate-limits stock_basic / daily_basic to
# 1 call / hour). The list below is 50 core 沪深300 blue-chips.
# Set to [] to fall back to the market-cap screening via API.
DEFAULT_SYMBOLS: list[str] = [
    # 白酒
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "000568.SZ",  # 泸州老窖
    "600809.SH",  # 山西汾酒
    "600436.SH",  # 片仔癀
    # 银行
    "600036.SH",  # 招商银行
    "601328.SH",  # 交通银行
    "601166.SH",  # 兴业银行
    "600016.SH",  # 民生银行
    "601398.SH",  # 工商银行
    "601939.SH",  # 建设银行
    "600000.SH",  # 浦发银行
    "000001.SZ",  # 平安银行
    # 保险 / 金融
    "601318.SH",  # 中国平安
    "601628.SH",  # 中国人寿
    "601601.SH",  # 中国太保
    "600030.SH",  # 中信证券
    "600837.SH",  # 海通证券
    "600999.SH",  # 招商证券
    # 消费 / 食品
    "600887.SH",  # 伊利股份
    "000333.SZ",  # 美的集团
    "000651.SZ",  # 格力电器
    # 新能源 / 汽车
    "300750.SZ",  # 宁德时代
    "002594.SZ",  # 比亚迪
    "601012.SH",  # 隆基绿能
    "600438.SH",  # 通威股份
    "601633.SH",  # 长城汽车
    # 医药 / 医疗
    "600276.SH",  # 恒瑞医药
    "300760.SZ",  # 迈瑞医疗
    "603259.SH",  # 药明康德
    # 科技 / 电子
    "002415.SZ",  # 海康威视
    "002230.SZ",  # 科大讯飞
    "300059.SZ",  # 东方财富
    # 地产 / 建筑
    "000002.SZ",  # 万科A
    "600048.SH",  # 保利发展
    "601800.SH",  # 中国交建
    "601390.SH",  # 中国中铁
    "601669.SH",  # 中国电建
    # 能源 / 化工
    "601088.SH",  # 中国神华
    "600309.SH",  # 万华化学
    "000338.SZ",  # 潍柴动力
    # 交通 / 物流
    "600009.SH",  # 上海机场
    "601888.SH",  # 中国中免
    # 重工 / 机械
    "600031.SH",  # 三一重工
    # 通信
    "600050.SH",  # 中国联通
    # 零售 / 互联网
    "603986.SH",  # 兆易创新
    "300122.SZ",  # 智飞生物
    "300124.SZ",  # 汇川技术
    "600196.SH",  # 复星医药
]

# USE_SCREENING=1 empties the curated list above, so the universe is built by
# whole-market market-cap screening (UNIVERSE_SIZE names). Use when you want a
# wider cross-section than the curated 50 (price fetched per-stock via pro_bar,
# which is not the rate-limited endpoint).
if os.environ.get("USE_SCREENING", "0") == "1":
    DEFAULT_SYMBOLS = []

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

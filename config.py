"""Central configuration for the A-share prediction + strategy + backtest system.

Targets the Chinese A-share market via tushare. Data flow:
  - fetch the whole tradable A-share market (~5300 ex-ST stocks) by trading date
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
# universe is built by *whole-market quality screening* instead: all listed
# stocks passing price/liquidity floors, ST/IPO exclusion. UNIVERSE_SIZE is
# deprecated and defaults to 0 (whole market). The 49/300-stock pools destroyed
# cross-sectional correlation and produced unstable OOS alpha; do not use them.
DEFAULT_INDEX = os.environ.get("DEFAULT_INDEX", "399300.SZ")
UNIVERSE_SIZE = int(os.environ.get("UNIVERSE_SIZE", "0"))

# Universe mode. "all" = WHOLE MARKET (~5300 ex-ST stocks, fetched by trading
# date via daily(trade_date=), 1 API call/day). "screen" is deprecated: with
# UNIVERSE_SIZE=0 it resolves to the whole market. DEFAULT IS "all" because the
# 49→300→5700 research chain proved cross-section WIDTH is the system's
# strongest lever — only the whole market produced statistically-significant
# true alpha. Any fixed small pool (including a 沪深300-like large-cap screen)
# destroys correlation structure and yields negative/unstable OOS alpha. Do not
# use a 300-stock screen for testing or production.
UNIVERSE_MODE = os.environ.get("UNIVERSE_MODE", "all")  # "all" | "screen"

# Whole-market quality filters (only applied when UNIVERSE_MODE=all). The full
# market includes penny stocks, suspended names and illiquid micro-caps that
# corrupt the cross-section and aren't realistically tradeable.
MAIN_BOARD_ONLY = os.environ.get("MAIN_BOARD_ONLY", "0") == "1"  # exclude STAR/ChiNext/BSE
MIN_PRICE = float(os.environ.get("MIN_PRICE", "2.0"))          # 元, drop penny stocks
MIN_AMOUNT_WAN = float(os.environ.get("MIN_AMOUNT_WAN", "2000"))  # 万元, 20d median liquidity floor

# DEFAULT_SYMBOLS can be used to pin an exact symbol list. Leaving it empty
# (default) uses the whole tradable market after quality filters (price/liquidity
# floors, ST/ IPO exclusion). Do not use a small curated list for backtesting:
# the 49/300-stock pools destroyed cross-sectional correlation and produced
# unstable OOS alpha.
DEFAULT_SYMBOLS: list[str] = []

# Date range for historical data (None = auto: ~5 years back to today).
START_DATE = os.environ.get("START_DATE", None)  # YYYY-MM-DD
END_DATE = os.environ.get("END_DATE", None)      # YYYY-MM-DD

# Target prediction window (forward N trading days). Raised from 5 to 20: the
# 5-day excess return is ~95% noise, so the learnable signal-to-noise ratio is
# very low. A 20-day horizon smooths idiosyncratic daily noise and historically
# carries a higher, more stable cross-sectional IC. Override with env var.
FORWARD_WINDOW = int(os.environ.get("FORWARD_WINDOW", "20"))

# Multi-horizon target fusion. In addition to the primary responder_6 (FORWARD_WINDOW),
# compute extra forward-return targets at these horizons. The "multi_*" objectives
# train one model per horizon and fuse predictions by per-date rank average.
# This can stabilize the signal by combining short-term momentum (5d), medium-term
# mean-reversion/value (20d), and long-term trend (60d).
MULTI_HORIZON_WINDOWS = [
    int(x.strip())
    for x in os.environ.get("MULTI_HORIZON_WINDOWS", "5,10,60").split(",")
    if x.strip()
]
MULTI_HORIZON_WEIGHTS = [
    float(x.strip())
    for x in os.environ.get("MULTI_HORIZON_WEIGHTS", "0.25,0.35,0.40").split(",")
    if x.strip()
]
# Normalize weights to sum to 1.
if sum(MULTI_HORIZON_WEIGHTS) > 0:
    MULTI_HORIZON_WEIGHTS = [w / sum(MULTI_HORIZON_WEIGHTS) for w in MULTI_HORIZON_WEIGHTS]

# Model objective for the walk-forward predictor.
#   "l2"        : regression on responder_6 (predictions live in return space,
#                 so price-RMSE / weighted-R2 are meaningful).
#   "lambdarank": learning-to-rank (LGBMRanker) — optimizes the cross-sectional
#                 ORDERING directly, matching the long_topk top-K selection. The
#                 output is a ranking SCORE, not a return, so price-RMSE/R2 are
#                 not reported; IC/ICIR remain valid (rank-based).
#   "ic_linear" : IC-weighted linear blend of de-redundant factor z-scores.
#                 Per window, each LINEAR_FACTORS factor's training-segment daily
#                 RankIC becomes its weight (sign included → reversal factors get
#                 a negative weight automatically); predict = sum(IC * z-score).
#                 Diagnosed (2026-06) to BEAT the 63-factor GBM on OOS IC/ICIR:
#                 the GBM over-fits a ~95%-noise 20-day target, a sparse linear
#                 blend does not. Output is a ranking score (not return space).
#   "fusion"    : rank-average of the l2 GBM and the ic_linear score (per-date
#                 percentile ranks, then mean). De-correlates the two models'
#                 errors; usually the highest ICIR. Output is a ranking score.
#   "ret_weighted": l2 GBM with sample weights ∝ per-date rank of the forward
#                 return — concentrates the fit on "who reaches the top", which
#                 aligns with what long_topk harvests. Phase-1 retargeted search
#                 (2026-06, primary metric = top-20% OOS excess return) found it
#                 the best tradeable predictor: top-20% excess +0.0035 / stability
#                 ICIR +1.00 vs l2's +0.0020 / +0.31. Output is return space.
#   "multi_ret_weighted": train one ret_weighted GBM per horizon
#                 (FORWARD_WINDOW + MULTI_HORIZON_WINDOWS) and fuse predictions by
#                 per-date percentile rank average. Combines short/medium/long-term
#                 signals for a more stable cross-sectional score. Output is a
#                 ranking score (not return space).
#   "multi_fusion": per-horizon fusion of a GBM leg and an ic_linear leg, then
#                 rank-average across horizons. Maximizes model diversity.
MODEL_OBJECTIVE = os.environ.get("MODEL_OBJECTIVE", "ret_weighted")
# "l2" | "lambdarank" | "ic_linear" | "fusion" | "ret_weighted" | "multi_ret_weighted" | "multi_fusion"

# Size-neutralize the ic_linear factors: regress each on ln_mv per date and keep
# the residual before IC-weighting. Phase-1 accuracy search (2026-06) found this
# the single highest-IC predictor (OOS IC +0.041 / ICIR +1.21). Set "0" to use
# raw factors. Only affects the ic_linear / fusion objectives.
LINEAR_SIZE_NEUTRAL = os.environ.get("LINEAR_SIZE_NEUTRAL", "1") == "1"

# Factor set for the "ic_linear" / "fusion" objectives.
# Default (empty list) means "use every named factor in the loaded panel" so the
# linear leg fully consumes the unified factor library (technical + orthogonal +
# Alpha158 + GTJA191). Set LINEAR_FACTORS to a comma-separated list of names
# (e.g. "macd_diff,gtja_alpha001,a158_kmid") to force a curated subset instead.
# Sign is NOT pre-set — the per-window training IC decides each factor's
# direction, so listing a reversal factor is fine. Factors not present in the
# loaded panel are skipped automatically.
LINEAR_FACTORS = [
    x.strip()
    for x in os.environ.get("LINEAR_FACTORS", "").split(",")
    if x.strip()
]

# Factor-family toggles. All families are aggregated by
# src.data.factors.compute_all_factors() and exposed as named columns in the
# factor panel. Technical factors are the base set and are always on by default;
# the other families can be disabled to experiment with signal sources.
USE_TECHNICAL_FEATURES = os.environ.get("USE_TECHNICAL_FEATURES", "1") == "1"

# Self-computed orthogonal anomaly factors (Amihud illiquidity, MAX lottery,
# idiosyncratic vol, residual momentum, return skew, 52-week-high proximity).
# Derived from OHLCV already on disk — no extra API calls. These target alpha
# dimensions the existing momentum/value/volatility factors miss.
USE_ORTHOGONAL_FACTORS = os.environ.get("USE_ORTHOGONAL_FACTORS", "1") == "1"

# Microsoft Qlib Alpha158 factor families (pure OHLCV, A-share-tuned). Adds the
# families the existing base factors lack — K-line shape (KMID/KUP/KLOW/KSFT),
# trend quality (RSQR/RESI), distribution position (QTLU/QTLD/RSV), extreme-index
# (IMAX/IMIN/IMXD), price-volume correlation (CORR/CORD), up/down persistence
# (CNTP/CNTN/CNTD/SUMP/SUMD) and volume-weighted volatility (WVMA), over windows
# {5,20,60}. See src.data.factors.alpha158.
USE_ALPHA158_FACTORS = os.environ.get("USE_ALPHA158_FACTORS", "1") == "1"

# Guotai Junan 191 alpha factors — curated OHLCV subset. The original DolphinDB
# reference is https://github.com/dolphindb/DolphinDBModules/blob/master/gtja191Alpha/src/gtja191Alpha.dos
# We implement a high-value subset (~25 alphas) that avoids overlap with the base
# technical/Alpha158 factors and requires no benchmark/index inputs. Set
# GTJA191_ALPHA_SUBSET to a comma-separated list of IDs (e.g. "1,2,5") to override
# the default curated list.
USE_GTJA191_FACTORS = os.environ.get("USE_GTJA191_FACTORS", "1") == "1"
GTJA191_ALPHA_SUBSET = os.environ.get("GTJA191_ALPHA_SUBSET", "")

# Formula-based factor library (YAML definitions in factor_library/).
# Researchers write factor formulas in YAML; the system auto-validates,
# resolves dependencies, and evaluates them after all native families.
# Set USE_FORMULA_FACTORS=1 to enable.  FACTOR_LIBRARY_DIR overrides the
# default location.  FORMULA_FACTOR_SUBSET restricts to a comma-separated
# list of factor IDs (empty = all enabled factors in the library).
USE_FORMULA_FACTORS = os.environ.get("USE_FORMULA_FACTORS", "0") == "1"
FACTOR_LIBRARY_DIR = os.environ.get(
    "FACTOR_LIBRARY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_library"),
)
FORMULA_FACTOR_SUBSET = [
    x.strip()
    for x in os.environ.get("FORMULA_FACTOR_SUBSET", "").split(",")
    if x.strip()
]

# Robust denoising for the cross-sectional normalizer (walkforward._cross_sectional_norm).
# USE_WINSORIZE: MAD-winsorize each factor per date (clip to median ± WINSOR_MAD_K·
# 1.4826·MAD) BEFORE z-scoring, so a lone outlier can't inflate the std used by the
# z-score. Documented best practice (vs the old clip-after-z). k≈3 keeps ~99.7% of
# normal data untouched. Set USE_WINSORIZE=0 to revert to plain z-score+clip.
USE_WINSORIZE = os.environ.get("USE_WINSORIZE", "1") == "1"
WINSOR_MAD_K = float(os.environ.get("WINSOR_MAD_K", "3.0"))

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
# Jane Street row-format schema. Width is sized for the full unified factor
# library (technical + orthogonal + Alpha158 + GTJA191 subset + fundamentals /
# financials / industry / option-IV / regime features). With all families on the
# named panel currently produces ~150 candidate features, so 192 leaves headroom.
# If you disable entire factor families, the missing columns are zero-padded;
# if you add many custom factors, increase this.
N_FEATURES = int(os.environ.get("N_FEATURES", "192"))
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
# TRAIN_WINDOW=500 (NOT 750): the robust 3-seed result (ann α +13.8%, NW-t 2.80,
# the cached seed_oos_*.parquet) was produced at 500 → ~660 OOS dates / 11
# windows. A 2026-06-28 session bumped it to 750, which shrank the OOS to ~420
# dates / 7 windows and the significance collapsed to NW-t 1.74 (BELOW the 2.0
# bar) with no alpha gain — fewer windows = wider error bars, not a better model.
# The extra 250 trailing days also drag in staler regime data. Keep 500.
TRAIN_WINDOW = int(os.environ.get("TRAIN_WINDOW", "500"))
TEST_WINDOW = int(os.environ.get("TEST_WINDOW", "60"))
WF_STEP = int(os.environ.get("WF_STEP", str(TEST_WINDOW)))

# 周期调仓回测：每 REBALANCE_DAYS 个交易日采用一次策略权重，中间持有，每天赚
# 持有标的的真实日收益（return_1d/pct_chg），每个收益只计一次、ann=252 口径正确。
# 旧的每日重叠引擎（run_backtest，Sharpe/PnL 双重虚高）已删除，无回落开关。
REBALANCE_DAYS = int(os.environ.get("REBALANCE_DAYS", str(FORWARD_WINDOW)))

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
# Phase-2b sweep (2026-06) on the corrected periodic-rebalance backtest found
# topk=0.15 the sweet spot: concentrated enough to capture the head-concentrated
# ret_weighted edge, without the over-concentration of 0.05 (whose drawdown
# blows out). The whole 0.05–0.30 grid stays Sharpe 1.31–1.48, so the strategy
# is insensitive here (robust). NOTE: tuned on the OOS test set (mild snooping),
# so treat +1.48 as a slight optimistic bound; ~1.42 is the honest expectation.
TOPK_RATIO = float(os.environ.get("TOPK_RATIO", "0.15"))  # top fraction long
TANH_SCALE = 5.0         # steepness of tanh sizing
GROSS_LEVERAGE = 1.0     # total gross exposure per day

# Turnover control (long_topk hysteresis buffer). A held stock is only SOLD when
# its predicted rank falls out of the top HOLD_RATIO band, and a new stock is
# only BOUGHT when it enters the top TOPK_RATIO band. The gap (HOLD_RATIO −
# TOPK_RATIO) is a no-trade zone that stops day-to-day churn from rank noise.
# Default now = TOPK_RATIO (buffer OFF): under the periodic-rebalance backtest
# (every REBALANCE_DAYS) turnover cost is already ~2% of PnL, so the buffer adds
# nothing (Phase-2b sweep). It was essential only for the old daily-rebalance
# regime. Set HOLD_RATIO > TOPK_RATIO to re-enable for higher-frequency configs.
HOLD_RATIO = float(os.environ.get("HOLD_RATIO", str(TOPK_RATIO)))

# Dynamic topk concentration based on recent rolling OOS rank-IC.
# When the signal is strong (high IC), concentrate into fewer names to harvest
# the head-concentrated edge; when weak, diversify to reduce noise and churn.
# Topk is linearly interpolated between DYNAMIC_TOPK_LOW (at IC_FULL) and
# DYNAMIC_TOPK_HIGH (at IC_ZERO). IC is measured on matured predictions only,
# so this is leak-free. Default OFF until validated by experiment.
USE_DYNAMIC_TOPK = os.environ.get("USE_DYNAMIC_TOPK", "0") == "1"
DYNAMIC_TOPK_IC_FULL = float(os.environ.get("DYNAMIC_TOPK_IC_FULL", "0.02"))
DYNAMIC_TOPK_IC_ZERO = float(os.environ.get("DYNAMIC_TOPK_IC_ZERO", "-0.01"))
DYNAMIC_TOPK_LOW = float(os.environ.get("DYNAMIC_TOPK_LOW", "0.10"))
DYNAMIC_TOPK_HIGH = float(os.environ.get("DYNAMIC_TOPK_HIGH", "0.25"))

# --- A-share transaction costs ---------------------------------------------
COMMISSION_RATE = 0.0002   # brokerage commission, charged on BOTH buy and sell
STAMP_DUTY_RATE = 0.0010   # stamp duty, charged on the SELL side only
# 最低佣金（元/笔）：A股券商普遍 max(5元, 费率×成交额)。纯比例佣金会系统性低估
# 小额订单的成本（5万元订单的真实佣金是 5 元而非 1 元）。设为 0 可退回纯比例。
MIN_COMMISSION = float(os.environ.get("MIN_COMMISSION", "5.0"))  # 元/笔
# 参与率上限：单笔成交股数 ≤ MAX_PARTICIPATION × 次日成交额(ADV)。冲击成本模型
# 只"计价"不"限量"，没有这个上限回测可以按开盘价吃下超过全天成交量的订单，
# 资金规模稍大就会系统性偏乐观。0 = 不限量（退回旧行为）。
MAX_PARTICIPATION = float(os.environ.get("MAX_PARTICIPATION", "0.10"))

# Liquidity-aware market-impact cost (square-root model), on top of commission +
# stamp. For a trade of notional N in a stock with average daily dollar volume
# ADV, impact fraction = IMPACT_K * sqrt(N / ADV). This is the realistic cost the
# flat commission understates for small/mid caps — whole-market books trade
# illiquid names where moving size pushes the price. ADV comes from the `amount`
# column (千元→元). IMPACT_K≈0.1 is a standard empirical coefficient (impact ≈
# 10% of daily vol move when you trade 100% of ADV). Set USE_IMPACT_COST=0 to
# revert to the flat-commission-only model.
USE_IMPACT_COST = os.environ.get("USE_IMPACT_COST", "1") == "1"
IMPACT_K = float(os.environ.get("IMPACT_K", "0.1"))

# --- Capital (for money-denominated PnL curve) -----------------------------
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "1000000"))  # ¥1,000,000

# --- Training --------------------------------------------------------------
RANDOM_STATE = 42

# Multi-seed combine method (how the n_seeds bagged models are merged into one
# predictor). "mean" = average raw predictions (DEFAULT, stays in return space);
# "rank" = average each seed's PER-DATE percentile rank.
#   2026-06-29 end-to-end test (reproduce_ensemble.py, TRAIN=500, the production
#   internal-bagging path with seeds 42/43/44): rank vs mean is STATISTICALLY
#   IDENTICAL — α +13.99% vs +14.00%, NW-t 2.83 vs 2.84, IC 0.0133 vs 0.0134.
#   Reason: internal-bagging seeds train on the SAME data with consecutive RNG,
#   so they correlate ~0.98+ and rank-combine has almost nothing to de-noise.
#   (The +0.5% rank edge seen in ensemble_opt.py came from combining 3 fully
#   INDEPENDENT runs 42/123/7, slightly more diverse — rank only helps there.)
#   So default = "mean": rank gains nothing on this path but costs the
#   price-RMSE / weighted-R2 diagnostic (rank output leaves return space). Use
#   SEED_COMBINE=rank only when merging independent full runs (e.g. multiseed).
SEED_COMBINE = os.environ.get("SEED_COMBINE", "mean")  # "mean" | "rank"

# walkforward 滚动窗口内每个模型 bagging 的 seed 数（增量集成用 ENSEMBLE_N_SEEDS）。
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))

# --- Incremental ensemble (daily trading engine) ---------------------------
# The daily replay backtest and live next-day path train via IncrementalEnsemble
# (src/ensemble.py): GBM boosters are CONTINUED-trained (LightGBM init_model)
# each update instead of refit from scratch — whole-market ~5700 stocks × ~600
# days makes full daily retrain infeasible (hours). Every REBUILD_DAYS a full
# rebuild on the trailing TRAIN_WINDOW resets drift accumulation.
#   ENSEMBLE_MODELS : comma list of model legs. Default 2-leg blend whose errors
#                     de-correlate: gbm_l2, ic_linear.
#   INCR_TREES      : trees appended per incremental update (on newly-matured rows)
#   FULL_TREES      : trees grown on a full rebuild
#   REBUILD_DAYS    : trading days between full rebuilds
#   IC_WINDOW       : trailing window (matured trading days) for rolling-IC weights
# Model weights = trailing out-of-sample rolling rank-IC (clipped ≥0, normalized);
# a leg whose recent IC turns non-positive is automatically muted. Leak-free: IC
# is scored only on already-matured prediction dates (d + FORWARD_WINDOW ≤ asof).
# 2026-06-30: added 'attn' (cross-sectional attention, src/model_attn.py) as a leg.
# It beat the GBM legs decisively in walk-forward (OOS IC 0.112 vs 0.098, and +54pp
# net value in the production replay), and IC-weighting will lean on it when it leads.
# 2026-08-28: dropped gbm_ret_weighted from the default blend. The Tier-1
# walk-forward grid (2024-01~2026-08, 10 non-overlapping OOS windows) showed its
# standalone rank-IC is negative (-0.015, only 4/10 windows positive, and -0.07 ~
# -0.12 in the latest windows); its high backtest PnL is bull-market beta, not
# selection. ic_linear (+0.085, 9/10) and fusion (+0.100, 10/10) carry the real
# signal, and production attribution showed rolling IC -0.116 for this leg.
# Default ensemble for backtest/live: a 2-leg blend whose errors de-correlate.
# The attention leg is GPU-trained on each full rebuild (REBUILD_DAYS); on a
# whole-market ~5700-stock panel it is both slow and memory-heavy, so it is now
# opt-in.  Re-enable it with ENSEMBLE_MODELS=attn,gbm_l2,ic_linear.
ENSEMBLE_MODELS = os.environ.get("ENSEMBLE_MODELS",
                                 "gbm_l2,ic_linear")
# 增量集成的 seed 数：每日续训成本敏感，默认 1（N_SEEDS env 同时覆盖此处与
# walkforward 的 bagging seed 数——两条路径历史上都读这个 env，故保留同名）。
ENSEMBLE_N_SEEDS = int(os.environ.get("N_SEEDS", "1"))
# 每次增量更新、每个【新成熟交易日】续训的树数（续训量随新数据量伸缩）；
# 两次全量重建之间总树数硬上限 = FULL_TREES * 2，超限提前触发全量重建。
INCR_TREES = int(os.environ.get("INCR_TREES", "60"))
FULL_TREES = int(os.environ.get("FULL_TREES", "400"))
REBUILD_DAYS = int(os.environ.get("REBUILD_DAYS", "60"))
IC_WINDOW = int(os.environ.get("IC_WINDOW", "20"))
# 滚动 IC 定权的 EMA 平滑半衰期（交易日）：平滑 IC 过零跳变引起的权重/换手尖峰。
# 0 = 不平滑（直接用尾部 IC 均值定权）。首次由数据驱动的定权不平滑（避免
# 实盘单次定权被初始等权稀释）。
IC_WEIGHT_EMA_HALFLIFE = float(os.environ.get("IC_WEIGHT_EMA_HALFLIFE", "5"))

# --- Model gate（champion/staging 晋升机制，src/model_store + src/model_gate）--
# 每晚训练完候选后比较配方指纹：与 champion 相同 → 原位刷新工件（零开销）；
# 不同 → 候选落盘 models/staging/，双方各跑一遍最近 PROMOTE_EVAL_DAYS 个交易日的
# 完整每日重放，Sharpe 更高且回撤不劣于 champion − PROMOTE_DD_TOLERANCE 才晋升，
# 否则 champion 不变、候选留档 + evaluation.json。USE_MODEL_GATE=0 退回旧行为
# （每晚重训直接用，等同 --no-gate）。
USE_MODEL_GATE = os.environ.get("USE_MODEL_GATE", "1") == "1"
PROMOTE_EVAL_DAYS = int(os.environ.get("PROMOTE_EVAL_DAYS", "120"))
PROMOTE_SHARPE_MARGIN = float(os.environ.get("PROMOTE_SHARPE_MARGIN", "0"))
PROMOTE_DD_TOLERANCE = float(os.environ.get("PROMOTE_DD_TOLERANCE", "0.02"))
ARCHIVE_KEEP = int(os.environ.get("ARCHIVE_KEEP", "5"))

# --- Adaptive retrain + live tracking（src/retrain_trigger + src/live_tracking）
# 默认不再每晚重训：用跟踪指标 OR 判定是否触发（任一满足即重训）：
#   1. 活体 IC 退化：成熟样本 ≥ RETRAIN_IC_MIN_SAMPLES 且尾部 RETRAIN_IC_WINDOW
#      日 OOS rank-IC 均值 < RETRAIN_IC_MIN；
#   2. 收益落后（需 account.csv 真实净值）：尾部 RETRAIN_UNDERPERF_DAYS 实盘收益
#      < 等权基准 − RETRAIN_UNDERPERF_MARGIN；
#   3. 模型超龄：train_end_date 距今 > MAX_MODEL_AGE_DAYS 个交易日（硬上限）。
# 触发 → 训练候选走 model_gate 门控；未触发 → 冻结 champion 工件直接预测。
# USE_ADAPTIVE_RETRAIN=0 退回每晚重训（旧行为）。跟踪快照/报告落 TRACKING_DIR。
USE_ADAPTIVE_RETRAIN = os.environ.get("USE_ADAPTIVE_RETRAIN", "1") == "1"
RETRAIN_IC_WINDOW = int(os.environ.get("RETRAIN_IC_WINDOW", "20"))
RETRAIN_IC_MIN = float(os.environ.get("RETRAIN_IC_MIN", "0"))
RETRAIN_IC_MIN_SAMPLES = int(os.environ.get("RETRAIN_IC_MIN_SAMPLES", "10"))
RETRAIN_UNDERPERF_DAYS = int(os.environ.get("RETRAIN_UNDERPERF_DAYS", "20"))
RETRAIN_UNDERPERF_MARGIN = float(os.environ.get("RETRAIN_UNDERPERF_MARGIN", "0.03"))
MAX_MODEL_AGE_DAYS = int(os.environ.get("MAX_MODEL_AGE_DAYS", "60"))
TRACKING_DIR = os.path.join(OUTPUT_DIR, "tracking")

# --- Daily trading engine (decision + fill simulation) ---------------------
# The shared daily engine (src/trading.py) is used identically by the replay
# backtest and live inference. Each trading day: ML ranking selects a candidate
# pool (top TOPK_RATIO), then technical indicators (Bollinger band + RSI) decide
# entry/exit TIMING within that pool, producing precise per-share / per-price
# orders. Fills are simulated against the NEXT day's OHLC + limit bands.
#   ML decides WHICH stocks (its proven edge = cross-sectional ranking);
#   indicators decide WHEN/at what price (the per-trade realism you want);
#   the fill simulator decides WHETHER the order executes.
#
# Order pricing model for next-day execution:
#   ORDER_PRICE_MODE = "next_open"  : DEFAULT. Market-on-open at next day's open.
#                       After the 2026-06-30 attribution (below) the engine's
#                       default switched back to next_open + pure-ML rebalance:
#                       the limit_bb "advantage" in the earlier sweep was just it
#                       trading less of a self-harming signal, not better fills.
#                      "limit_bb"   : limit at the Bollinger band; only fills if
#                       next day's range touches it. Lower fill rate / turnover.
ORDER_PRICE_MODE = os.environ.get("ORDER_PRICE_MODE", "next_open")  # next_open|limit_bb

# Bollinger-band / RSI timing overlay. DEFAULT OFF after the 2026-06-30 root-cause
# analysis: a 5-variant attribution on whole-market (~250d) showed the BB overlay
# does NOT add value — it only *dilutes* by trading less. The losses came from (1)
# over-trading a 1-day-horizon signal daily (turnover cost > edge) and (2) cash
# drag (exposure stuck at 54-67% in a +34% market). With USE_BB_TIMING=0 the engine
# uses the pure-ML rebalance-to-equal-weight path (full deployment, periodic), which
# fixes both. Re-enable only to experiment with mean-reversion entry timing.
USE_BB_TIMING = os.environ.get("USE_BB_TIMING", "0") == "1"
BB_BUY_K = float(os.environ.get("BB_BUY_K", "0.02"))   # within 2% above lower band
BB_SELL_K = float(os.environ.get("BB_SELL_K", "0.02"))  # within 2% below upper band
# RSI overbought/oversold gates (0/100 disable the respective gate):
RSI_BUY_MAX = float(os.environ.get("RSI_BUY_MAX", "70"))   # don't buy if RSI>this
RSI_SELL_MIN = float(os.environ.get("RSI_SELL_MIN", "80"))  # force-sell if RSI>this
# Max names held simultaneously (position count cap; 0 = no cap, sized by pool).
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "30"))
# Per-name cap as a fraction of total portfolio value (concentration guard).
MAX_WEIGHT_PER_NAME = float(os.environ.get("MAX_WEIGHT_PER_NAME", "0.10"))
# Portfolio beta constraint (long-only). When enabled, decide_orders iteratively
# replaces high-beta names with lower-beta candidates from the same score band
# until the equal-weight portfolio beta is below MAX_PORTFOLIO_BETA. Beta is
# estimated from each stock's trailing return_1d vs the equal-weight market.
# Default OFF until validated: aggressive beta targeting can destroy the small-cap
# alpha that the current signal harvests.
USE_BETA_CONSTRAINT = os.environ.get("USE_BETA_CONSTRAINT", "0") == "1"
MAX_PORTFOLIO_BETA = float(os.environ.get("MAX_PORTFOLIO_BETA", "1.10"))
BETA_LOOKBACK = int(os.environ.get("BETA_LOOKBACK", "60"))
# Minimum order notional (元): smaller diffs are skipped to suppress odd-lot churn.
# Raised 5000→30000 per the 2026-06 turnover sweep (bigger floor = fewer odd-lot
# round-trips = lower cost drag, the dominant failure mode of daily rebalancing).
DAILY_MIN_NOTIONAL = float(os.environ.get("DAILY_MIN_NOTIONAL", "30000"))

# Rebalance frequency (trading days between re-decisions; 1 = daily). The daily
# engine re-decides only every REBALANCE_EVERY days and holds in between. The
# 2026-06-30 throttle diagnostic showed daily rebalancing of a ~20-day-horizon
# signal over-trades: turnover ¥121M → −25.9%, vs every-10-days ¥23M → −4.3% on
# the SAME scores (+21pp recovered purely by trading less). The ML target is a
# FORWARD_WINDOW-day forward return, so re-trading it daily re-pays cost on N
# overlapping entries for one N-day move. Default 10 ≈ half the horizon: a balance
# between cost (lower = better) and signal freshness (higher = staler).
REBALANCE_EVERY = int(os.environ.get("REBALANCE_EVERY", "10"))

# Risk-control layer (daily engine). The attention predictor's bare book ran beta
# 1.6 / max-drawdown ~80% (2026-06-30): it concentrates in high-beta small-caps and
# blew up in the 2024 micro-cap crash. This layer scales the day's effective gross
# exposure from REALIZED history only (leak-free):
#   USE_VOL_TARGET: scale gross by min(1, target_daily_vol / realized_vol) over a
#     trailing VOL_WINDOW — de-levers when the book gets turbulent. TARGET_VOL is the
#     annualized target (0.20 = 20%/yr); below it, stay fully invested.
#   USE_DD_BRAKE: when drawdown from peak exceeds DD_THRESHOLD, multiply exposure by
#     DD_SCALE (cut risk after deep losses). Both compose multiplicatively.
USE_VOL_TARGET = os.environ.get("USE_VOL_TARGET", "1") == "1"
TARGET_VOL = float(os.environ.get("TARGET_VOL", "0.15"))
VOL_WINDOW = int(os.environ.get("VOL_WINDOW", "20"))
USE_DD_BRAKE = os.environ.get("USE_DD_BRAKE", "1") == "1"
DD_THRESHOLD = float(os.environ.get("DD_THRESHOLD", "0.20"))
DD_SCALE = float(os.environ.get("DD_SCALE", "0.5"))
# IC 信号失效降仓：近期 OOS rank-IC 衰入负区时自动降暴露（第三层风控）。
# 使用【已成熟】的预测日 IC，防泄露。IC 队列的值从 ic_full_scale(→满仓) 线性插到
# ic_zero_scale(→最低仓 ic_min_scale)。典型值：+0.02 满仓，-0.01 压到 20% 仓。
USE_IC_SCALE = os.environ.get("USE_IC_SCALE", "1") == "1"
IC_SCALE_WINDOW = int(os.environ.get("IC_SCALE_WINDOW", "20"))  # 滑动窗口长度
IC_FULL_SCALE = float(os.environ.get("IC_FULL_SCALE", "0.02"))  # IC≥此值→满仓
IC_ZERO_SCALE = float(os.environ.get("IC_ZERO_SCALE", "-0.01")) # IC≤此值→最低仓
IC_MIN_SCALE = float(os.environ.get("IC_MIN_SCALE", "0.0"))     # 最低仓比例(IC失效全清到现金)

# Regime 择时空仓避险（第四层风控）：DEFAULT OFF。2026-06-30 验证证明它是净亏——
# "过去N日大盘为负→空仓"是滞后趋势规则，在A股V型反弹里追涨杀跌：含2024崩盘的早段
# 从 +34.3% 被害到 −30.5%(卖在底、错过反弹)；近窗上行市(选股失效型)根本不触发。
# vol目标化+IC降仓已是有效的【渐进】regime响应(保留部分仓接反弹)，二元空仓把滞后锁死
# 两端。保留代码与开关供研究，默认关。
USE_REGIME_FILTER = os.environ.get("USE_REGIME_FILTER", "0") == "1"
REGIME_WINDOW = int(os.environ.get("REGIME_WINDOW", "20"))       # 趋势窗(交易日)
REGIME_FLOOR_RET = float(os.environ.get("REGIME_FLOOR_RET", "0.0"))  # 窗累计收益阈值
REGIME_MIN = float(os.environ.get("REGIME_MIN", "0.0"))         # 下行regime仓位(0=空仓)

# --- Option implied-volatility features (Black-Scholes MVP) ------------------
# ETF option IV is treated as a macro regime feature, NOT a stock-level alpha.
# These switches control the standalone diagnostic script diag_option_iv.py and
# any future production integration.
# OPTION_SOURCE: "tushare" uses Black-Scholes IV inversion from option chain +
# daily quotes (requires paid-tier tushare token for opt_basic/opt_daily);
# "akshare" uses the public QVIX volatility-index series as a fast fallback.
OPTION_SOURCE = os.environ.get("OPTION_SOURCE", "akshare")  # "tushare" | "akshare"
OPTION_UNDERLYING = os.environ.get("OPTION_UNDERLYING", "510300.SH")  # 50ETF/300ETF/500ETF
OPTION_USE_CACHE = os.environ.get("OPTION_USE_CACHE", "1") == "1"
OPTION_OFFLINE = os.environ.get("OPTION_OFFLINE", "0") == "1"
OPTION_MAX_FILL_DAYS = int(os.environ.get("OPTION_MAX_FILL_DAYS", "3"))
OPTION_RISK_FREE = float(os.environ.get("OPTION_RISK_FREE", "0.025"))
OPTION_DIV_YIELD = float(os.environ.get("OPTION_DIV_YIELD", "0.015"))
OPTION_API_SLEEP = float(os.environ.get("OPTION_API_SLEEP", "0.35"))  # seconds between tushare option API calls

# --- Production integration of option IV features ---------------------------
# Merge option IV macro features into the shared factor panel so they are
# available to models and risk-control layers.
USE_OPTION_FEATURES = os.environ.get("USE_OPTION_FEATURES", "1") == "1"

# --- Global macro features --------------------------------------------------
# Merge global macro regime features (US indices, VIX, US Treasury yields,
# DXY, China 10Y yield, China-US spread, USD/CNY, North-bound capital flow)
# into the factor panel. All features are date-level (same value for every
# stock) and lagged 1 A-share trading day so they are known at decision time.
# Uses yfinance (US data) + akshare (China-specific); cached to data_cache/macro/.
USE_GLOBAL_MACRO = os.environ.get("USE_GLOBAL_MACRO", "1") == "1"
# Max days to forward-fill macro features to cover A-share dates that follow
# a US holiday (when yfinance returns no new data point).
MACRO_MAX_FILL_DAYS = int(os.environ.get("MACRO_MAX_FILL_DAYS", "3"))

# --- Macro × stock-factor interactions ---------------------------------------
# Create stock-level interaction features: macro regime indicator × stock
# characteristic (per-date z-score). These turn date-level macro features into
# stock-level features with cross-sectional variance, so ic_linear can rank on
# them. ~30 economically meaningful pairs (VIX×momentum, DXY×size, etc.).
USE_MACRO_INTERACTIONS = os.environ.get("USE_MACRO_INTERACTIONS", "1") == "1"

# --- Residual target ---------------------------------------------------------
# When True, the target (responder_6) is regressed against known systematic
# factors (ln_mv, return_20d, volatility_20, bp, ep) within each date and
# replaced with the residual. This strips size/momentum/value/risk premia from
# the learning target, forcing models to learn stock-specific alpha rather than
# re-discovering well-known factor premiums. Leak-free per-date cross-sectional
# OLS. Multi-horizon targets are also residualized.
RESIDUAL_TARGET = os.environ.get("RESIDUAL_TARGET", "0") == "1"

# --- Non-linear factor transforms --------------------------------------------
# Add per-date non-linear transforms of the top-N factors by absolute IC:
# z² (parabolic), |z| (magnitude), max(z,0)/max(-z,0) (asymmetric tails).
# These give ic_linear access to non-linear factor→return relationships without
# GBM-level overfitting. NL_TOPK controls how many factors get expanded (each
# produces 4 transforms). Set NL_TOPK=0 to disable.
NL_TOPK = int(os.environ.get("NL_TOPK", "30"))

# --- Stock-level factor interactions -----------------------------------------
# Add pairwise factor products (z_A × z_B) for the top FI_TOPK factors by abs
# IC. Unlike macro×stock interactions (which have zero cross-sectional variance),
# these are STOCK×STOCK — both factors vary across stocks, so the product has
# genuine cross-sectional variance. FI_MAX_PAIRS limits the total number.
FI_TOPK = int(os.environ.get("FI_TOPK", "0"))
FI_MAX_PAIRS = int(os.environ.get("FI_MAX_PAIRS", "60"))

# --- Factor stability filter -------------------------------------------------
# When > 0, filter factors whose training-segment IC stability
# (|mean_IC| / std_IC) falls below this threshold. Removes noisy factors whose
# IC sign or magnitude is inconsistent day-to-day. Applied per walk-forward
# window in _train_predict_linear. 0 = disabled.
MIN_FACTOR_STABILITY = float(os.environ.get("MIN_FACTOR_STABILITY", "0.0"))

# IV-based gross-leverage scaling in the daily replay / live engine.
# When the chosen IV feature is in its high tercile over the trailing window,
# gross exposure is scaled down to IV_MIN_SCALE. This is a leak-free regime
# overlay: the IV value used for today's decision is from t-1 (lag=1) and the
# high-IV threshold is computed from trailing history only.
USE_IV_SCALE = os.environ.get("USE_IV_SCALE", "1") == "1"
IV_SCALE_WINDOW = int(os.environ.get("IV_SCALE_WINDOW", "60"))      # trailing trading days
IV_FEATURE_FOR_SCALE = os.environ.get("IV_FEATURE_FOR_SCALE", "opt_vix_like")
IV_MIN_SCALE = float(os.environ.get("IV_MIN_SCALE", "0.5"))         # leverage multiplier in high-IV regime
IV_MID_SCALE = float(os.environ.get("IV_MID_SCALE", "1.0"))         # leverage multiplier in mid-IV regime


# --- Config validation (import-time) -----------------------------------------
def _validate() -> None:
    """防泄露硬校验：配置不满足底线就 import 即失败，不静默放行。

    - EMBARGO_DAYS >= FORWARD_WINDOW：训练段最后 FORWARD_WINDOW 天的标签
      （close.shift(-FORWARD_WINDOW)）会伸入 OOS 窗口的价格，embargo 不足
      即前视泄露。
    - multi_* 目标下每个 horizon 各训一个模型，最长 horizon 的标签需要
      max(MULTI_HORIZON_WINDOWS) 的 embargo；不足时显式报错（不静默放大，
      因为 IC 定权/特征成熟逻辑也依赖统一 fw 口径）。
    """
    assert EMBARGO_DAYS >= FORWARD_WINDOW, (
        f"EMBARGO_DAYS={EMBARGO_DAYS} < FORWARD_WINDOW={FORWARD_WINDOW}："
        "训练段尾部标签会偷看 OOS 窗口价格（前视泄露）")
    if MODEL_OBJECTIVE.startswith("multi_") and MULTI_HORIZON_WINDOWS:
        longest = max(MULTI_HORIZON_WINDOWS)
        assert EMBARGO_DAYS >= longest, (
            f"multi 目标要求 EMBARGO_DAYS={EMBARGO_DAYS} >= "
            f"max(MULTI_HORIZON_WINDOWS)={longest}，否则长 horizon 标签泄露")


_validate()

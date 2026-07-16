"""ETF option implied-volatility diagnostic.

Validates whether Black-Scholes-backed option IV features carry incremental
regime information for the existing stock predictor. Steps:
  1. Load named stock factor panel (load_factor_panel).
  2. Fetch/compute lagged ETF option IV features (src.data.option_features).
  3. Evaluate each IV feature as a single-factor predictor of responder_6.
  4. Test the regime hypothesis: compare stock-factor OOS IC on high-IV vs low-IV days.
  5. Save CSVs + a dashboard plot to outputs/.

Usage:
    python diag_option_iv.py --underlying 510300.SH --lag 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from data.option_features import get_option_iv_features
from factor_eval import evaluate_factor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ETF option IV regime-feature diagnostic (Black-Scholes MVP)"
    )
    ap.add_argument(
        "--underlying", type=str, default=C.OPTION_UNDERLYING,
        help="Underlying ETF code (510050.SH / 510300.SH / 510500.SH)",
    )
    ap.add_argument(
        "--start-date", type=str, default=None,
        help="Start date (YYYY-MM-DD); defaults to config START_DATE",
    )
    ap.add_argument(
        "--end-date", type=str, default=None,
        help="End date (YYYY-MM-DD); defaults to config END_DATE",
    )
    ap.add_argument(
        "--lag", type=int, default=1,
        help="Trading days to lag option IV (default 1)",
    )
    ap.add_argument(
        "--regime-factor", type=str, default="opt_vix_like",
        help="IV feature used to define high/low volatility regimes",
    )
    ap.add_argument(
        "--symbols-file", type=str, default=None,
        help="Path to a parquet/CSV with a 'ts_code' column; bypasses universe screening",
    )
    ap.add_argument(
        "--output-dir", type=str, default=C.OUTPUT_DIR,
        help="Directory for CSV/PNG outputs",
    )
    return ap.parse_args()


def load_symbols(path: str | None) -> list[str] | None:
    """Load symbol list from a cached universe file."""
    if not path:
        return None
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "ts_code" not in df.columns:
        raise ValueError(f"symbols file {path} must contain a 'ts_code' column")
    return df["ts_code"].astype(str).tolist()


def merge_iv_features(
    panel: pd.DataFrame,
    iv: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:
    """Merge IV macro features onto the stock panel by date.

    The IV DataFrame already has the feature date shifted by `lag` trading days,
    so a left merge by date aligns option observation t-lag with stock target t.
    """
    # Ensure compatible date types.
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    iv = iv.copy()
    iv["date"] = pd.to_datetime(iv["date"])

    iv_cols = [c for c in iv.columns if c != "date"]
    merged = panel.merge(iv, on="date", how="left")

    # Forward fill IV features up to configured max days.
    if C.OPTION_MAX_FILL_DAYS > 0:
        merged[iv_cols] = merged.groupby("symbol")[iv_cols].ffill(
            limit=C.OPTION_MAX_FILL_DAYS
        )
    return merged


def evaluate_iv_features(panel: pd.DataFrame, iv_cols: list[str]) -> pd.DataFrame:
    """Run evaluation for each IV feature.

    IV features are macro-level (same value for every stock on a given date),
    so the usual cross-sectional RankIC against responder_6 is degenerate.
    Instead we report:
      - time-series correlation with the daily equal-weighted target
        (proxy for market return)
      - time-series correlation with the daily cross-sectional dispersion
        of the target (high-IV regimes often coincide with wider stock
        return dispersions)
    """
    rows = []
    daily = panel.groupby("date").agg(
        target_mean=(C.TARGET, "mean"),
        target_std=(C.TARGET, "std"),
    ).reset_index()

    for col in iv_cols:
        iv_daily = panel.groupby("date")[col].first().reset_index()
        df = daily.merge(iv_daily, on="date").dropna()
        if len(df) < 10:
            rows.append({
                "factor": col, "n_dates": 0, "ts_corr_mkt": np.nan,
                "ts_corr_disp": np.nan, "rank_ic_mean": np.nan,
                "rank_icir": np.nan, "ic_t_stat": np.nan,
            })
            continue

        corr_mkt = stats.spearmanr(df[col], df["target_mean"]).correlation
        corr_disp = stats.spearmanr(df[col], df["target_std"]).correlation

        rows.append({
            "factor": col,
            "n_dates": len(df),
            "ts_corr_mkt": corr_mkt,
            "ts_corr_disp": corr_disp,
            "rank_ic_mean": np.nan,
            "rank_icir": np.nan,
            "ic_t_stat": np.nan,
        })
    return pd.DataFrame(rows)


def _daily_rank_ic(sub: pd.DataFrame, factor: str, target: str) -> pd.Series:
    """Daily cross-sectional Spearman RankIC."""
    def _one(g):
        x, y = g[factor], g[target]
        m = x.notna() & y.notna()
        if m.sum() < 5 or x[m].nunique() < 2 or y[m].nunique() < 2:
            return np.nan
        return stats.spearmanr(x[m], y[m]).correlation
    return sub.groupby("date", group_keys=False).apply(_one, include_groups=False)


def regime_ic_analysis(
    panel: pd.DataFrame,
    iv_cols: list[str],
    stock_factors: list[str],
    regime_factor: str = "opt_vix_like",
) -> pd.DataFrame:
    """Compare daily stock-factor RankIC on high-IV vs low-IV days.

    For each stock factor, compute its daily RankIC, then group days by the
    chosen IV feature into high / medium / low terciles. Report mean RankIC in
    each regime and a t-stat for high-vs-low difference.
    """
    panel = panel.copy()
    for col in iv_cols + stock_factors + [C.TARGET, regime_factor]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    if regime_factor not in panel.columns:
        print(f"⚠️ regime factor {regime_factor} not in panel; skipping regime analysis")
        return pd.DataFrame()

    # Drop dates where regime factor is missing.
    dates = panel[["date", regime_factor]].drop_duplicates().dropna()
    dates["tercile"] = pd.qcut(dates[regime_factor], 3, labels=["low", "mid", "high"])
    regime_map = dates.set_index("date")["tercile"].to_dict()

    rows = []
    for fac in stock_factors:
        if fac not in panel.columns:
            continue
        ic = _daily_rank_ic(panel, fac, C.TARGET).dropna()
        if ic.empty:
            continue
        ic_df = ic.reset_index()
        ic_df.columns = ["date", "ic"]
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df["regime"] = ic_df["date"].map(regime_map)
        ic_df = ic_df.dropna(subset=["regime"])
        if ic_df.empty:
            continue

        grp = ic_df.groupby("regime")["ic"]
        means = grp.mean()
        stds = grp.std()
        counts = grp.count()

        low = ic_df[ic_df["regime"] == "low"]["ic"]
        high = ic_df[ic_df["regime"] == "high"]["ic"]
        diff = np.nan
        tstat = np.nan
        if len(low) > 2 and len(high) > 2:
            diff = high.mean() - low.mean()
            pooled_std = np.sqrt(
                ((len(low) - 1) * low.std(ddof=1) ** 2
                 + (len(high) - 1) * high.std(ddof=1) ** 2)
                / (len(low) + len(high) - 2)
            )
            se = pooled_std * np.sqrt(1 / len(low) + 1 / len(high))
            tstat = diff / se if se > 1e-12 else np.nan

        rows.append({
            "stock_factor": fac,
            "ic_low": means.get("low", np.nan),
            "ic_mid": means.get("mid", np.nan),
            "ic_high": means.get("high", np.nan),
            "n_low": counts.get("low", 0),
            "n_high": counts.get("high", 0),
            "high_minus_low": diff,
            "t_stat": tstat,
        })

    return pd.DataFrame(rows).sort_values("t_stat", key=abs, ascending=False)


def plot_iv_dashboard(
    iv: pd.DataFrame,
    eval_table: pd.DataFrame,
    regime_table: pd.DataFrame,
    out_path: str,
) -> str:
    """Plot IV time series + regime IC comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: ATM IV + VIX-like.
    ax = axes[0]
    ax.plot(iv["date"], iv["opt_iv_atm_mean"], label="ATM mean IV", alpha=0.9)
    if "opt_vix_like" in iv.columns:
        ax.plot(iv["date"], iv["opt_vix_like"], label="VIX-like", alpha=0.8)
    ax.set_title("ETF Option Implied Volatility")
    ax.set_ylabel("IV (annualized)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: term spread + skew.
    ax = axes[1]
    if "opt_iv_term_1m3m" in iv.columns:
        ax.plot(iv["date"], iv["opt_iv_term_1m3m"], label="Term spread (1m/3m)", alpha=0.8)
    if "opt_iv_skew_25d" in iv.columns:
        ax.plot(iv["date"], iv["opt_iv_skew_25d"], label="25Δ skew", alpha=0.8)
    ax.set_title("Option Term Structure & Skew")
    ax.set_ylabel("ratio")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 3: regime IC comparison for top stock factors.
    ax = axes[2]
    if not regime_table.empty:
        top = regime_table.head(8).copy()
        x = np.arange(len(top))
        width = 0.25
        ax.bar(x - width, top["ic_low"], width, label="low IV", alpha=0.8)
        ax.bar(x, top["ic_mid"], width, label="mid IV", alpha=0.8)
        ax.bar(x + width, top["ic_high"], width, label="high IV", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(top["stock_factor"], rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Stock Factor RankIC by IV Regime")
        ax.set_ylabel("mean RankIC")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"ETF Option IV Diagnostic | underlying={args.underlying} | lag={args.lag}")
    print("=" * 70)

    # 1. Load stock factor panel.
    print("\n1) 加载股票因子面板...")
    symbols = load_symbols(args.symbols_file)
    panel = load_factor_panel(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        keep_price=True,
    )
    stock_factors = list_factor_columns(panel)
    print(f"   面板: {len(panel):,} 行, {panel['symbol'].nunique()} 只股票, "
          f"{len(stock_factors)} 个命名因子, "
          f"{panel['date'].nunique()} 个交易日")

    # 2. Fetch option IV features.
    print(f"\n2) 获取 {args.underlying} 期权隐含波动率特征...")
    iv = get_option_iv_features(
        underlying=args.underlying,
        start_date=args.start_date,
        end_date=args.end_date,
        lag=args.lag,
    )
    iv_cols = [c for c in iv.columns if c != "date"]
    print(f"   期权特征: {len(iv)} 交易日 | 列: {iv_cols}")

    # 3. Merge.
    print(f"\n3) 合并 (lag={args.lag} 交易日)...")
    merged = merge_iv_features(panel, iv, args.lag)
    merged = merged.dropna(subset=[C.TARGET]).reset_index(drop=True)
    coverage = merged[iv_cols].notna().mean()
    print(f"   面板合并后: {len(merged):,} 行")
    print("   IV 特征覆盖率:")
    for col, cov in coverage.items():
        print(f"      {col}: {cov:.1%}")

    # 4. Single-factor evaluation.
    print("\n4) 单因子评估 (macro IV 特征: 与日均等权 target / target 横截面离散度的时序秩相关)...")
    eval_table = evaluate_iv_features(merged, iv_cols)
    print(eval_table[[
        "factor", "n_dates", "ts_corr_mkt", "ts_corr_disp"
    ]].to_string(index=False))
    eval_path = os.path.join(args.output_dir, "diag_option_iv_eval.csv")
    eval_table.to_csv(eval_path, index=False)
    print(f"   -> {eval_path}")

    # 5. Regime analysis.
    print(f"\n5) Regime 分析 (按 {args.regime_factor} 分高/中/低三组)...")
    # Use a representative subset of stock factors for regime comparison.
    regime_candidates = [
        "momentum_20", "momentum_5", "volatility_20", "turnover_rate",
        "rsi_14", "price_to_ma_20", "macd_diff",
    ]
    regime_factors = [f for f in regime_candidates if f in stock_factors]
    regime_table = regime_ic_analysis(
        merged, iv_cols, regime_factors, regime_factor=args.regime_factor
    )
    if not regime_table.empty:
        print(regime_table.to_string(index=False))
    regime_path = os.path.join(args.output_dir, "diag_option_iv_regime.csv")
    regime_table.to_csv(regime_path, index=False)
    print(f"   -> {regime_path}")

    # 6. Save IV feature time series.
    iv_path = os.path.join(args.output_dir, "diag_option_iv_features.csv")
    iv.to_csv(iv_path, index=False)
    print(f"\n6) 保存期权特征时间序列 -> {iv_path}")

    # 7. Plot dashboard.
    print("\n7) 绘制 dashboard...")
    plot_path = os.path.join(args.output_dir, "diag_option_iv.png")
    plot_iv_dashboard(iv, eval_table, regime_table, plot_path)
    print(f"   -> {plot_path}")

    # 8. Summary.
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if not eval_table.empty and eval_table["ts_corr_mkt"].notna().any():
        best = eval_table.loc[eval_table["ts_corr_mkt"].abs().idxmax()]
        print(f"与日均等权收益相关性最强的 IV 因子: {best['factor']} | "
              f"ts_corr_mkt={best['ts_corr_mkt']:+.4f} | "
              f"ts_corr_disp={best['ts_corr_disp']:+.4f}")
    else:
        print("无可用的 IV 因子时序相关指标")
    if not regime_table.empty:
        top_reg = regime_table.iloc[0]
        print(f"Regime 效应最大因子: {top_reg['stock_factor']} | "
              f"高IV IC={top_reg['ic_high']:+.4f} | "
              f"低IV IC={top_reg['ic_low']:+.4f} | "
              f"差={top_reg['high_minus_low']:+.4f} (t={top_reg['t_stat']:+.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()

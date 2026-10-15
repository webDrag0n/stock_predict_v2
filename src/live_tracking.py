"""回测 vs 实盘偏差的长期跟踪（数据层）。

每日 nightly 运行产出一行快照（outputs/tracking/live_daily.parquet，同日重跑
覆盖），并重生成分层汇总报告（tracking_report.json）：

  champion   : 当前 champion 身份 + 晋升时的回测预期指标（promotion 记录）
  live       : 实盘实际 —— 活体 OOS rank-IC（pred_hist 成熟打分）+ 真实净值
               表现（live_risk_state.json，依赖 account.csv 维护）
  divergence : 预期 vs 实际的逐项差值
  alerts     : 活跃告警（IC 退化 / 收益落后 / 方向背离 / 临近超龄）

无真实净值（account.csv 未维护）时优雅降级：净值相关字段为 None，只跟踪
IC 维度，并在告警里醒目提示。重训触发（retrain_trigger）消费本模块的指标；
本模块只做采集/汇总/告警，不做决策。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config as C
from live_state import collect_matured_ics, load_risk_state

_SNAPSHOT_FILE = "live_daily.parquet"
_REPORT_FILE = "tracking_report.json"


# ---------------------------------------------------------------------------
# 指标采集
# ---------------------------------------------------------------------------
def matured_ic_stats(panel: pd.DataFrame, pred_hist: pd.DataFrame,
                     window: int | None = None) -> dict:
    """截至面板最新日的活体 OOS IC 统计（成熟口径与 IC 降仓层一致）。"""
    window = int(window or getattr(C, "RETRAIN_IC_WINDOW", 20))
    ics: dict = {}
    if pred_hist is not None and len(pred_hist) and len(panel):
        latest = panel["date"].max()
        fw = int(getattr(C, "FORWARD_WINDOW", 20))
        ics, _ = collect_matured_ics(panel, pred_hist, latest, fw)
    vals = pd.Series(ics, dtype=float).sort_index() if ics else pd.Series(dtype=float)
    tail = vals.iloc[-window:] if len(vals) else vals
    return {
        "ics": {str(k): float(v) for k, v in ics.items()},
        "n_ic": int(len(vals)),
        "ic_mean": float(vals.mean()) if len(vals) else None,
        "ic_mean_w": float(tail.mean()) if len(tail) else None,
        "window": window,
    }


def nav_stats(risk_state_path: str, panel: pd.DataFrame,
              window: int | None = None) -> dict:
    """实盘净值表现（尾部 window 日）：收益/Sharpe/回撤 + 等权基准对齐。

    净值来自 live_risk_state.json 的 dates/total_values（由 account.csv 真实
    现金驱动）；不足 2 个净值点视为无净值（has_nav=False），全部指标 None。
    """
    window = int(window or getattr(C, "RETRAIN_UNDERPERF_DAYS", 20))
    out = {"has_nav": False, "n_nav": 0, "window": window,
           "total_value": None, "live_ret_1d": None, "bench_ret_1d": None,
           "live_ret_w": None, "bench_ret_w": None, "sharpe_w": None,
           "max_dd_w": None}
    state = load_risk_state(risk_state_path)
    tvs = [float(v) for v in state.total_values]
    dts = [str(d) for d in state.dates]
    if len(tvs) < 2:
        return out
    out["has_nav"] = True
    out["n_nav"] = len(tvs)
    out["total_value"] = tvs[-1]

    tv = pd.Series(tvs, index=pd.Index(dts, name="date"), dtype=float)
    rets = tv.pct_change().dropna()
    out["live_ret_1d"] = float(rets.iloc[-1]) if len(rets) else None

    # 基准：全市场等权日收益，对齐到净值日期
    bench = None
    ret_col = ("return_1d" if "return_1d" in panel.columns
               else ("pct_chg" if "pct_chg" in panel.columns else None))
    if ret_col is not None and len(panel):
        bser = panel.groupby("date")[ret_col].mean()
        if ret_col == "pct_chg":
            bser = bser / 100.0
        bser.index = bser.index.astype(str)
        bench = bser.reindex(rets.index).fillna(0.0)
        out["bench_ret_1d"] = float(bench.iloc[-1]) if len(bench) else None

    rw = rets.iloc[-window:]
    if len(rw) >= 2:
        out["live_ret_w"] = float((1.0 + rw).prod() - 1.0)
        out["sharpe_w"] = (float(rw.mean() / rw.std() * np.sqrt(252))
                           if rw.std() > 1e-12 else 0.0)
        cum = (1.0 + rw).cumprod()
        out["max_dd_w"] = float(((cum - cum.cummax()) / cum.cummax()).min())
        if bench is not None:
            bw = bench.iloc[-len(rw):]
            out["bench_ret_w"] = float((1.0 + bw).prod() - 1.0)
    return out


# ---------------------------------------------------------------------------
# 每日快照（append-only parquet，同日覆盖）
# ---------------------------------------------------------------------------
def snapshot_path(tracking_dir: str | None = None) -> str:
    d = tracking_dir or getattr(C, "TRACKING_DIR",
                                os.path.join(C.OUTPUT_DIR, "tracking"))
    return os.path.join(d, _SNAPSHOT_FILE)


def report_path(tracking_dir: str | None = None) -> str:
    d = tracking_dir or getattr(C, "TRACKING_DIR",
                                os.path.join(C.OUTPUT_DIR, "tracking"))
    return os.path.join(d, _REPORT_FILE)


def load_daily(tracking_dir: str | None = None) -> pd.DataFrame:
    path = snapshot_path(tracking_dir)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[警告] 跟踪快照 {path} 读取失败（{type(e).__name__}: {e}），"
              "按空历史处理。")
        return pd.DataFrame()


def append_daily_snapshot(date: object, model_id: str, action: str,
                          n_pred: int, ic_stats: dict, nav: dict,
                          tracking_dir: str | None = None) -> pd.DataFrame:
    """追加当日快照行（同日重跑覆盖）。返回更新后的全表。"""
    row = {
        "date": str(pd.Timestamp(date).date()),
        "model_id": model_id,
        "action": action,
        "n_pred": int(n_pred),
        "n_ic_total": ic_stats.get("n_ic", 0),
        "ic_mean_20": ic_stats.get("ic_mean_w"),
        "live_total_value": nav.get("total_value"),
        "live_ret_1d": nav.get("live_ret_1d"),
        "bench_ret_1d": nav.get("bench_ret_1d"),
        "live_ret_20d": nav.get("live_ret_w"),
        "bench_ret_20d": nav.get("bench_ret_w"),
    }
    df = load_daily(tracking_dir)
    if len(df):
        df = df[df["date"] != row["date"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path = snapshot_path(tracking_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# 汇总报告 + 告警
# ---------------------------------------------------------------------------
def compute_report(champion_manifest: dict | None, ic_stats: dict,
                   nav: dict, panel: pd.DataFrame | None = None) -> dict:
    """回测预期 vs 实盘实际的分层报告（含 divergence 与 alerts）。"""
    champion_manifest = champion_manifest or {}
    promo = champion_manifest.get("promotion") or {}
    bt_metrics = promo.get("candidate_metrics") or {}

    live = {
        "n_ic": ic_stats.get("n_ic", 0),
        "ic_mean": ic_stats.get("ic_mean"),
        "ic_mean_w": ic_stats.get("ic_mean_w"),
        "ic_window": ic_stats.get("window"),
        "has_nav": nav.get("has_nav", False),
        "n_nav": nav.get("n_nav", 0),
        "window": nav.get("window"),
        "live_ret_w": nav.get("live_ret_w"),
        "bench_ret_w": nav.get("bench_ret_w"),
        "sharpe_w": nav.get("sharpe_w"),
        "max_dd_w": nav.get("max_dd_w"),
    }
    divergence = {
        # 回测预期 Sharpe（晋升评估窗口）vs 实盘近窗口 Sharpe（有净值时）
        "sharpe_gap": (live["sharpe_w"] - bt_metrics["sharpe"]
                       if live["sharpe_w"] is not None
                       and bt_metrics.get("sharpe") is not None else None),
        "ret_gap_vs_bench": (live["live_ret_w"] - live["bench_ret_w"]
                             if live["live_ret_w"] is not None
                             and live["bench_ret_w"] is not None else None),
    }

    alerts = []
    w = ic_stats.get("window", 20)
    min_samples = int(getattr(C, "RETRAIN_IC_MIN_SAMPLES", 10))
    ic_min = float(getattr(C, "RETRAIN_IC_MIN", 0.0))
    if (ic_stats.get("n_ic", 0) >= min_samples
            and ic_stats.get("ic_mean_w") is not None
            and ic_stats["ic_mean_w"] < ic_min):
        alerts.append(f"活体 IC 退化：尾部{w}日 OOS IC 均值 "
                      f"{ic_stats['ic_mean_w']:+.4f} < {ic_min:+.4f}"
                      f"（{ic_stats['n_ic']} 个成熟样本）")
    margin = float(getattr(C, "RETRAIN_UNDERPERF_MARGIN", 0.03))
    if (divergence["ret_gap_vs_bench"] is not None
            and divergence["ret_gap_vs_bench"] < -margin):
        alerts.append(f"实盘落后基准：{nav.get('window', 20)} 日实盘 "
                      f"{live['live_ret_w']:+.1%} vs 基准 "
                      f"{live['bench_ret_w']:+.1%}"
                      f"（差距 {divergence['ret_gap_vs_bench']:+.1%}，"
                      f"容忍 {-margin:.1%}）")
    if (live["sharpe_w"] is not None and live["sharpe_w"] < 0
            and bt_metrics.get("sharpe") is not None
            and bt_metrics["sharpe"] > 0.5):
        alerts.append(f"回测实盘方向背离：晋升回测 Sharpe "
                      f"{bt_metrics['sharpe']:+.2f}，实盘近{nav.get('window', 20)}日 "
                      f"Sharpe {live['sharpe_w']:+.2f}——回测可能过拟合或分布漂移")
    if not nav.get("has_nav"):
        alerts.append("无真实净值（account.csv 未维护 cash 列）：净值维度跟踪"
                      "与收益类告警未启用，仅跟踪 IC 维度")

    age = None
    if panel is not None and len(panel):
        from retrain_trigger import model_age_days
        age = model_age_days(champion_manifest, panel)
    max_age = int(getattr(C, "MAX_MODEL_AGE_DAYS", 60))
    if age is not None and age > max_age * 0.8:
        alerts.append(f"模型临近超龄：已 {age} 个交易日未重训"
                      f"（{max_age} 天强制重训）")

    return {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "champion": {
            "model_id": champion_manifest.get("model_id"),
            "train_end_date": champion_manifest.get("train_end_date"),
            "age_days": age,
            "promoted_at": promo.get("decided_at"),
            "promotion_reason": promo.get("reason"),
            "backtest_expectation": bt_metrics or None,
            "eval_window": promo.get("eval_window"),
        },
        "live": live,
        "divergence": divergence,
        "alerts": alerts,
    }


def write_report(report: dict, tracking_dir: str | None = None) -> str:
    path = report_path(tracking_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)
    return path


def print_summary(report: dict) -> None:
    """nightly stdout 的 2~3 行跟踪摘要。"""
    ch, lv = report["champion"], report["live"]
    age = ch.get("age_days")
    ic = lv.get("ic_mean_w")
    print("[跟踪] champion={mid}（年龄 {age} 天）· 活体 IC{w}日均值 {ic}".format(
        mid=ch.get("model_id"), age=age if age is not None else "?",
        w=lv.get("ic_window") or "",
        ic=f"{ic:+.4f}" if ic is not None else "样本不足"))
    if lv.get("has_nav") and lv.get("live_ret_w") is not None:
        print(f"[跟踪] 近{lv.get('window', 20)}日实盘 {lv['live_ret_w']:+.1%} "
              f"vs 基准 {lv['bench_ret_w']:+.1%} · Sharpe {lv['sharpe_w']:+.2f}")
    for a in report["alerts"]:
        print(f"[跟踪·告警] {a}")

"""滚动走动回测 CLI —— 严格防泄露的样本外验证。

用法
----
  python run_walkforward.py                 # 全特征滚动回测
  python run_walkforward.py --select-features  # 每窗口在训练段内做因子筛选
  python run_walkforward.py --compare       # 两者对比
  python run_walkforward.py --engine incremental  # 线上同款逐日增量集成

对比上一轮单次 60 天窗口（Sharpe 14.5，疑似窗口运气 + 因子选择泄露），本脚本
拼接多个不重叠样本外窗口，得到跨 1~2 年的连续 OOS 净值与各窗口指标分布。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import pandas as pd

# pandas 碎片化告警（PerformanceWarning）属构建期噪声；入口脚本显式屏蔽
# （库模块不再在 import 期动全局 warnings 状态）。
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward
from perf_metrics import format_metrics
from capital import with_capital_profile


def _summarize(tag: str, res: dict) -> None:
    m = res["oos_result"]["metrics"]
    wm = res["window_metrics"]
    print(f"\n===== {tag} | 连续样本外汇总（{res['n_windows']} 个窗口拼接）=====")
    print(format_metrics(m))
    print("各窗口 OOS 指标分布（稳健性）：")
    print(
        f"  Sharpe: 均值={wm['Sharpe'].mean():+.2f} 中位={wm['Sharpe'].median():+.2f} "
        f"最差={wm['Sharpe'].min():+.2f} 最好={wm['Sharpe'].max():+.2f} "
        f"为正窗口={int((wm['Sharpe']>0).sum())}/{len(wm)}"
    )
    print(f"  PnL:    均值={wm['PnL'].mean():+.4f} 为正窗口={int((wm['PnL']>0).sum())}/{len(wm)}")
    print(f"  胜率:   均值={wm['hit'].mean():.1%}")
    if "pred_ic" in m and m["pred_ic"] == m["pred_ic"]:
        wmean = wm["PredIC"].mean() if "PredIC" in wm else float("nan")
        npos = int((wm["PredIC"] > 0).sum()) if "PredIC" in wm else 0
        print(
            f"  预测OOS IC（精度核心指标）: 连续={m['pred_ic']:+.4f} "
            f"ICIR={m['pred_icir']:+.3f} | 各窗口均值={wmean:+.4f} "
            f"为正窗口={npos}/{len(wm)}"
        )
    # 「为正窗口占比」门禁信号（项目铁律）：Sharpe / PredIC 两个占比显式给出，
    # 任一不足 50% 打醒目警告（alpha 不成立时正收益不可信）。
    sharpe_ratio = res.get("sharpe_pos_ratio")
    predic_ratio = res.get("predic_pos_ratio")
    if sharpe_ratio is None and len(wm):
        sharpe_ratio = float((wm["Sharpe"] > 0).mean())
    if predic_ratio is None and "PredIC" in wm and len(wm):
        _v = wm["PredIC"].dropna()
        if len(_v):
            predic_ratio = float((_v > 0).mean())
    if sharpe_ratio is not None:
        print(f"  Sharpe 为正窗口占比: {sharpe_ratio:.0%}")
    if predic_ratio is not None:
        print(f"  PredIC 为正窗口占比: {predic_ratio:.0%}")
        if predic_ratio < 0.5:
            print("  ⚠️⚠️ PredIC 为正窗口占比不足 50%——选股 alpha 不成立，正收益不可信 ⚠️⚠️")
    if sharpe_ratio is not None and sharpe_ratio < 0.5:
        print("  ⚠️⚠️ Sharpe 为正窗口占比不足 50%——信号跨窗口稳健性不足 ⚠️⚠️")
    if "mean_price_rmse" in m and m["mean_price_rmse"] == m["mean_price_rmse"]:
        print(
            f"  预测价格RMSE: 连续OOS均值=¥{m['mean_price_rmse']:.4f}"
            + (f" | 各窗口均值=¥{wm['PriceRMSE'].mean():.4f}" if "PriceRMSE" in wm else "")
        )


def _null_run():
    """RUN_REGISTRY=0 时的空登记器：主流程不必处处判空。"""
    from types import SimpleNamespace

    noop = lambda *a, **k: None  # noqa: E731
    return SimpleNamespace(
        step=noop,
        step_done=noop,
        artifact=noop,
        note=noop,
        set_param=noop,
        callback=lambda: None,
        finish=noop,
        fail=noop,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="滚动走动回测（防泄露样本外验证）")
    ap.add_argument(
        "--select-features",
        action="store_true",
        help="每窗口在训练段内做因子准入筛选（隔离，无泄露）",
    )
    ap.add_argument("--compare", action="store_true", help="对比 全特征 vs 窗口内因子选择")
    ap.add_argument("--strategy", default="long_topk")
    ap.add_argument("--engine", choices=["windowed", "incremental"], default="windowed")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument(
        "--capital-profile",
        choices=["standard", "small"],
        default=os.environ.get("CAPITAL_PROFILE", "standard"),
        help="small：少仓、较大单笔、最小建仓金额与现金校验",
    )
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--max-weight", type=float, default=None)
    ap.add_argument("--min-notional", type=float, default=None)
    ap.add_argument("--min-entry-notional", type=float, default=None)
    ap.add_argument("--cash-aware", action="store_true")
    ap.add_argument("--cash-buffer", type=float, default=None)
    ap.add_argument("--allowed-boards", default="")
    ap.add_argument("--skip-unaffordable", action="store_true")
    args = ap.parse_args()

    # 登记到 outputs/runs/：前端只读该目录即可看到状态与进度（RUN_REGISTRY=0 关闭）。
    # 登记是旁路，RunReporter 写盘失败只降级，不影响回测本身。
    if os.environ.get("RUN_REGISTRY", "1") == "0":
        run = _null_run()
    else:
        from run_registry import RunReporter

        run = RunReporter(
            "walkforward",
            label=f"滚动 walk-forward · {args.engine}"
            + (" · 因子选择" if args.select_features else ""),
            steps=["load_panel", "walkforward"],
            params={k: v for k, v in vars(args).items() if v not in (None, "")},
            run_id=os.environ.get("RUN_ID") or None,
            parent=os.environ.get("RUN_PARENT") or None,
        )
    try:
        _run(args, run)
    except SystemExit as exc:
        run.finish("done" if exc.code in (None, 0) else "failed", exit_code=exc.code or 0)
        raise
    except KeyboardInterrupt:
        run.finish("killed", exit_code=130)
        raise
    except BaseException as exc:
        run.fail(exc)
        raise
    else:
        run.finish("done", exit_code=0)


def _run(args, run) -> None:
    if args.capital is not None:
        C.INITIAL_CAPITAL = float(args.capital)
    replay_params = {}
    for key in ("max_positions", "max_weight", "min_notional", "min_entry_notional", "cash_buffer"):
        value = getattr(args, key, None)
        if value is not None:
            replay_params[key] = value
    if args.allowed_boards:
        replay_params["allowed_boards"] = [
            board.strip() for board in args.allowed_boards.split(",") if board.strip()
        ]
    if args.skip_unaffordable:
        replay_params["skip_unaffordable"] = True
    if getattr(args, "cash_aware", False):
        replay_params["cash_aware"] = True
    replay_params = with_capital_profile(getattr(args, "capital_profile", "standard"), replay_params)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print("加载命名因子面板...")
    run.step("load_panel", "加载因子面板")
    # keep_trade_cols=True：每日重放引擎（唯一回测口径）需要 OHLC/布林带/涨跌停
    # 等撮合列（close 随之保留，keep_price 不再需要单独开）。
    panel = load_factor_panel(keep_trade_cols=True)
    factors = list_factor_columns(panel)
    print(
        f"候选因子 {len(factors)} 个 | 训练窗口 {C.TRAIN_WINDOW} | "
        f"测试窗口 {C.TEST_WINDOW} | embargo {C.EMBARGO_DAYS} | step {C.WF_STEP}\n"
    )
    run.step_done("load_panel", f"{len(panel):,} 行 · {len(factors)} 因子")
    run.step("walkforward", "滚动 walk-forward")

    runs = []
    if args.compare:
        runs = [("全特征", False), ("窗口内因子选择", True)]
    else:
        runs = [("窗口内因子选择" if args.select_features else "全特征", args.select_features)]

    results = {}
    for tag, sel in runs:
        print(f"--- 运行: {tag} ---")
        res = run_walkforward(
            panel,
            factors,
            select_features=sel,
            strategy=args.strategy,
            engine=args.engine,
            replay_params=replay_params,
            progress_cb=run.callback(),
        )
        results[tag] = res
        _summarize(tag, res)
        # canonical 产物由 run_walkforward 内部 save_walkforward_result 落盘
        # （outputs/walkforward/：oos_curve / window_metrics / oos_bench /
        # meta.json，前端会话页直接读取展示）；--compare 两次运行共用该目录，
        # 后者覆盖前者。
        print(f"walkforward 结果 -> {os.path.join(C.OUTPUT_DIR, 'walkforward')}")
        run.artifact(
            "walkforward", os.path.join(C.OUTPUT_DIR, "walkforward", "meta.json"), kind="report"
        )
        _m = res["oos_result"]["metrics"]
        run.set_param(
            **{
                f"result_{tag}": {
                    "sharpe": _m.get("sharpe"),
                    "pred_ic": _m.get("pred_ic"),
                    "nw_t": _m.get("nw_t"),
                    "n_windows": res.get("n_windows"),
                    "sharpe_pos_ratio": res.get("sharpe_pos_ratio"),
                }
            }
        )
    run.step_done("walkforward", f"{len(results)} 组完成")

    print(
        "\n说明：连续样本外 Sharpe 通常远低于单窗口数字——这才是可信的真实水平。"
        "重点看「为正窗口占比」，它衡量信号能否跨时间稳健复现。"
    )


if __name__ == "__main__":
    main()

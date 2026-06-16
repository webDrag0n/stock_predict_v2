"""滚动走动回测 CLI —— 严格防泄露的样本外验证。

用法
----
  python run_walkforward.py                 # 全特征滚动回测
  python run_walkforward.py --select-features  # 每窗口在训练段内做因子筛选
  python run_walkforward.py --compare       # 两者对比

对比上一轮单次 60 天窗口（Sharpe 14.5，疑似窗口运气 + 因子选择泄露），本脚本
拼接多个不重叠样本外窗口，得到跨 1~2 年的连续 OOS 净值与各窗口指标分布。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import run_walkforward, plot_dashboard
from backtest import format_metrics, plot_results


def _summarize(tag: str, res: dict) -> None:
    m = res["oos_result"]["metrics"]
    wm = res["window_metrics"]
    print(f"\n===== {tag} | 连续样本外汇总（{res['n_windows']} 个窗口拼接）=====")
    print(format_metrics(m))
    print("各窗口 OOS 指标分布（稳健性）：")
    print(f"  Sharpe: 均值={wm['Sharpe'].mean():+.2f} 中位={wm['Sharpe'].median():+.2f} "
          f"最差={wm['Sharpe'].min():+.2f} 最好={wm['Sharpe'].max():+.2f} "
          f"为正窗口={int((wm['Sharpe']>0).sum())}/{len(wm)}")
    print(f"  PnL:    均值={wm['PnL'].mean():+.4f} 为正窗口={int((wm['PnL']>0).sum())}/{len(wm)}")
    print(f"  胜率:   均值={wm['hit'].mean():.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="滚动走动回测（防泄露样本外验证）")
    ap.add_argument("--select-features", action="store_true",
                    help="每窗口在训练段内做因子准入筛选（隔离，无泄露）")
    ap.add_argument("--compare", action="store_true",
                    help="对比 全特征 vs 窗口内因子选择")
    ap.add_argument("--strategy", default="long_topk")
    args = ap.parse_args()

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print("加载命名因子面板...")
    panel = load_factor_panel()
    factors = list_factor_columns(panel)
    print(f"候选因子 {len(factors)} 个 | 训练窗口 {C.TRAIN_WINDOW} | "
          f"测试窗口 {C.TEST_WINDOW} | embargo {C.EMBARGO_DAYS} | step {C.WF_STEP}\n")

    runs = []
    if args.compare:
        runs = [("全特征", False), ("窗口内因子选择", True)]
    else:
        runs = [("窗口内因子选择" if args.select_features else "全特征",
                 args.select_features)]

    results = {}
    for tag, sel in runs:
        print(f"--- 运行: {tag} ---")
        res = run_walkforward(panel, factors, select_features=sel,
                              strategy=args.strategy)
        results[tag] = res
        _summarize(tag, res)
        safe = "selected" if sel else "allfeat"
        fig = plot_results(res["oos_result"]["curve"], res["oos_result"]["metrics"],
                           os.path.join(C.OUTPUT_DIR, f"walkforward_{safe}.png"))
        dash = plot_dashboard(res, os.path.join(C.OUTPUT_DIR, f"dashboard_{safe}.png"),
                              tag=tag)
        res["window_metrics"].to_csv(
            os.path.join(C.OUTPUT_DIR, f"walkforward_windows_{safe}.csv"), index=False)
        print(f"OOS 净值曲线 -> {fig}")
        print(f"综合仪表盘（含金额盈亏曲线）-> {dash}")

    print("\n说明：连续样本外 Sharpe 通常远低于单窗口数字——这才是可信的真实水平。"
          "重点看「为正窗口占比」，它衡量信号能否跨时间稳健复现。")


if __name__ == "__main__":
    main()

"""单因子评估 CLI：对现有因子跑 IC / ICIR / 分层回测，输出准入排名表。

用法
----
  python eval_factors.py                 # 评估全部命名因子
  python eval_factors.py --layers 10     # 10 层分层回测
  python eval_factors.py --csv out.csv   # 额外导出 CSV

这是持续因子优化的第一步：只有通过准入门槛（RankIC/RankICIR/单调性）的
因子才应进入模型。复用 load_data 的原始K线缓存，不产生额外 API 调用。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from factor_eval import evaluate_panel, format_panel_report, plot_layered_returns


def main() -> None:
    ap = argparse.ArgumentParser(description="单因子有效性评估")
    ap.add_argument("--layers", type=int, default=5, help="分层回测层数")
    ap.add_argument("--csv", default=None, help="导出汇总表到 CSV 路径")
    ap.add_argument("--plot", action="store_true",
                    help="为 |IC_t| 最高的前 6 个因子画分层累计净值曲线")
    args = ap.parse_args()

    print("加载因子面板（保留因子原名）...")
    panel = load_factor_panel()
    factors = list_factor_columns(panel)
    print(f"待评估因子 {len(factors)} 个 | 预测窗口 {C.FORWARD_WINDOW} 日 | "
          f"目标 {'横截面超额' if C.MARKET_NEUTRAL_TARGET else '原始'}收益\n")

    table = evaluate_panel(panel, factors, n_layers=args.layers)
    print(format_panel_report(table))

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    out = args.csv or os.path.join(C.OUTPUT_DIR, "factor_eval.csv")
    table.to_csv(out, index=False)
    print(f"\n汇总表已导出 -> {out}")

    if args.plot and not table.empty:
        top = table.head(6)["factor"].tolist()
        fig = plot_layered_returns(
            panel, top, os.path.join(C.OUTPUT_DIR, "factor_layers.png"),
            n_layers=args.layers)
        print(f"分层净值曲线已导出 -> {fig}")

    # 给出对模型特征选择的建议
    if not table.empty:
        passed = table[table["准入"] == "✓"]["factor"].tolist()
        print(f"\n建议：{len(passed)} 个因子通过准入，可优先保留进入模型：")
        print("  " + (", ".join(passed) if passed else "（无因子达标——信号普遍偏弱，"
              "考虑引入基本面因子或调整预测窗口）"))


if __name__ == "__main__":
    main()

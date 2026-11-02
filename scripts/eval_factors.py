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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from factor_eval import evaluate_panel, format_panel_report


def main() -> None:
    ap = argparse.ArgumentParser(description="单因子有效性评估")
    ap.add_argument("--layers", type=int, default=5, help="分层回测层数")
    ap.add_argument("--csv", default=None, help="导出汇总表到 CSV 路径")
    ap.add_argument(
        "--dump-layers",
        choices=["none", "passing", "all"],
        default="passing",
        help="分层净值/逐日RankIC 落盘范围（供前端因子详情 P2/P3 面板；"
        "产物 outputs/factor_layer_<id>.parquet 与 factor_ic_daily_<id>.parquet）",
    )
    args = ap.parse_args()

    print("加载因子面板（保留因子原名）...")
    panel = load_factor_panel()
    factors = list_factor_columns(panel)
    print(
        f"待评估因子 {len(factors)} 个 | 预测窗口 {C.FORWARD_WINDOW} 日 | "
        f"目标 {'横截面超额' if C.MARKET_NEUTRAL_TARGET else '原始'}收益\n"
    )

    table = evaluate_panel(panel, factors, n_layers=args.layers)
    print(format_panel_report(table))

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    out = args.csv or os.path.join(C.OUTPUT_DIR, "factor_eval.csv")
    table.to_csv(out, index=False)
    print(f"\n汇总表已导出 -> {out}")

    # 分层净值落盘（设计 5.4 契约①：前端因子详情 P2/P3 面板数据源）
    if args.dump_layers != "none" and not table.empty:
        from factor_layers import dump_layers

        sel = (
            table["factor"].tolist()
            if args.dump_layers == "all"
            else table[table["准入"] == "✓"]["factor"].tolist()
        )
        summary = dump_layers(panel, sel, C.OUTPUT_DIR, n_layers=args.layers)
        n_ok = int(summary["ok"].sum())
        print(
            f"分层净值已落盘 {n_ok}/{len(sel)} 个因子 -> {C.OUTPUT_DIR}/factor_layer_<id>.parquet"
        )
        for _, r in summary[~summary["ok"]].iterrows():
            print(f"  [跳过] {r['factor']}: {r['error']}")

    # 给出对模型特征选择的建议
    if not table.empty:
        passed = table[table["准入"] == "✓"]["factor"].tolist()
        print(f"\n建议：{len(passed)} 个因子通过准入，可优先保留进入模型：")
        print(
            "  "
            + (
                ", ".join(passed)
                if passed
                else "（无因子达标——信号普遍偏弱，" "考虑引入基本面因子或调整预测窗口）"
            )
        )


if __name__ == "__main__":
    main()

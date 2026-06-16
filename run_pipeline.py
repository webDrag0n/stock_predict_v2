"""端到端管线：训练 -> 预测 -> 策略 -> 回测（A股 / tushare / 沪深300）。

切分方式（三段，时序）：
    train 段 —— 训练集成模型
    val   段 —— 选择策略（比较各策略，挑验证集表现最好者）
    test  段 —— 仅对选中的策略做最终样本外汇报

在 test 段挑选最优策略属于数据窥探偏差，会高估表现；因此选择只在 val 段进行。

用法
----
  python run_pipeline.py                  # 自动选策略（val选, test报）
  python run_pipeline.py --strategy long_topk   # 指定策略，跳过自动选择
  python run_pipeline.py --compare        # 在 test 段展示所有策略（仅供研究对比）
  python run_pipeline.py --no-train       # 复用 models/ 中已训练模型
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C
from data import load_data, split_dates
from model import EnsembleModel
from strategy import generate_positions, ALL_STRATEGIES, TRADEABLE
from backtest import run_backtest, format_metrics, plot_results


def _backtest_on(df, dates, model, strat):
    seg = df.loc[df[C.DATE_COL].isin(dates)].reset_index(drop=True)
    pred = model.predict(seg)
    pos = generate_positions(seg, pred, strategy=strat)
    res = run_backtest(seg, pred, pos)
    res["segment"] = seg
    res["pred"] = pred
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="A股预测 + 策略 + 回测")
    ap.add_argument("--strategy", default=None, choices=list(ALL_STRATEGIES))
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="在 test 段展示所有策略（仅研究对比）")
    ap.add_argument("--tradeable-only", action="store_true")
    ap.add_argument("--select-features", action="store_true",
                    help="只用通过因子准入门槛的特征（需先跑 eval_factors.py）")
    args = ap.parse_args()

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    # 1) 数据：三段切分 ----------------------------------------------------
    df, is_real = load_data()
    train_dates, val_dates, test_dates = split_dates(df)
    print(f"切分 -> 训练:{len(train_dates)} 验证:{len(val_dates)} 测试:{len(test_dates)} 个交易日")

    # 2) 模型 --------------------------------------------------------------
    model_path = os.path.join(C.MODEL_DIR, "ensemble.joblib")
    if args.no_train and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = EnsembleModel.load(model_path)
    else:
        feature_cols = None
        if args.select_features:
            from factor_eval import select_features
            feature_cols = select_features()   # None if no eval CSV or no passing factors
        print("训练集成模型（按日期 purged CV）...")
        model = EnsembleModel(feature_cols=feature_cols).fit(df, train_dates, val_dates)
        print(f"已保存模型 -> {model.save(model_path)}")

    # 3) 策略选择：在 VAL 段比较 ------------------------------------------
    if args.strategy:
        chosen = args.strategy
        print(f"\n使用指定策略: {chosen}（跳过验证集自动选择）")
        val_results = {}
    else:
        candidates = TRADEABLE if args.tradeable_only else ALL_STRATEGIES
        print(f"\n=== 在验证集上选择策略（候选: {', '.join(candidates)}）===")
        val_results = {}
        for strat in candidates:
            res = _backtest_on(df, val_dates, model, strat)
            val_results[strat] = res
            m = res["metrics"]
            print(f"  {strat:<16} Sharpe={m['sharpe']:+.3f}  "
                  f"PnL={m['total_pnl']:+.4f}  MaxDD={m['max_drawdown']:+.4f}")
        # 选验证集 Sharpe 最高者。
        chosen = max(val_results, key=lambda s: val_results[s]["metrics"]["sharpe"])
        note = "" if chosen in TRADEABLE else "（注意：非 A股可实盘策略，仅研究）"
        print(f"选中策略: {chosen} {note}")

    # 4) 最终汇报：仅在 TEST 段评估选中的策略 -----------------------------
    print(f"\n=== 测试集最终评估（样本外）: {chosen} ===")
    final = _backtest_on(df, test_dates, model, chosen)
    print(format_metrics(final["metrics"]))
    fig = plot_results(final["curve"], final["metrics"],
                       os.path.join(C.OUTPUT_DIR, f"equity_{chosen}.png"))
    print(f"已保存净值曲线 -> {fig}")

    # 5) 可选：test 段展示所有策略（仅研究对比，非选择依据）---------------
    test_all = {}
    if args.compare:
        print("\n=== 测试集各策略对比（仅供研究，已包含数据窥探风险提示）===")
        hdr = f"{'策略':<16}{'PnL':>10}{'Sharpe':>9}{'MaxDD':>9}{'胜率':>8}  可交易"
        print(hdr); print("-" * (len(hdr) + 4))
        for strat in ALL_STRATEGIES:
            res = test_all.get(strat) or _backtest_on(df, test_dates, model, strat)
            test_all[strat] = res
            m = res["metrics"]
            tradeable = "✓" if strat in TRADEABLE else "研究"
            print(f"{strat:<16}{m['total_pnl']:>+10.3f}{m['sharpe']:>+9.2f}"
                  f"{m['max_drawdown']:>+9.3f}{m['hit_rate']:>8.1%}  {tradeable}")

    # 6) 落盘：预测、持仓、报告 -------------------------------------------
    seg, pred = final["segment"], final["pred"]
    out = seg[[C.DATE_COL, C.TIME_COL, C.SYMBOL_COL, C.TARGET, C.WEIGHT]].copy()
    out["prediction"] = pred
    out["position"] = generate_positions(seg, pred, strategy=chosen)
    out["net_pnl"] = final["net_pnl"]
    pred_csv = os.path.join(C.OUTPUT_DIR, "predictions.csv")
    out.to_csv(pred_csv, index=False)

    report = {
        "data_source": C.DATA_SOURCE,
        "index": C.DEFAULT_INDEX,
        "n_rows": int(len(df)),
        "split": {"train": len(train_dates), "val": len(val_dates),
                  "test": len(test_dates)},
        "forward_window": C.FORWARD_WINDOW,
        "market_neutral_target": C.MARKET_NEUTRAL_TARGET,
        "cost": {"commission": C.COMMISSION_RATE, "stamp_duty": C.STAMP_DUTY_RATE},
        "chosen_strategy": chosen,
        "chosen_is_tradeable": chosen in TRADEABLE,
        "test_metrics": final["metrics"],
        "val_selection": {s: val_results[s]["metrics"] for s in val_results},
        "test_all": {s: test_all[s]["metrics"] for s in test_all},
    }
    report_path = os.path.join(C.OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n选中策略: {chosen}（{'可实盘' if chosen in TRADEABLE else '研究对照'}）")
    print(f"预测明细 -> {pred_csv}")
    print(f"报告     -> {report_path}")


if __name__ == "__main__":
    main()

"""阶段二-诊断: 锁定预测器(ic_linear+size_neutral)的 OOS 分层收益。

回答核心矛盾"IC 最高(+0.0415)但 long_topk 亏"——把每个 OOS 日按预测分成 D 层,
看各层真实未来收益(responder_6, 行业内超额),揭示 edge 到底在头部、中部还是
"只在空头侧"。long_topk 做多 top-20%,只有当顶层显著为正时才赚得到。

输出:
  - 各 decile 的平均未来收益(全 OOS 拼接)
  - top层 / bottom层 / 多空(top-bottom)
  - top-20% vs 全体均值的"超额"(这正是 long_topk 能吃到的部分)
  - 单调性(层序与层收益的秩相关)
全部用 walkforward 锁定的预测器逐窗口 OOS 预测,严格防泄露。
"""
from __future__ import annotations

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
from walkforward import _train_predict
from factor_eval import _windows


def main() -> None:
    n_layers = int(os.environ.get("DECILE_N", "10"))
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print(f"加载面板... 预测器={C.MODEL_OBJECTIVE} "
          f"size_neutral={getattr(C,'LINEAR_SIZE_NEUTRAL',True)} | 分{n_layers}层")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)
    dates = np.sort(panel["date"].unique())

    oos = []
    for win_id, tr, te in _windows(dates):
        train = panel[panel["date"].isin(tr)]
        test = panel[panel["date"].isin(te)].copy()
        test["prediction"] = _train_predict(train, test, factors)
        oos.append(test[["date", "symbol", "prediction", C.TARGET]])
    oos = pd.concat(oos, ignore_index=True)

    # 每个截面日按预测分 n_layers 层(0=最低分, n-1=最高分)
    def _assign(g):
        if g["prediction"].notna().sum() < n_layers:
            return pd.Series(np.nan, index=g.index)
        return pd.qcut(g["prediction"].rank(method="first"), n_layers,
                       labels=False, duplicates="drop")
    oos["layer"] = oos.groupby("date", group_keys=False).apply(
        _assign, include_groups=False)

    valid = oos.dropna(subset=["layer", C.TARGET])
    layer_ret = valid.groupby("layer")[C.TARGET].mean().reindex(range(n_layers))
    overall = valid[C.TARGET].mean()

    print("\n===== OOS 分层未来收益(行业内超额, responder_6)=====")
    print(f"{'层(0=最低分)':<14}{'平均未来收益':>14}{'相对全体超额':>14}")
    for i in range(n_layers):
        r = layer_ret.get(i, np.nan)
        print(f"  Q{i:<11}{r:>+14.5f}{(r-overall):>+14.5f}")
    top, bot = layer_ret.iloc[-1], layer_ret.iloc[0]
    print(f"\n全体均值          : {overall:+.5f}")
    print(f"顶层 Q{n_layers-1}(最高分)   : {top:+.5f}  (超额 {top-overall:+.5f})")
    print(f"底层 Q0(最低分)    : {bot:+.5f}  (超额 {bot-overall:+.5f})")
    print(f"多空 (顶-底)       : {top-bot:+.5f}")

    # long_topk 真正吃到的: top-20% 组合超额
    topk = int(os.environ.get("TOPK_LAYERS", str(max(1, n_layers // 5))))  # 顶部~20%
    top_layers = list(range(n_layers - topk, n_layers))
    top_ret = valid[valid["layer"].isin(top_layers)][C.TARGET].mean()
    print(f"\nlong_topk 视角: 顶部 {topk}/{n_layers} 层(≈top-{100*topk//n_layers}%) "
          f"平均未来收益={top_ret:+.5f}  超额={top_ret-overall:+.5f}")

    vv = layer_ret.dropna()
    mono = stats.spearmanr(vv.index, vv.values).correlation if len(vv) >= 3 else np.nan
    print(f"分层单调性(层序 vs 层收益秩相关): {mono:+.3f}")

    # 诊断结论
    print("\n--- 结论 ---")
    if top - overall <= 0:
        print("⚠ 顶层不跑赢全体 → edge 不在头部。long_topk(只做多top)结构上吃不到,"
              "  alpha 多在'避开底层'(空头侧),散户 long-only 无法收割。")
        if bot - overall < (top - overall):
            print("  底层显著跑输 → 信号有效但偏空头。可行方向: 多空策略(研究)/"
                  "  反向用作'排雷'剔除底层后再等权,而非追顶层。")
    else:
        print("✓ 顶层跑赢全体 → edge 在头部,long_topk 可收割。下一步调 TOPK_RATIO/"
              "  换手缓冲/调仓频率放大净值即可。")

    pd.DataFrame({"layer": range(n_layers), "future_ret": layer_ret.values,
                  "excess": (layer_ret - overall).values}).to_csv(
        os.path.join(C.OUTPUT_DIR, "decile_returns.csv"), index=False)
    print("-> outputs/decile_returns.csv")


if __name__ == "__main__":
    main()

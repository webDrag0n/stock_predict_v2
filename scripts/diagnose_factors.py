"""因子层诊断三件套 —— 在决定下一步优化方向前,先把"信号有没有 / 在哪 / 模型有没有榨出组合增益"看清楚。

复用 walkforward 完全相同的窗口切分(TRAIN_WINDOW/TEST_WINDOW/EMBARGO/STEP)与
_train_predict,因此所有 IC 都是严格防泄露的样本外(OOS)数字,可直接与
run_walkforward 的连续 OOS 结果对照。

输出三张表:
  1. 单因子 OOS IC 衰减表:每个因子在每个 OOS 窗口的 RankIC,看哪些因子持续
     有效、哪些只是个别窗口运气(ICIR 高=稳定,为正窗口多=可复现)。
  2. 因子相关矩阵:按日截面 z-score 后的 |相关| 热图,找冗余因子簇(冗余会
     稀释 GBM 的有效自由度)。
  3. 模型 IC vs 最强单因子 IC:逐窗口对比。若模型≈最强单因子,说明 GBM 没榨出
     组合增益,瓶颈在"组合方式"而非"因子数量"。

用法
----
  python diagnose_factors.py                 # 跑全部三件套,CSV+图存 outputs/
  python diagnose_factors.py --no-model      # 跳过模型(只看因子,快)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import config as C
from data import load_factor_panel, list_factor_columns
from walkforward import _train_predict, _cross_sectional_norm
from factor_eval import _windows, _daily_rank_ic


def factor_ic_decay(panel: pd.DataFrame, factors: list[str],
                    target: str) -> pd.DataFrame:
    """诊断1: 每个因子 × 每个 OOS 窗口的 RankIC,汇总成衰减/稳定性表。

    只在每个窗口的【测试段(OOS)】上算 IC,与模型预测同一批日期,因此因子强弱
    和模型强弱可在同一标尺上对照。返回每因子一行:各窗口 IC 的均值/标准差/
    ICIR/为正窗口数,按 |ICIR| 降序。
    """
    dates = np.sort(panel["date"].unique())
    wins = list(_windows(dates))
    rows = []
    for f in factors:
        per_win = []
        for _, _tr, te in wins:
            sub = panel[panel["date"].isin(te)]
            ic = _daily_rank_ic(sub, f, target).dropna()
            per_win.append(float(ic.mean()) if len(ic) else np.nan)
        arr = np.array(per_win, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        mean_ic = float(valid.mean())
        std_ic = float(valid.std()) if len(valid) > 1 else np.nan
        icir = mean_ic / std_ic if std_ic and std_ic > 1e-12 else np.nan
        rows.append({
            "factor": f,
            "IC_mean": mean_ic,
            "IC_std": std_ic,
            "ICIR": icir,
            "pos_windows": int((valid > 0).sum()),
            "n_windows": len(valid),
            **{f"w{i}": v for i, v in enumerate(per_win)},
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_abs"] = table["ICIR"].abs()
    table = table.sort_values("_abs", ascending=False, na_position="last")
    return table.drop(columns="_abs").reset_index(drop=True)


def factor_correlation(panel: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """诊断2: 按日截面 z-score 后的因子相关矩阵(Pearson on pooled z-scores)。

    先对每个因子做逐日 z-score(与模型输入完全一致),再在全样本上算相关。
    高相关因子簇 = 冗余,会稀释 GBM 的有效自由度。返回相关矩阵 DataFrame。
    """
    z, _ = _cross_sectional_norm(panel, factors)
    zdf = pd.DataFrame(z, columns=factors)
    return zdf.corr(method="pearson")


def redundant_clusters(corr: pd.DataFrame, thresh: float = 0.8) -> list[list[str]]:
    """从相关矩阵里抽出 |corr|>=thresh 的因子簇(简单连通分量),用于报告冗余。"""
    cols = list(corr.columns)
    adj = {c: set() for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= thresh:
                adj[a].add(b)
                adj[b].add(a)
    seen, clusters = set(), []
    for c in cols:
        if c in seen or not adj[c]:
            continue
        stack, comp = [c], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            stack.extend(adj[node] - seen)
        if len(comp) > 1:
            clusters.append(sorted(comp))
    return clusters


def model_vs_best_factor(panel: pd.DataFrame, factors: list[str],
                         target: str) -> pd.DataFrame:
    """诊断3: 逐窗口对比【模型 OOS IC】vs【该窗口最强单因子 OOS IC】。

    模型用 walkforward._train_predict(同款防泄露训练)。若模型 IC 全程不超过
    最强单因子,说明 GBM 没榨出组合增益——瓶颈在组合方式,不在因子数量。
    """
    dates = np.sort(panel["date"].unique())
    rows = []
    for win_id, tr_dates, te_dates in _windows(dates):
        train = panel[panel["date"].isin(tr_dates)]
        test = panel[panel["date"].isin(te_dates)].copy()
        pred = _train_predict(train, test, factors)
        test["_pred"] = pred
        # 模型 OOS IC
        mdl_ic = _daily_rank_ic(
            test.rename(columns={"_pred": "__p"}), "__p", target).dropna()
        model_ic = float(mdl_ic.mean()) if len(mdl_ic) else np.nan
        # 该窗口每个单因子的 OOS IC,取最强
        best_f, best_ic = None, -np.inf
        for f in factors:
            ic = _daily_rank_ic(test, f, target).dropna()
            v = abs(float(ic.mean())) if len(ic) else np.nan
            if v == v and v > best_ic:
                best_ic, best_f = v, f
        rows.append({
            "window": win_id,
            "model_IC": model_ic,
            "best_factor": best_f,
            "best_factor_IC": best_ic if best_ic != -np.inf else np.nan,
            "model_gain": (abs(model_ic) - best_ic)
            if (model_ic == model_ic and best_ic != -np.inf) else np.nan,
        })
        print(f"  窗口{win_id}: 模型IC={model_ic:+.4f} | "
              f"最强单因子={best_f}({best_ic:+.4f}) | "
              f"组合增益={rows[-1]['model_gain']:+.4f}")
    return pd.DataFrame(rows)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"

    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.32), max(7, n * 0.30)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title("Factor correlation (per-date cross-sectional z-score, Pearson)\n"
                 "Red blocks off-diagonal = redundant factor clusters",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="因子层诊断三件套(防泄露 OOS)")
    ap.add_argument("--no-model", action="store_true",
                    help="跳过诊断3(模型 vs 最强单因子),只跑因子诊断,更快")
    ap.add_argument("--corr-thresh", type=float, default=0.8,
                    help="冗余因子簇的 |相关| 阈值(默认0.8)")
    args = ap.parse_args()

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    print("加载命名因子面板...")
    panel = load_factor_panel(keep_price=True)
    factors = list_factor_columns(panel)
    dates = np.sort(panel["date"].unique())
    n_win = sum(1 for _ in _windows(dates))
    print(f"候选因子 {len(factors)} 个 | {len(dates)} 个交易日 | "
          f"{n_win} 个 OOS 窗口 | FW={C.FORWARD_WINDOW} TRAIN={C.TRAIN_WINDOW} "
          f"TEST={C.TEST_WINDOW} embargo={C.EMBARGO_DAYS}\n")

    # ---- 诊断1: 单因子 OOS IC 衰减/稳定性 ----
    print("=" * 70)
    print("诊断1: 单因子 OOS IC 衰减表(按 |ICIR| 降序,IC 均为防泄露样本外)")
    print("=" * 70)
    decay = factor_ic_decay(panel, factors, C.TARGET)
    show_cols = ["factor", "IC_mean", "IC_std", "ICIR", "pos_windows", "n_windows"]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print(decay[show_cols].to_string(index=False))
    decay.to_csv(os.path.join(C.OUTPUT_DIR, "diag_factor_ic_decay.csv"), index=False)
    print(f"\n-> outputs/diag_factor_ic_decay.csv")

    # ---- 诊断2: 因子相关矩阵 + 冗余簇 ----
    print("\n" + "=" * 70)
    print(f"诊断2: 因子相关矩阵 + 冗余簇(|corr| >= {args.corr_thresh})")
    print("=" * 70)
    corr = factor_correlation(panel, factors)
    corr.to_csv(os.path.join(C.OUTPUT_DIR, "diag_factor_corr.csv"))
    heat = plot_corr_heatmap(corr, os.path.join(C.OUTPUT_DIR, "diag_factor_corr.png"))
    clusters = redundant_clusters(corr, args.corr_thresh)
    if clusters:
        print(f"发现 {len(clusters)} 个冗余因子簇(簇内高度相关,可考虑降维/择一):")
        for i, cl in enumerate(clusters):
            print(f"  簇{i+1}: {', '.join(cl)}")
    else:
        print(f"无 |corr|>={args.corr_thresh} 的冗余簇。")
    print(f"-> outputs/diag_factor_corr.csv, outputs/diag_factor_corr.png")

    # ---- 诊断3: 模型 vs 最强单因子 ----
    if not args.no_model:
        print("\n" + "=" * 70)
        print("诊断3: 逐窗口 模型 OOS IC vs 最强单因子 OOS IC")
        print("=" * 70)
        mvb = model_vs_best_factor(panel, factors, C.TARGET)
        mvb.to_csv(os.path.join(C.OUTPUT_DIR, "diag_model_vs_factor.csv"), index=False)
        wins_model_better = int((mvb["model_gain"] > 0).sum())
        print(f"\n模型 IC 均值={mvb['model_IC'].mean():+.4f} | "
              f"最强单因子 IC 均值={mvb['best_factor_IC'].mean():+.4f} | "
              f"模型胜出窗口={wins_model_better}/{len(mvb)}")
        if wins_model_better <= len(mvb) // 2:
            print("⚠ 模型未稳定超过最强单因子 → 组合增益不足,瓶颈在【组合方式】"
                  "(优先做集成/中性化),而非因子数量。")
        else:
            print("✓ 模型多数窗口跑赢最强单因子 → GBM 确有组合增益,"
                  "瓶颈更可能在【因子维度】(优先加正交因子/扩横截面)。")
        print(f"-> outputs/diag_model_vs_factor.csv")

    print("\n诊断完成。下一步方向由诊断3的结论 + 诊断1的为正窗口数 + "
          "诊断2的冗余簇共同决定。")


if __name__ == "__main__":
    main()

"""单因子分层净值与逐日 RankIC 落盘（前端因子详情 P2/P3 面板的后端契约）。

设计文档 docs/frontend_redesign.html 5.4 第 ① 项：
``factor_eval.layer_equity()`` 已存在，eval 会话把分层净值落盘供前端直接读。

产物契约（写到 outputs/ 下，一因子两份小文件）::

    factor_layer_<id>.parquet     长表 (date, layer, equity)
                                  layer 为 L1..Ln（L1=因子最小层，Ln=最大层），
                                  equity 为各层逐日平均收益的累计净值
    factor_ic_daily_<id>.parquet  长表 (date, rank_ic)：逐截面日 Spearman RankIC

文件名里的 <id> 经 ``safe_factor_id`` 过滤（仅保留 [0-9A-Za-z_.-]）。
两文件均缺失时前端显示空态并提示重跑 scripts/eval_factors.py。
"""

from __future__ import annotations

import os
import re

import pandas as pd

import factor_eval

_SUMMARY_COLS = ["factor", "ok", "layer_path", "ic_path", "n_dates", "error"]


def safe_factor_id(fid) -> str:
    """因子 id → 文件名片段：只保留字母数字与 _ . -，其余折叠为 _。"""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(fid)).strip("_") or "factor"


def layer_equity_path(out_dir: str, factor: str) -> str:
    return os.path.join(out_dir, f"factor_layer_{safe_factor_id(factor)}.parquet")


def daily_ic_path(out_dir: str, factor: str) -> str:
    return os.path.join(out_dir, f"factor_ic_daily_{safe_factor_id(factor)}.parquet")


def dump_factor_layer(
    panel: pd.DataFrame,
    factor: str,
    out_dir: str,
    target: str | None = None,
    n_layers: int = 5,
) -> dict:
    """把单个因子的分层净值 + 逐日 RankIC 落盘，返回一行摘要 dict。

    逐因子容错：任何一步失败记 error 返回，不抛异常（批量场景不拖垮整批）。
    """
    import config as C

    target = target or C.TARGET
    res = {
        "factor": factor,
        "ok": False,
        "layer_path": "",
        "ic_path": "",
        "n_dates": 0,
        "error": "",
    }
    try:
        if factor not in panel.columns:
            raise KeyError(f"panel 中不存在因子列 {factor!r}")
        os.makedirs(out_dir, exist_ok=True)

        # 分层净值：index=date、列=层号的宽表 → 长表 (date, layer, equity)
        eq = factor_eval.layer_equity(panel, factor, target, n_layers)
        if eq.empty:
            raise ValueError("分层净值为空（有效样本不足）")
        long = (
            eq.rename(columns={c: f"L{int(c) + 1}" for c in eq.columns})
            .reset_index()
            .melt(id_vars="date", var_name="layer", value_name="equity")
            .dropna(subset=["equity"])
        )
        lpath = layer_equity_path(out_dir, factor)
        long.to_parquet(lpath, index=False)
        res.update(layer_path=lpath, n_dates=int(eq.shape[0]))

        # 逐日 RankIC（P3 IC 分布直方图数据源）
        ric = factor_eval.daily_rank_ic(panel, factor, target).dropna()
        if len(ric):
            ipath = daily_ic_path(out_dir, factor)
            ric.rename("rank_ic").reset_index()[["date", "rank_ic"]].to_parquet(ipath, index=False)
            res["ic_path"] = ipath
        res["ok"] = True
    except Exception as ex:  # noqa: BLE001 —— 批量落盘逐因子容错
        res["error"] = f"{type(ex).__name__}: {ex}"
    return res


def dump_layers(
    panel: pd.DataFrame,
    factors: list[str],
    out_dir: str,
    target: str | None = None,
    n_layers: int = 5,
) -> pd.DataFrame:
    """批量落盘，返回逐因子摘要表（列见 _SUMMARY_COLS）。"""
    rows = [dump_factor_layer(panel, f, out_dir, target=target, n_layers=n_layers) for f in factors]
    return pd.DataFrame(rows, columns=_SUMMARY_COLS)

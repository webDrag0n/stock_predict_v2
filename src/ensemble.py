"""多模型增量集成 —— 每日续训 + 近期滚动 IC 定权 + 截面 rank 加权融合。

设计目标
--------
取代回测/实盘里"每天从零全量重训"的不可行做法。全市场 ~5700 股 × 数百回测日，
全量重训每天都要拟合数百万行，回测一轮要数小时。这里改为：

  1. 增量续训（init_model）：GBM 用 LightGBM 的 `init_model` 续训接口，每个更新点
     只在【当日新成熟的标签行】上再长 INCR_TREES 棵树，接在昨日 booster 之上；
     每 REBUILD_DAYS 个交易日做一次全量重建（在尾部 TRAIN_WINDOW 天上从零训练），
     防止增量漂移累积。速度从"小时级"降到"分钟级"。
  2. 多模型：gbm_l2（l2 回归）、gbm_ret_weighted（样本权重∝当日前瞻收益排名，
     与验证回测同口径）、ic_linear（IC 加权线性 blend，抗过拟合）。三者误差来源
     不同，融合后更稳。
  3. 近期滚动 IC 定权（真正样本外、无泄露）：每个模型在【过去已成熟】交易日上的
     预测 vs 实际，算每日 rank-IC，取尾部 IC_WINDOW 天均值作权重（截断到 ≥0 再
     归一）。表现差的模型权重自动趋零。全为非正时退回等权。
  4. 截面 rank 融合：各模型预测先按日转百分位 rank，再按权重加权平均 —— 尺度稳健，
     避免某模型预测量纲大主导结果（与 walkforward SEED_COMBINE=rank 同理）。

防泄露铁律
----------
- 续训/全量重建只用【标签已成熟】的行：date ≤ asof − FORWARD_WINDOW（responder_6
  在 date 当天前瞻 FORWARD_WINDOW 天，只有 asof 走到 date+FORWARD_WINDOW 才已知）。
  调用方（backtest_daily / live）负责传入已切好的成熟训练帧。
- 定权用的 IC 也只在【已成熟】的历史预测日上计算，绝不偷看未成熟标签。
- 截面归一化（_cross_sectional_norm）每天只用当日自身横截面，逐项无泄露。

复用 walkforward 的截面归一化、ic_linear、size-neutral 与超参，保证与已验证回测
同分布。
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C

# 复用 walkforward 的、已验证的处理函数，保证同分布
from walkforward import (_cross_sectional_norm, _train_predict_linear,
                         _per_date_rank)


# GBM 超参（镜像 walkforward._gbm_l2_predict，但用低层 train 以支持 init_model 续训）
_GBM_PARAMS = dict(
    objective="l2", learning_rate=0.03, num_leaves=31,
    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.6,
    min_child_samples=40, lambda_l1=0.1, lambda_l2=0.5,
    verbose=-1,
)


def _n_threads(n_rows: int) -> int:
    """按数据量选线程数，避免小增量上的线程超订（22 核抢一个 60 行的 LGB
    数据集会空转数分钟）。全市场全量重建（百万行）用满核；小增量按 ~每核 2 万行
    粗略分配，至少 1、至多机器核数。"""
    import os as _os
    ncpu = _os.cpu_count() or 4
    return int(max(1, min(ncpu, n_rows // 20000 + 1)))

# 默认模型集合：两类 GBM（可续训）+ 线性（每次轻量重算）
DEFAULT_MODELS = ("gbm_l2", "gbm_ret_weighted", "ic_linear")


def _ret_weighted_sample_w(train: pd.DataFrame) -> np.ndarray:
    """样本权重 = 0.2 + 当日前瞻收益百分位排名（头部≈6×尾部），同 walkforward。

    叠加已有的 inverse-vol/recency 权重（若存在）。仅用训练帧自身的前瞻收益排名，
    无泄露。"""
    rnk = (train.groupby("date", group_keys=False)[C.TARGET]
                .rank(pct=True)).to_numpy()
    w = 0.2 + np.nan_to_num(rnk, nan=0.5)
    if C.WEIGHT in train:
        w = w * train[C.WEIGHT].to_numpy(dtype=np.float64)
    return w


class IncrementalEnsemble:
    """增量多模型集成。每日 update（续训）→ predict（出各模型分）→ combine（rank 加权）。

    用法（回测重放 / 实盘共用）::

        ens = IncrementalEnsemble(feat_cols)
        for t in trading_days:
            matured = panel[panel.date <= t - FORWARD_WINDOW]      # 已成熟标签
            ens.update(matured, asof_date=t)                       # 续训/重建
            ens.refresh_weights(panel, asof_date=t)                # 滚动 IC 定权
            today = panel[panel.date == t]
            preds = ens.predict(today)                             # {model: pred}
            score = ens.combine(preds, today.date.values)          # 融合排序分
            ens.observe(t, today, preds)                           # 存预测供日后定权
    """

    def __init__(self, feat_cols: list[str], models: tuple[str, ...] = None,
                 n_seeds: int = None, incr_trees: int = None,
                 rebuild_days: int = None, train_window: int = None,
                 ic_window: int = None):
        self.feat_cols = list(feat_cols)
        # 模型清单优先级：显式参数 > 环境变量 > config.ENSEMBLE_MODELS > DEFAULT。
        # （此前直接读 os.environ 漏掉了 config 默认，导致 attn 默认 leg 不生效。）
        _cfg_models = getattr(C, "ENSEMBLE_MODELS", "") or ""
        _env_models = os.environ.get("ENSEMBLE_MODELS", "")
        _src = models or _env_models or _cfg_models or ""
        if isinstance(_src, str):
            _src = _src.split(",")
        self.models = tuple(m.strip() for m in _src if m and m.strip()) or DEFAULT_MODELS
        self.n_seeds = int(n_seeds if n_seeds is not None
                           else os.environ.get("N_SEEDS", "1"))
        self.incr_trees = int(incr_trees if incr_trees is not None
                              else os.environ.get("INCR_TREES", "60"))
        self.rebuild_days = int(rebuild_days if rebuild_days is not None
                                else os.environ.get("REBUILD_DAYS", "60"))
        self.train_window = int(train_window if train_window is not None
                                else getattr(C, "TRAIN_WINDOW", 500))
        self.ic_window = int(ic_window if ic_window is not None
                             else os.environ.get("IC_WINDOW", "20"))

        # 各 GBM 模型的 booster 列表（多 seed），ic_linear 不持 booster
        self._boosters: dict[str, list[lgb.Booster]] = {
            m: [] for m in self.models if m.startswith("gbm")}
        self._attn_model = None      # attn leg：全量重建时训练，期间复用
        self._updates_since_rebuild = 0
        self._last_trained_date = None      # 增量续训进度（只在新成熟日上续树）
        self._last_train_frame: pd.DataFrame | None = None  # 供 ic_linear 取 IC 权重
        self._last_trained_date = None     # 已续训到的最新成熟标签日（防重复 boost）

        # 滚动 IC 定权：每个模型一个尾部 IC 队列；预测历史供日后成熟时打分
        self._ic_hist: dict[str, deque] = {
            m: deque(maxlen=self.ic_window) for m in self.models}
        self.weights: dict[str, float] = {m: 1.0 / len(self.models)
                                          for m in self.models}
        self._pred_hist: dict[object, pd.DataFrame] = {}  # date -> [symbol×model preds]
        self._scored_dates: set = set()

    # ------------------------------------------------------------------ train
    def update(self, train_frame: pd.DataFrame, asof_date=None,
               force_full: bool = False) -> dict:
        """续训（或周期全量重建）各 GBM。train_frame 必须只含【已成熟标签】行。

        返回 {model: n_trees} 诊断信息。ic_linear 不在此训练（predict 时按尾部
        TRAIN_WINDOW 帧实时算 IC 权重，开销小）。"""
        train = train_frame.dropna(subset=[C.TARGET])
        if train.empty:
            return {}
        # 尾部 TRAIN_WINDOW 天用于全量重建 / 线性 IC 权重
        dates = np.sort(train["date"].unique())
        win_dates = set(dates[-self.train_window:].tolist())
        win = train[train["date"].isin(win_dates)]
        self._last_train_frame = win

        do_full = (force_full or self._updates_since_rebuild >= self.rebuild_days
                   or all(len(self._boosters.get(m, [None])) == 0
                          for m in self._boosters))
        info = {}
        # attn leg：仅在全量重建时（重）训练（神经网不做增量），期间复用模型。
        if "attn" in self.models and (do_full or self._attn_model is None):
            try:
                from model_attn import fit_attn
                self._attn_model = fit_attn(win, self.feat_cols)
                info["attn"] = "fit"
            except Exception as e:
                info["attn"] = f"skip({type(e).__name__})"
        if do_full:
            for m in self._boosters:
                self._boosters[m] = self._fit_full(win, m)
                info[m] = f"full×{len(self._boosters[m])}"
            self._updates_since_rebuild = 0
            self._last_trained_date = dates[-1]
            return info

        # 增量：只在【上次续训之后新成熟】的日期上续树（绝不重复 boost 同一批行，
        # 既防过拟合又快）。首次没有 base 时退回全量。
        if self._last_trained_date is None:
            new_dates = dates
        else:
            new_dates = dates[dates > self._last_trained_date]
        if len(new_dates) == 0:
            self._updates_since_rebuild += 1
            return {m: "no-new" for m in self._boosters}
        incr = train[train["date"].isin(set(new_dates.tolist()))]
        for m in self._boosters:
            self._boosters[m] = self._fit_incremental(incr, m)
            info[m] = f"+{self.incr_trees}@{len(new_dates)}d"
        self._updates_since_rebuild += 1
        self._last_trained_date = dates[-1]
        return info

    def _make_dataset(self, frame: pd.DataFrame, model: str) -> lgb.Dataset:
        X, _ = _cross_sectional_norm(frame, self.feat_cols)
        y = frame[C.TARGET].to_numpy(dtype=np.float64)
        if model == "gbm_ret_weighted":
            w = _ret_weighted_sample_w(frame)
        elif C.WEIGHT in frame:
            w = frame[C.WEIGHT].to_numpy(dtype=np.float64)
        else:
            w = None
        return lgb.Dataset(X, label=y, weight=w, free_raw_data=False)

    def _fit_full(self, frame: pd.DataFrame, model: str) -> list[lgb.Booster]:
        """全量重建：在尾部 TRAIN_WINDOW 帧上从零训练 n_seeds 个 booster。"""
        n_full = int(os.environ.get("FULL_TREES", "400"))
        nt = _n_threads(len(frame))
        boosters = []
        for s in range(self.n_seeds):
            ds = self._make_dataset(frame, model)
            params = dict(_GBM_PARAMS, seed=C.RANDOM_STATE + s,
                          bagging_seed=C.RANDOM_STATE + s, num_threads=nt)
            bst = lgb.train(params, ds, num_boost_round=n_full,
                            keep_training_booster=True)
            boosters.append(bst)
        return boosters

    def _fit_incremental(self, frame: pd.DataFrame, model: str) -> list[lgb.Booster]:
        """续训：在新成熟行上接着昨日 booster 再长 INCR_TREES 棵树。"""
        prev = self._boosters.get(model) or []
        if frame.empty or not prev:
            return prev or self._fit_full(frame, model)
        nt = _n_threads(len(frame))
        boosters = []
        for s, base in enumerate(prev):
            ds = self._make_dataset(frame, model)
            params = dict(_GBM_PARAMS, seed=C.RANDOM_STATE + s,
                          bagging_seed=C.RANDOM_STATE + s, num_threads=nt)
            bst = lgb.train(params, ds, num_boost_round=self.incr_trees,
                            init_model=base, keep_training_booster=True)
            boosters.append(bst)
        return boosters

    # --------------------------------------------------------------- predict
    def predict(self, test_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """各模型对 test_frame 出分（原始空间）。返回 {model: pred array}。"""
        out: dict[str, np.ndarray] = {}
        Xte = None
        for m in self.models:
            if m.startswith("gbm"):
                boosters = self._boosters.get(m) or []
                if not boosters:
                    out[m] = np.zeros(len(test_frame))
                    continue
                if Xte is None:
                    Xte, _ = _cross_sectional_norm(test_frame, self.feat_cols)
                preds = [np.asarray(b.predict(Xte), dtype=np.float64).ravel()
                         for b in boosters]
                out[m] = np.mean(preds, axis=0)
            elif m == "ic_linear":
                if self._last_train_frame is None:
                    out[m] = np.zeros(len(test_frame))
                else:
                    out[m] = _train_predict_linear(
                        self._last_train_frame, test_frame, self.feat_cols)
            elif m == "attn":
                if self._attn_model is None:
                    out[m] = np.zeros(len(test_frame))
                else:
                    from model_attn import predict_attn
                    out[m] = predict_attn(self._attn_model, test_frame, self.feat_cols)
            else:
                out[m] = np.zeros(len(test_frame))
        return out

    def combine(self, preds: dict[str, np.ndarray], dates: np.ndarray) -> np.ndarray:
        """按当前权重做"按日 rank 加权平均"，得到融合排序分（尺度稳健）。"""
        dates = np.asarray(dates)
        total_w = sum(max(self.weights.get(m, 0.0), 0.0) for m in preds) or 1.0
        score = np.zeros(len(dates), dtype=np.float64)
        for m, p in preds.items():
            w = max(self.weights.get(m, 0.0), 0.0)
            if w <= 0:
                continue
            score += (w / total_w) * _per_date_rank(p, dates)
        return score

    # ------------------------------------------------------- weight bookkeeping
    def observe(self, date, test_frame: pd.DataFrame,
                preds: dict[str, np.ndarray]) -> None:
        """存下某交易日各模型的预测（按 symbol 索引），供其标签成熟后打分定权。"""
        df = pd.DataFrame({"symbol": test_frame["symbol"].to_numpy()})
        for m, p in preds.items():
            df[m] = np.asarray(p, dtype=np.float64)
        self._pred_hist[date] = df.set_index("symbol")

    def refresh_weights(self, panel: pd.DataFrame, asof_date) -> dict:
        """对【已成熟但未打分】的历史预测日算每模型 rank-IC，更新尾部 IC → 权重。

        某预测日 d 的标签在 asof ≥ d + FORWARD_WINDOW 时成熟。这里把 panel 中
        date=d 且 responder_6 非空的真实值与当日各模型预测做 spearman rank 相关，
        作为该模型在 d 的日度 IC，推入尾部 IC_WINDOW 队列。权重 = 尾部 IC 均值
        截断 ≥0 后归一；全为非正则等权。无泄露：只用已成熟标签。"""
        from scipy import stats
        fw = getattr(C, "FORWARD_WINDOW", 20)
        # 哪些历史预测日已成熟且没打过分
        mature_cut = self._date_minus(panel, asof_date, fw)
        pend = [d for d in self._pred_hist
                if d not in self._scored_dates and d <= mature_cut]
        if not pend:
            self._recompute_weights()
            return self.weights
        for d in sorted(pend):
            day = panel[(panel["date"] == d) & panel[C.TARGET].notna()]
            if day.empty:
                self._scored_dates.add(d)
                continue
            yser = day.set_index("symbol")[C.TARGET]
            ph = self._pred_hist[d]
            joined = ph.join(yser.rename("_y"), how="inner").dropna(subset=["_y"])
            if len(joined) < 5:
                self._scored_dates.add(d)
                continue
            for m in self.models:
                if m not in joined.columns or joined[m].nunique() < 2:
                    continue
                ic = stats.spearmanr(joined[m], joined["_y"]).correlation
                if ic == ic:                       # not NaN
                    self._ic_hist[m].append(float(ic))
            self._scored_dates.add(d)
            # 控内存：打过分的预测可丢弃
            self._pred_hist.pop(d, None)
        self._recompute_weights()
        return self.weights

    def _recompute_weights(self) -> None:
        """权重 = 各模型尾部 IC 均值，截断 ≥0 归一；数据不足/全非正 → 等权。"""
        means = {}
        for m in self.models:
            h = self._ic_hist[m]
            means[m] = float(np.mean(h)) if len(h) >= 3 else None
        if any(v is None for v in means.values()):
            self.weights = {m: 1.0 / len(self.models) for m in self.models}
            return
        pos = {m: max(v, 0.0) for m, v in means.items()}
        tot = sum(pos.values())
        if tot <= 1e-12:
            self.weights = {m: 1.0 / len(self.models) for m in self.models}
        else:
            self.weights = {m: pos[m] / tot for m in self.models}

    @staticmethod
    def _date_minus(panel: pd.DataFrame, asof_date, n_days: int):
        """返回交易日序列里 asof_date 往前数 n_days 的那个日期（标签成熟分界）。"""
        dts = np.sort(panel["date"].unique())
        idx = np.searchsorted(dts, asof_date)
        j = max(0, idx - n_days)
        return dts[j]

    def ic_means(self) -> dict[str, float]:
        """当前各模型尾部 IC 均值（诊断/展示用）。"""
        return {m: (float(np.mean(self._ic_hist[m])) if self._ic_hist[m] else float("nan"))
                for m in self.models}

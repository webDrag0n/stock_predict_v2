"""多模型增量集成 —— 每日续训 + 近期滚动 IC 定权 + 截面 rank 加权融合。

设计目标
--------
取代回测/实盘里"每天从零全量重训"的不可行做法。全市场 ~5700 股 × 数百回测日，
全量重训每天都要拟合数百万行，回测一轮要数小时。这里改为：

  1. 增量续训（init_model）：GBM 用 LightGBM 的 `init_model` 续训接口，每个更新点
     在【当日新成熟的标签行】上按 INCR_TREES×新成熟天数 续树（两次重建之间总树数
     硬上限 FULL_TREES×2，超限提前重建），接在昨日 booster 之上；
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

from collections import deque

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C

# 复用 walkforward 的、已验证的处理函数，保证同分布；GBM 超参唯一真相源也在
# walkforward（GBM_PARAMS，sklearn 命名），这里经 GBM_PARAMS_NATIVE 映射成
# 低层 lgb.train 命名以支持 init_model 续训，两处不再各自手工维护。
from walkforward import (_cross_sectional_norm, _linear_fit, _linear_score,
                         _per_date_rank, GBM_PARAMS_NATIVE)

_GBM_PARAMS = GBM_PARAMS_NATIVE


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
        # 模型清单优先级：显式参数 > config.ENSEMBLE_MODELS（env 覆盖在 config
        # import 时读取，单一来源）> 内置默认。
        _src = models or C.ENSEMBLE_MODELS or ""
        if isinstance(_src, str):
            _src = _src.split(",")
        self.models = tuple(m.strip() for m in _src if m and m.strip()) or DEFAULT_MODELS
        # attn leg 依赖可选的 torch（pyproject [attn] extra）。未安装时优雅降级：
        # 剔除该 leg 并警告一次，而非等训练时才在 fit_attn 里炸 ImportError。
        if "attn" in self.models:
            from model_attn import TORCH_AVAILABLE
            if not TORCH_AVAILABLE:
                print("[ensemble] 警告: 未安装 torch，attn leg 已剔除"
                      "（安装：uv sync --extra attn）")
                self.models = tuple(m for m in self.models if m != "attn")
                if not self.models:
                    print("[ensemble] 警告: 剔除 attn 后无可用模型，退回默认集合 "
                          f"{DEFAULT_MODELS}")
                    self.models = DEFAULT_MODELS
        self.n_seeds = int(n_seeds if n_seeds is not None else C.ENSEMBLE_N_SEEDS)
        self.incr_trees = int(incr_trees if incr_trees is not None else C.INCR_TREES)
        self.rebuild_days = int(rebuild_days if rebuild_days is not None
                                else C.REBUILD_DAYS)
        self.train_window = int(train_window if train_window is not None
                                else C.TRAIN_WINDOW)
        self.ic_window = int(ic_window if ic_window is not None else C.IC_WINDOW)

        # 各 GBM 模型的 booster 列表（多 seed），ic_linear 不持 booster
        self._boosters: dict[str, list[lgb.Booster]] = {
            m: [] for m in self.models if m.startswith("gbm")}
        self._attn_model = None      # attn leg：全量重建时训练，期间复用
        self._updates_since_rebuild = 0
        self._last_trained_date = None      # 已续训到的最新成熟标签日（防重复 boost）
        self._last_train_frame: pd.DataFrame | None = None  # 供 ic_linear 取训练帧
        # ic_linear 训练侧缓存（因子集 + IC 权重 + NL/IX 变换列选择）：按
        # REBUILD_DAYS 节奏刷新（全量重建时置 None），每日续训期间复用，
        # 避免每日全量重算 IC 权重。_linear_fit_date = 拟合时训练数据截止日。
        self._linear_state: dict | None = None
        self._linear_fit_date = None

        # 滚动 IC 定权：每个模型一个尾部 IC 队列；预测历史供日后成熟时打分
        self._ic_hist: dict[str, deque] = {
            m: deque(maxlen=self.ic_window) for m in self.models}
        self.weights: dict[str, float] = {m: 1.0 / len(self.models)
                                          for m in self.models}
        # 是否已有过一次【数据驱动】的定权（EMA 平滑只对之后的更新生效，
        # 避免首次定权被初始等权稀释——如实盘路径只 refresh 一次）。
        self._w_initialized = False
        self._pred_hist: dict[object, pd.DataFrame] = {}  # date -> [symbol×model preds]
        self._scored_dates: set = set()

    # ------------------------------------------------------------------ train
    def update(self, train_frame: pd.DataFrame, asof_date=None,
               force_full: bool = False) -> dict:
        """续训（或周期全量重建）各 GBM。train_frame 必须只含【已成熟标签】行。

        返回 {model: n_trees} 诊断信息。ic_linear 不在此训练：其训练侧状态
        （IC 权重等）在 predict 时按需拟合并缓存，随全量重建一起刷新。"""
        train = train_frame.dropna(subset=[C.TARGET])
        if train.empty:
            return {}
        # 尾部 TRAIN_WINDOW 天用于全量重建 / 线性 IC 权重
        dates = np.sort(train["date"].unique())
        win_dates = set(dates[-self.train_window:].tolist())
        win = train[train["date"].isin(win_dates)]
        self._last_train_frame = win

        # 显式判空：无 gbm 腿时 self._boosters 为空 dict，all() 空集为 True
        # 会导致每天都走全量重建分支（空转），所以这里要求非空。
        do_full = (force_full or self._updates_since_rebuild >= self.rebuild_days
                   or (bool(self._boosters)
                       and all(len(self._boosters[m]) == 0 for m in self._boosters)))
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
            self._linear_state = None        # 训练分布已重置，IC 权重缓存失效
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
        # 续训树数 = INCR_TREES × 新成熟天数（续训量随新数据量伸缩）；
        # 两次重建之间总树数硬上限 FULL_TREES×2，超限提前触发全量重建，
        # 防止增量漂移无界累积。
        n_boost = self.incr_trees * len(new_dates)
        cur_trees = max((b.num_trees() for bl in self._boosters.values() for b in bl),
                        default=0)
        if cur_trees > 0 and cur_trees + n_boost > C.FULL_TREES * 2:
            for m in self._boosters:
                self._boosters[m] = self._fit_full(win, m)
                info[m] = f"full×{len(self._boosters[m])}(tree-cap)"
            self._updates_since_rebuild = 0
            self._last_trained_date = dates[-1]
            self._linear_state = None
            return info
        incr = train[train["date"].isin(set(new_dates.tolist()))]
        for m in self._boosters:
            self._boosters[m] = self._fit_incremental(incr, m, n_boost)
            info[m] = f"+{n_boost}@{len(new_dates)}d"
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
        # free_raw_data 保持默认 True：construct 后释放原始 X（全市场训练帧
        # ~2.7GB/腿）。续训经由 lgb.train(init_model=booster) 在新 Dataset 上进行，
        # 不需要旧 Dataset 的原始数据（已实测验证）。
        return lgb.Dataset(X, label=y, weight=w)

    def _fit_full(self, frame: pd.DataFrame, model: str) -> list[lgb.Booster]:
        """全量重建：在尾部 TRAIN_WINDOW 帧上从零训练 n_seeds 个 booster。"""
        n_full = C.FULL_TREES
        nt = _n_threads(len(frame))
        # _make_dataset 含逐日截面归一化，开销不小：各 seed 共享同一份 Dataset
        # （仅 seed/bagging_seed 不同），不再每 seed 重复归一化。
        ds = self._make_dataset(frame, model)
        boosters = []
        for s in range(self.n_seeds):
            params = dict(_GBM_PARAMS, seed=C.RANDOM_STATE + s,
                          bagging_seed=C.RANDOM_STATE + s, num_threads=nt)
            # keep_training_booster=False：booster 不持有训练 Dataset 引用，
            # 否则 3 腿各自的全量 Dataset（binned+raw，数 GB）会被常驻保留。
            # 续训用 lgb.train(init_model=bst) 不需要旧训练数据（已实测）。
            bst = lgb.train(params, ds, num_boost_round=n_full,
                            keep_training_booster=False)
            boosters.append(bst)
        return boosters

    def _fit_incremental(self, frame: pd.DataFrame, model: str,
                         num_boost_round: int) -> list[lgb.Booster]:
        """续训：在新成熟行上接着昨日 booster 再长 num_boost_round 棵树
        （= INCR_TREES × 新成熟天数，由 update 计算）。"""
        prev = self._boosters.get(model) or []
        if frame.empty or not prev:
            return prev or self._fit_full(frame, model)
        nt = _n_threads(len(frame))
        ds = self._make_dataset(frame, model)   # 各 seed 共享，见 _fit_full
        boosters = []
        for s, base in enumerate(prev):
            params = dict(_GBM_PARAMS, seed=C.RANDOM_STATE + s,
                          bagging_seed=C.RANDOM_STATE + s, num_threads=nt)
            bst = lgb.train(params, ds, num_boost_round=num_boost_round,
                            init_model=base, keep_training_booster=False)
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
                # 训练侧状态（因子集 + IC 权重 + NL/IX 变换列选择）按
                # REBUILD_DAYS 节奏缓存：全量重建时 update() 置 None 触发
                # 重新拟合，每日续训期间直接复用。正确性依据：权重只依赖
                # 训练帧（截止日记录在 _linear_fit_date），test 侧变换全部
                # 逐日自包含，每天只重算当日横截面。
                # from_state_dict 恢复的集成没有 _last_train_frame（不序列化
                # 训练帧），但 _linear_state 已随快照恢复，可直接打分。
                if self._linear_state is None and self._last_train_frame is None:
                    out[m] = np.zeros(len(test_frame))
                else:
                    if self._linear_state is None:
                        self._linear_state = _linear_fit(self._last_train_frame,
                                                         self.feat_cols)
                        self._linear_fit_date = self._last_train_frame["date"].max()
                    out[m] = _linear_score(test_frame, self._linear_state)
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
        """权重 = 各模型尾部 IC 均值，截断 ≥0 归一；数据不足/全非正 → 等权。

        数据驱动的定权再按 C.IC_WEIGHT_EMA_HALFLIFE 做 EMA 平滑，消除 IC 过零
        跳变引起的权重（→换手）尖峰；halflife=0 关闭。首次数据驱动的定权不平滑
        （否则实盘只 refresh 一次时权重会被初始等权稀释）。
        """
        means = {}
        for m in self.models:
            h = self._ic_hist[m]
            means[m] = float(np.mean(h)) if len(h) >= 3 else None
        if any(v is None for v in means.values()):
            target = {m: 1.0 / len(self.models) for m in self.models}
            data_driven = False
        else:
            pos = {m: max(v, 0.0) for m, v in means.items()}
            tot = sum(pos.values())
            if tot <= 1e-12:
                target = {m: 1.0 / len(self.models) for m in self.models}
            else:
                target = {m: pos[m] / tot for m in self.models}
            data_driven = True
        hl = float(C.IC_WEIGHT_EMA_HALFLIFE)
        if hl > 0 and self._w_initialized:
            a = 1.0 - 0.5 ** (1.0 / hl)
            self.weights = {m: (1.0 - a) * self.weights.get(m, 0.0) + a * target[m]
                            for m in self.models}
        else:
            self.weights = target
        if data_driven:
            self._w_initialized = True

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

    # ------------------------------------------------------------- serialize
    def state_dict(self) -> dict:
        """可 pickle 的完整状态快照（模型工件 + 定权状态 + 超参）。

        供 model_store 落盘 champion/staging。GBM booster 存 model_to_string()
        文本（LightGBM 最稳的跨进程序列化）；attn 腿经 torch.save 存 bytes。
        不存 _pred_hist（原始预测逐日累积、体积大，实盘路径本就从 pred_hist
        parquet 重建 observe），但保留 _ic_hist/_scored_dates/weights 使
        定权状态恢复后连续。
        """
        attn_bytes = None
        if self._attn_model is not None:
            import io
            import torch
            buf = io.BytesIO()
            torch.save(self._attn_model, buf)
            attn_bytes = buf.getvalue()
        return {
            "models": tuple(self.models),
            "n_seeds": self.n_seeds,
            "incr_trees": self.incr_trees,
            "rebuild_days": self.rebuild_days,
            "train_window": self.train_window,
            "ic_window": self.ic_window,
            "boosters": {m: [b.model_to_string() for b in bl]
                         for m, bl in self._boosters.items()},
            "attn": attn_bytes,
            "linear_state": self._linear_state,
            "linear_fit_date": self._linear_fit_date,
            "updates_since_rebuild": self._updates_since_rebuild,
            "last_trained_date": self._last_trained_date,
            "weights": dict(self.weights),
            "ic_hist": {m: list(h) for m, h in self._ic_hist.items()},
            "w_initialized": self._w_initialized,
            "scored_dates": set(self._scored_dates),
        }

    @classmethod
    def from_state_dict(cls, state: dict, feat_cols: list[str]
                        ) -> "IncrementalEnsemble":
        """从 state_dict() 快照重建集成（权重/IC 队列/booster 全部恢复）。

        恢复后 predict/combine 与快照前逐位一致（同一批 booster）；update 可
        继续增量续训（_last_trained_date 已恢复，不会重复 boost）。attn 腿在
        无 torch 环境下跳过（predict 对该腿出 0 分，与 __init__ 降级一致）。
        """
        ens = cls(feat_cols, models=state["models"], n_seeds=state["n_seeds"],
                  incr_trees=state["incr_trees"],
                  rebuild_days=state["rebuild_days"],
                  train_window=state["train_window"],
                  ic_window=state["ic_window"])
        ens._boosters = {
            m: [lgb.Booster(model_str=s) for s in bl]
            for m, bl in state["boosters"].items()}
        if state.get("attn") is not None:
            try:
                import io
                import torch
                ens._attn_model = torch.load(io.BytesIO(state["attn"]),
                                             weights_only=False)
            except ImportError:
                print("[ensemble] 警告: 快照含 attn 腿但当前环境无 torch，"
                      "该腿已跳过（predict 出 0 分）")
        ens._linear_state = state["linear_state"]
        ens._linear_fit_date = state["linear_fit_date"]
        ens._updates_since_rebuild = state["updates_since_rebuild"]
        ens._last_trained_date = state["last_trained_date"]
        ens.weights = dict(state["weights"])
        ens._ic_hist = {m: deque(h, maxlen=ens.ic_window)
                        for m, h in state["ic_hist"].items()}
        ens._w_initialized = state["w_initialized"]
        ens._scored_dates = set(state["scored_dates"])
        return ens

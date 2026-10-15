"""retrain_trigger 触发判定测试。

覆盖：model_age_days（train_end_date / decided_at 回退 / 缺失）；
should_retrain 三条件各自触发与不触发、OR 语义、IC 样本不足跳过、force。

路径由 tests/conftest.py 统一处理；standalone 运行：
PYTHONPATH=src:. uv run python tests/test_retrain_trigger.py
"""
from __future__ import annotations

import pandas as pd

import config as C
from retrain_trigger import model_age_days, should_retrain


def _panel(n_dates: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    return pd.DataFrame({"date": dates, "symbol": "S000"})


def _manifest(train_end: str | None = None, decided: str | None = None) -> dict:
    m = {"model_id": "m1"}
    if train_end:
        m["train_end_date"] = train_end
    if decided:
        m["promotion"] = {"decided_at": decided}
    return m


# ---------------------------------------------------------------------------
def test_model_age_days():
    panel = _panel(80)
    dates = sorted(panel["date"].unique())
    # train_end 在倒数第 11 天 → 年龄 10
    m = _manifest(train_end=str(pd.Timestamp(dates[-11]).date()))
    assert model_age_days(m, panel) == 10
    # train_end = 最新日 → 0
    m = _manifest(train_end=str(pd.Timestamp(dates[-1]).date()))
    assert model_age_days(m, panel) == 0
    # 无 train_end → 回退 decided_at
    m = _manifest(decided=str(pd.Timestamp(dates[-6]).date()) + "T18:00:00")
    assert model_age_days(m, panel) == 5
    # 都没有 → None；空 manifest → None
    assert model_age_days(_manifest(), panel) is None
    assert model_age_days({}, panel) is None
    # 非交易日（周末）→ 指向其后第一个交易日，年龄取保守值
    d = pd.Timestamp(dates[-11])
    weekend = d + pd.Timedelta(days=(5 - d.weekday()) % 7)  # 推到周六
    if weekend.weekday() < 5:
        weekend += pd.Timedelta(days=2)
    m = _manifest(train_end=str(weekend.date()))
    age = model_age_days(m, panel)
    assert age is not None and 0 <= age <= 12


def _stats(ic_n=30, ic_mean_w=0.03, live_ret=0.04, bench_ret=0.01):
    ic = {"n_ic": ic_n, "ic_mean_w": ic_mean_w, "window": 20, "ics": {}}
    nav = {"has_nav": live_ret is not None, "window": 20,
           "live_ret_w": live_ret, "bench_ret_w": bench_ret}
    return ic, nav


def test_should_retrain_no_champion_and_force():
    panel = _panel()
    ic, nav = _stats()
    fire, reasons = should_retrain(None, ic, nav, panel)
    assert fire and "bootstrap" in reasons[0]
    fire, reasons = should_retrain(_manifest(train_end="2026-03-30"), ic, nav,
                                   panel, force=True)
    assert fire and "强制" in reasons[0]


def test_should_retrain_all_clear():
    panel = _panel(80)
    dates = sorted(panel["date"].unique())
    m = _manifest(train_end=str(pd.Timestamp(dates[-3]).date()))  # 年龄 2
    ic, nav = _stats(ic_n=30, ic_mean_w=0.03, live_ret=0.04, bench_ret=0.01)
    fire, reasons = should_retrain(m, ic, nav, panel)
    assert not fire and reasons == []


def test_should_retrain_ic_degradation():
    panel = _panel(80)
    dates = sorted(panel["date"].unique())
    m = _manifest(train_end=str(pd.Timestamp(dates[-3]).date()))
    # 样本充足 + IC 均值 < 0 → 触发
    ic, nav = _stats(ic_n=30, ic_mean_w=-0.01)
    fire, reasons = should_retrain(m, ic, nav, panel)
    assert fire and any("IC 退化" in r for r in reasons)
    # 样本不足 → IC 条件跳过
    ic, nav = _stats(ic_n=int(C.RETRAIN_IC_MIN_SAMPLES) - 1, ic_mean_w=-0.01)
    fire, reasons = should_retrain(m, ic, nav, panel)
    assert not any("IC 退化" in r for r in reasons)
    # ic_mean_w 缺失 → 不触发
    ic = {"n_ic": 30, "ic_mean_w": None, "window": 20, "ics": {}}
    fire, _ = should_retrain(m, ic, nav, panel)
    assert not fire


def test_should_retrain_underperformance_and_no_nav_skip():
    panel = _panel(80)
    dates = sorted(panel["date"].unique())
    m = _manifest(train_end=str(pd.Timestamp(dates[-3]).date()))
    # 落后基准超容忍 → 触发
    ic, nav = _stats(live_ret=-0.03, bench_ret=0.02)
    fire, reasons = should_retrain(m, ic, nav, panel)
    assert fire and any("落后" in r for r in reasons)
    # 无净值 → 收益条件跳过，不触发
    ic, nav = _stats(live_ret=None, bench_ret=None)
    fire, _ = should_retrain(m, ic, nav, panel)
    assert not fire
    # 差距在容忍内 → 不触发
    ic, nav = _stats(live_ret=0.005, bench_ret=0.02)   # gap -1.5% > -3%
    fire, _ = should_retrain(m, ic, nav, panel)
    assert not fire


def test_should_retrain_age_limit():
    panel = _panel(80)
    dates = sorted(panel["date"].unique())
    max_age = int(C.MAX_MODEL_AGE_DAYS)
    # 超龄 → 触发
    m = _manifest(train_end=str(pd.Timestamp(dates[0]).date()))
    ic, nav = _stats()
    fire, reasons = should_retrain(m, ic, nav, panel)
    assert fire and any("超龄" in r for r in reasons)
    # 未超龄 → 不触发
    m = _manifest(train_end=str(pd.Timestamp(dates[-max_age]).date()))
    fire, _ = should_retrain(m, ic, nav, panel)
    assert not fire


if __name__ == "__main__":
    test_model_age_days()
    test_should_retrain_no_champion_and_force()
    test_should_retrain_all_clear()
    test_should_retrain_ic_degradation()
    test_should_retrain_underperformance_and_no_nav_skip()
    test_should_retrain_age_limit()
    print("retrain_trigger tests passed")

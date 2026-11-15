from __future__ import annotations

import os

import pandas as pd

import config as C


def run_preflight(
    panel: pd.DataFrame,
    account_path: str,
    today=None,
    max_cash_age_days: int | None = None,
) -> list[str]:
    failures: list[str] = []
    current = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    max_age = int(C.PREFLIGHT_MAX_CASH_AGE_DAYS if max_cash_age_days is None else max_cash_age_days)

    if not os.path.exists(account_path):
        failures.append(f"账户现金文件不存在：{account_path}")
    else:
        try:
            account = pd.read_csv(account_path)
            if "cash" not in account.columns:
                failures.append(f"账户文件缺少 cash 列：{account_path}")
            else:
                cash = pd.to_numeric(account["cash"], errors="coerce").dropna()
                if cash.empty:
                    failures.append(f"账户文件最后一个非空 cash 值无法转为数字：{account_path}")
        except Exception as exc:
            failures.append(f"账户现金文件读取失败：{type(exc).__name__}: {exc}")
        try:
            modified = pd.Timestamp.fromtimestamp(os.path.getmtime(account_path)).normalize()
            age = int((current - modified).days)
            if age > max_age:
                failures.append(f"账户现金已过期：文件修改时间距今天 {age} 天，上限 {max_age} 天")
        except OSError as exc:
            failures.append(f"账户现金修改时间读取失败：{exc}")

    if panel.empty or "date" not in panel.columns or "symbol" not in panel.columns:
        failures.append("因子面板为空或缺少 date/symbol 列")
        return failures

    dates = pd.to_datetime(panel["date"], errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        failures.append("因子面板没有有效交易日")
        return failures
    latest = valid_dates.max().normalize()
    panel_age = int((current - latest).days)
    if panel_age > 4:
        failures.append(f"面板最新日 {latest.date()} 距今天 {panel_age} 天，超过 4 天上限")

    counts = panel.assign(_date=dates).groupby("_date")["symbol"].nunique().sort_index().tail(20)
    latest_count = int(counts.iloc[-1]) if len(counts) else 0
    if latest_count <= 0:
        failures.append("面板最新日股票数为 0")
    if len(counts):
        median = float(counts.median())
        minimum = median * float(C.PREFLIGHT_MIN_COVERAGE)
        if latest_count < minimum:
            failures.append(
                f"最新日横截面覆盖不足：{latest_count} 只，低于最近 20 日中位数 "
                f"{median:g} 只的 {float(C.PREFLIGHT_MIN_COVERAGE):.0%}"
            )
    return failures

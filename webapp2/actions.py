"""前端写操作：全部是不依赖 dash 的纯函数，回调只是薄调用，CLI 与前端行为一致。

每个函数返回 (ok: bool, message: str)，不抛异常（异常转成 ok=False 的消息）。
路径经 webapp2.data 的模块变量解析，测试 monkeypatch 即可隔离真实目录。

可写的东西只有这些，别无其它：
  outputs/account.csv            现金维护（追加一行）
  outputs/holdings.csv           成交确认/手动校正（与 CLI 共用 execution.*）
  outputs/holdings_history.csv   每次持仓变更的完整历史快照
  outputs/execution_state_*.json 执行勾选/拒绝
  outputs/trading_settings.json 个性化费用与撮合设置（新 CLI 读取）
  models/                        晋升 / 删除 staging（与 --promote 共用 model_store.promote）
  factor_library/custom.yaml     因子草稿
  outputs/runs/<id>/             终止运行（只发信号，不改 run.json，由进程自己写终态）
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

from . import data

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_FACTOR_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
PY = os.path.join(data.ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable


def _src_on_path() -> None:
    """actions 需要 import src/ 下的 execution/model_store；入口脚本通常已注入，这里兜底。"""
    for p in (data.ROOT, os.path.join(data.ROOT, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _audit(line: str) -> None:
    """models/ 与 outputs/ 的写操作审计，追加一行；失败静默（审计是旁路）。"""
    try:
        os.makedirs(data.MODELS, exist_ok=True)
        with open(os.path.join(data.MODELS, "promotions.log"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")
    except OSError:
        pass


# ------------------------------------------------------------------ 现金
def save_cash(cash, day_iso: str | None = None) -> tuple[bool, str]:
    """向 account.csv 追加 {date, cash}。文件不存在则创建；表头缺 date/cash 则拒绝（防错位）。"""
    try:
        v = float(cash)
    except (TypeError, ValueError):
        return False, "请输入数字"
    if v != v or v in (float("inf"), float("-inf")) or v < 0:
        return False, "现金必须是非负有限数"
    day = day_iso or datetime.now().date().isoformat()
    path = os.path.join(data.OUTPUTS, "account.csv")
    try:
        os.makedirs(data.OUTPUTS, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("date,cash\n")
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip()
        cols = [c.strip() for c in header.split(",") if c.strip()]
        if "cash" not in cols:
            return False, f"account.csv 表头缺 cash 列（实际：{header!r}），请手工修复"
        row = {"date": day, "cash": f"{v:.2f}"}
        with open(path, "a", encoding="utf-8") as f:
            f.write(",".join(row.get(c, "") for c in cols) + "\n")
    except Exception as exc:
        return False, f"保存失败：{type(exc).__name__}: {exc}"
    data.refresh()
    _audit(f"CASH {v:.2f} day={day}")
    return True, f"已保存现金 ¥{v:,.2f}"


# ------------------------------------------------------------------ 持仓精确校正
def holdings_editor_text() -> str:
    """总览编辑器的 CSV 文本；零持仓也返回表头，便于新增第一笔持仓。"""
    _src_on_path()
    from execution import HOLDING_COLUMNS

    return data.holdings().reindex(columns=HOLDING_COLUMNS).to_csv(index=False)


def save_holdings_editor(text: str) -> tuple[bool, str]:
    """校验并替换精确持仓状态，旧状态留在历史快照中，不改变订单执行标记。"""
    _src_on_path()
    try:
        from execution import parse_holdings_editor, replace_holdings

        holdings = parse_holdings_editor(text)
        path = os.path.join(data.OUTPUTS, "holdings.csv")
        replace_holdings(path, holdings, datetime.now().date())
        n = len(holdings)
    except Exception as exc:
        return False, f"持仓校正失败：{type(exc).__name__}: {exc}"
    data.refresh()
    _audit(f"EDIT_HOLDINGS n_hold={n} by=webapp2")
    return True, f"已保存精确持仓：{n} 只；完整快照已追加到 holdings_history.csv"


# ------------------------------------------------------------------ 个性化交易费用与撮合设置
def trading_preferences() -> dict:
    _src_on_path()
    from trading_settings import load_trading_settings

    return load_trading_settings(os.path.join(data.OUTPUTS, "trading_settings.json"))


def save_trading_preferences(values: dict) -> tuple[bool, str, dict]:
    """保存个人交易设置，并同步当前进程的默认值；新 CLI 在 import config 时读取同一文件。"""
    _src_on_path()
    try:
        from trading_settings import save_trading_settings

        path = os.path.join(data.OUTPUTS, "trading_settings.json")
        saved = save_trading_settings(values, path)
        import config as C

        for key in saved:
            env_name = key.upper()
            if env_name not in os.environ:
                setattr(C, env_name, saved[key])
    except Exception as exc:
        return False, f"交易设置保存失败：{type(exc).__name__}: {exc}", {}
    data.refresh()
    _audit(f"TRADING_SETTINGS {json.dumps(saved, sort_keys=True)}")
    return (
        True,
        "已保存交易设置；新启动的出单和回测生效，进行中的任务不变；环境变量仍优先",
        saved,
    )


# ------------------------------------------------------------------ 执行勾选
def _rid(side: str, symbol: str) -> str:
    return f"{side}-{symbol}"


def set_execution_mark(day: str, side: str, symbol: str, mark: str | None) -> tuple[bool, str]:
    """mark ∈ done | rejected | None（清除）。键格式与 execution.load_rejected_rids 一致。"""
    if not re.fullmatch(r"\d{8}", str(day)):
        return False, f"非法日期 {day!r}"
    if mark not in ("done", "rejected", None):
        return False, f"非法标记 {mark!r}"
    path = os.path.join(data.OUTPUTS, f"execution_state_{day}.json")
    state = data.execution_state(day)
    key = _rid(side, symbol)
    if mark is None:
        state.pop(key, None)
    else:
        state[key] = mark
    try:
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        return False, f"保存失败：{exc}"
    data.refresh()
    return True, "已记录"


# ------------------------------------------------------------------ 成交确认
def today_str() -> str:
    """今天 YYYYMMDD。执行勾选与 --commit 都按「执行当天」落盘：订单当晚生成、次日执行。
    独立成函数，测试 monkeypatch 它即可。"""
    return datetime.now().strftime("%Y%m%d")


def already_confirmed(day: str | None = None) -> bool:
    """holdings.csv 不早于最新订单 ⇒ 这批订单大概率已确认过。重复确认会把买单再加一遍。"""
    _, odate = data.latest_orders()
    o_mt, h_mt = data.orders_mtime(day or odate), data.holdings_mtime()
    return o_mt is not None and h_mt is not None and h_mt >= o_mt


def confirm_fills(day: str | None = None, force: bool = False) -> tuple[bool, str]:
    """按最新订单（剔除拦截单与被拒绝单）推算并写回 holdings.csv，等价于 --commit。

    两道防呆：订单已过期（行情已到更新的交易日）一律拒绝；holdings.csv 已晚于订单时
    默认拒绝（重复确认会重复加仓），force=True 才放行。
    """
    _src_on_path()
    try:
        from execution import load_holdings_table, load_rejected_rids, save_holdings

        from .trade_logic import orders_fresh

        orders, odate = data.latest_orders()
        if orders.empty or odate is None:
            return False, "没有订单可确认（最新订单为空或不存在）"
        if day is not None and odate != day:
            return False, f"最新订单日期是 {odate}，与请求的 {day} 不一致"
        if not orders_fresh(odate, data.latest_data_day()):
            return False, f"订单 {odate} 已过期（行情已到 {data.latest_data_day()}），不能确认成交"
        if not force and already_confirmed(odate):
            return False, "holdings.csv 比订单新，可能已确认过；重复确认会重复加仓"
        orders = orders.copy()
        blocked = (
            orders["blocked"].astype(str).str.lower().isin(["true", "1"])
            if "blocked" in orders.columns
            else pd.Series(False, index=orders.index)
        )
        rej = load_rejected_rids(data.OUTPUTS, today_str())
        rids = pd.Series(
            [f"{r.side}-{r.symbol}" for r in orders.itertuples(index=False)], index=orders.index
        )
        blocked = blocked | rids.isin(rej)
        if int((~blocked).sum()) == 0:
            return False, "没有可确认成交的订单（全部被拦截或拒绝）"
        orders["blocked"] = blocked.to_numpy()
        path = os.path.join(data.OUTPUTS, "holdings.csv")
        tbl = load_holdings_table(path)
        current = (
            {str(s): int(sh) for s, sh in zip(tbl["symbol"], tbl["shares"])} if len(tbl) else {}
        )
        asof = (
            pd.Timestamp(orders["asof_date"].dropna().iloc[-1]) if "asof_date" in orders else odate
        )
        save_holdings(path, orders, current, asof)
        after = load_holdings_table(path)
        n = int((pd.to_numeric(after["shares"], errors="coerce").fillna(0) > 0).sum())
    except Exception as exc:
        return False, f"确认失败：{type(exc).__name__}: {exc}"
    data.refresh()
    _audit(f"CONFIRM_FILLS day={odate} n_hold={n}")
    return True, f"已写回持仓，当前 {n} 只（假设未拦截订单全部成交，请按券商实际成交核对）"


# ------------------------------------------------------------------ 模型
def _active_nightly() -> dict | None:
    import run_registry as rr

    busy = [
        m
        for m in rr.active_runs(root=os.path.join(data.OUTPUTS, "runs"))
        if m.get("kind") == "nightly"
    ]
    return busy[0] if busy else None


MIN_FORCED_REASON = 6


def promote_staging(model_id: str, reason: str = "") -> tuple[bool, str]:
    """晋升 staging 为 champion（旧 champion 归档 archive/）。

    候选未达门槛或评估可疑（model_logic.promote_plan.need_reason）时必须填写理由：
    绕过门控是人的决定，要留下记录。

    夜间任务运行中拒绝：它正在读 champion 工件，晋升会让同一晚的预测与门控用到不同模型。
    """
    if not isinstance(model_id, str) or not _ID_RE.fullmatch(model_id):
        return False, f"非法 model_id: {model_id!r}"
    busy = _active_nightly()
    if busy:
        return False, f"夜间任务 {busy['run_id']} 正在运行，等它结束再晋升"
    from .model_logic import promote_plan

    plan = promote_plan(model_id)
    if not plan["can"]:
        return False, plan["why"]
    if plan["need_reason"] and len((reason or "").strip()) < MIN_FORCED_REASON:
        return False, "该候选未通过门控或评估可疑，晋升必须填写理由（至少 6 个字）"
    _src_on_path()
    try:
        import model_store

        # model_store 按 config.MODEL_DIR 解析路径，页面按 data.MODELS 展示。两者不一致时，
        # 页面上看到的候选和实际被晋升的会是不同目录里的东西，必须拒绝而不是碰运气。
        if os.path.abspath(model_store.C.MODEL_DIR) != os.path.abspath(data.MODELS):
            return False, (
                f"模型目录不一致：页面读 {data.MODELS}，晋升会作用于 {model_store.C.MODEL_DIR}。"
                "已拒绝，避免晋升到页面上看不到的模型。"
            )
        record = {
            "reason": reason or "前端人工晋升",
            "decided_at": datetime.now().isoformat(timespec="seconds"),
        }
        model_store.promote(model_id, promotion_record=record)
    except Exception as exc:
        return False, f"晋升失败：{type(exc).__name__}: {exc}"
    data.refresh()
    _audit(f"PROMOTE {model_id} by=webapp2 reason={(reason or '').strip()[:120]!r}")
    return True, f"已晋升 {model_id}，旧 champion 在 models/archive/，可回滚"


def delete_staging(model_id: str) -> tuple[bool, str]:
    import shutil

    if not isinstance(model_id, str) or not _ID_RE.fullmatch(model_id):
        return False, f"非法 model_id: {model_id!r}"
    busy = _active_nightly()
    if busy:
        return False, f"夜间任务 {busy['run_id']} 正在运行，门控可能正在读取候选，等它结束再删除"
    d = os.path.join(data.MODELS, "staging", model_id)
    if not os.path.isdir(d):
        return False, f"staging 候选不存在：{model_id}"
    try:
        shutil.rmtree(d)
    except Exception as exc:
        return False, f"删除失败：{exc}"
    data.refresh()
    _audit(f"DELETE_STAGING {model_id} by=webapp2")
    return True, f"已删除 staging 候选 {model_id}"


# ------------------------------------------------------------------ 启动 CLI
# 只允许这几个入口：前端不拼任意命令，只能从白名单挑一个并带参数。
CLI_ENTRIES = {
    "nightly": ["predict_tomorrow.py"],
    "walkforward": ["scripts/run_walkforward.py"],
    "impact": ["scripts/research/impact_sensitivity.py"],
    "tracking": ["predict_tomorrow.py", "--tracking-report"],
    # pipeline 研究模式：统一经 scripts/run_pipeline.py 登记到 outputs/runs/
    "train": ["scripts/run_pipeline.py", "--mode", "train"],
    "modeleval": ["scripts/run_pipeline.py", "--mode", "modeleval"],
    "factoreval": ["scripts/run_pipeline.py", "--mode", "factoreval"],
}
_SAFE_ARG = re.compile(r"^[A-Za-z0-9_.,:/=+\-]*$")

def nightly_preset(profile: str | None = None) -> list[str]:
    """Build the web preset from the same capital profile used by the CLI."""
    _src_on_path()
    from capital import capital_profile_params

    profile = profile or os.environ.get("CAPITAL_PROFILE", "standard")
    params = capital_profile_params(profile)
    capital = os.environ.get("NIGHTLY_CAPITAL", "20000" if profile == "small" else "30000")
    args = ["--capital", capital, "--capital-profile", profile]
    for key, flag in (
        ("max_positions", "--max-positions"),
        ("max_weight", "--max-weight"),
        ("min_notional", "--min-notional"),
    ):
        fallback = {"max_positions": 6, "max_weight": 0.2, "min_notional": 2000}[key]
        args.extend([flag, str(params.get(key, fallback))])
    args.extend(["--allowed-boards", "main", "--skip-unaffordable"])
    return args


# 前端「生成次日订单」与运行页按钮共用同一份资金配置。
NIGHTLY_PRESET = nightly_preset()

# 重型任务（加载全市场面板，数 GB 内存）同时只允许一个在跑，不分种类：
# 夜间出单、回测、训练、评估并发会一起读写同一份面板缓存，32GB 机器上实测会 OOM。
_HEAVY_KINDS = {"nightly", "walkforward", "train", "modeleval", "factoreval", "impact"}


def launch_cli(entry: str, args: list[str] | None = None) -> tuple[bool, str, str | None]:
    """后台启动一个 CLI，返回 (ok, 消息, run_id)。

    前端不创建 run 目录：CLI 自己登记（RunReporter）。这里预先指定 RUN_ID 环境变量，
    使前端能立刻跳转到 /runs/<run_id>，即使 CLI 还没来得及写第一帧。
    stdout/stderr 重定向到 outputs/runs/<run_id>/run.log。
    """
    if entry not in CLI_ENTRIES:
        return False, f"未知入口 {entry!r}", None
    args = list(args or [])
    for a in args:
        if not isinstance(a, str) or not _SAFE_ARG.fullmatch(a):
            return False, f"参数含不允许的字符：{a!r}", None
    kind = {"nightly": "nightly", "walkforward": "walkforward", "impact": "impact"}.get(
        entry, entry
    )
    if kind in _HEAVY_KINDS:
        import run_registry as rr

        busy = [
            m
            for m in rr.active_runs(root=os.path.join(data.OUTPUTS, "runs"))
            if m.get("kind") in _HEAVY_KINDS
        ]
        if busy:
            b = busy[0]
            return (
                False,
                f"已有重型任务在运行：{b.get('label') or b.get('kind')}（{b['run_id']}）。"
                "它们共用因子面板缓存且各占数 GB 内存，请等它结束",
                b["run_id"],
            )
    rid = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(data.OUTPUTS, "runs", rid)
    n = 1
    while os.path.exists(run_dir):
        rid = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}_{n}"
        run_dir, n = os.path.join(data.OUTPUTS, "runs", rid), n + 1
    os.makedirs(run_dir, exist_ok=True)
    cmd = [PY, "-u", *CLI_ENTRIES[entry], *args]
    env = {**os.environ, "RUN_ID": rid}
    try:
        logf = open(os.path.join(run_dir, "run.log"), "w")
        try:
            subprocess.Popen(
                cmd,
                cwd=data.ROOT,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            logf.close()
    except Exception as exc:
        return False, f"启动失败：{type(exc).__name__}: {exc}", None
    _audit(f"LAUNCH {entry} {' '.join(args)} run={rid}")
    return True, f"已启动 {entry}，运行 {rid}", rid


def stop_run(run_id: str) -> tuple[bool, str]:
    """向运行进程组发 SIGTERM。只读 run.json 取 pid，且必须 liveness 为 alive/stale。"""
    import run_registry as rr

    meta = rr.read_run(run_id, root=os.path.join(data.OUTPUTS, "runs"))
    if meta is None:
        return False, f"运行不存在：{run_id}"
    if meta.get("liveness") not in ("alive", "stale"):
        return False, f"该运行不在运行中（{meta.get('status')}/{meta.get('liveness')}）"
    pid = meta.get("pid")
    if not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
        return False, "pid 不可用"
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            return False, f"终止失败：{exc}"
    _audit(f"STOP run={run_id} pid={pid}")
    return True, f"已向 {run_id}（pid {pid}）发送终止信号"


# ------------------------------------------------------------------ 因子评估
DEFAULT_PANEL_LOOKBACK_DAYS = 912  # 约 2.5 年，与夜间任务的面板范围一致
DRAFT_LOOKBACK_DAYS = 730  # 与旧前端「立即评估该因子」一致


def full_eval_lookback_days(today=None) -> int:
    """全库评估的回看天数：取到因子面板缓存的起点为止，不再往前。

    默认 1825 天会越过缓存起点（面板缓存只覆盖约 2.5 年）：那样「请求区间 ⊄ 缓存」，
    load_factor_panel 会重算并用更长区间覆盖缓存，夜间任务随后每次都要读一份更大的面板。
    回看天数只会让 start 更晚（运行日期 ≥ 点击日期），所以按点击当天算是安全侧。
    """
    meta = data.panel_cache_meta() or {}
    start = meta.get("req_start") or meta.get("data_start")
    if not start:
        return DEFAULT_PANEL_LOOKBACK_DAYS
    try:
        day = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
        return int(min(3650, max(120, (day - pd.Timestamp(start)).days)))
    except Exception:
        return DEFAULT_PANEL_LOOKBACK_DAYS


def plan_factor_eval(factor_id: str, library_ids, today=None) -> list[str]:
    """单因子评估的 CLI 参数。

    库里的公式因子：默认不在因子面板里（USE_FORMULA_FACTORS=0），必须用 --draft 现算，
    且 --draft 不读写面板缓存；原生因子：直接取缓存面板，范围同全库评估。
    """
    fid = str(factor_id or "").strip()
    # 与 factor_draft / run_pipeline --draft 同一规则：合法列名（字母或下划线开头，≤64 字符）。
    # 不复用 _ID_RE：它是模型目录名规则，允许数字开头、连字符与任意长度。
    if not _FACTOR_ID_RE.fullmatch(fid):
        raise ValueError(f"非法因子 id：{factor_id!r}")
    if fid in set(library_ids):
        return ["--draft", fid, "--set", f"lookback_days={DRAFT_LOOKBACK_DAYS}"]
    return [
        "--set",
        f"factor_subset={fid}",
        "--set",
        f"lookback_days={full_eval_lookback_days(today)}",
    ]


def launch_factor_eval(factor_id: str) -> tuple[bool, str, str | None]:
    try:
        args = plan_factor_eval(factor_id, set(data.factor_library()["id"]))
    except ValueError as exc:
        return False, str(exc), None
    return launch_cli("factoreval", args)


def launch_full_factor_eval(with_diagnostics: bool = False) -> tuple[bool, str, str | None]:
    args = ["--set", f"lookback_days={full_eval_lookback_days()}"]
    if with_diagnostics:
        args += ["--set", "with_diagnostics=1", "--set", "diag_with_model=1"]
    return launch_cli("factoreval", args)


# ------------------------------------------------------------------ 因子草稿
def save_factor_draft(
    fid, name, category, formula, tags="", expected_sign=None, description=""
) -> tuple[bool, str, dict]:
    """校验并追加到 factor_library/custom.yaml（enabled=false）。返回 (ok, 消息, 校验详情)。

    校验用生产解析器；id 与整个库（含 gtja/momentum/... 各文件）及面板原生因子去重。
    保存后因子库哈希变化，下一次读取因子面板会触发一次全量重建——消息里说明，不让用户意外。
    """
    _src_on_path()
    from factor_draft import save_draft, validate_draft

    lib = data.factor_library()
    existing = set(lib["id"]) | set(data.panel_factor_columns())
    res = validate_draft(
        fid,
        name,
        category,
        formula,
        tags,
        expected_sign,
        description,
        existing_ids=existing,
        known_columns=data.base_columns() | set(lib["id"]),
    )
    if res["errors"]:
        return False, "校验未通过：" + "；".join(res["errors"]), res
    path = os.path.join(data.FACTOR_LIBRARY, "custom.yaml")
    try:
        save_draft(res["record"], path, existing_ids=set(lib["id"]))
    except Exception as exc:
        return False, f"保存失败：{type(exc).__name__}: {exc}", res
    data.refresh()
    _audit(f"FACTOR_DRAFT {res['record']['id']} by=webapp2")
    return (
        True,
        f"已保存草稿 {res['record']['id']}（enabled=false）。因子库内容变了，"
        "下一次读取因子面板会全量重建一次。",
        res,
    )

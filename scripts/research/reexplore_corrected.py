"""重新探索(价格修正后): 在缓存的 corrected ic_linear OOS 上测两个数据指向的假设。

修正未复权bug后,分层诊断显示 edge 在【空头侧/中部】:D0(最不看好)显著跑输
−1.19%,D9(最看好)反而也跑输 −0.14%,中部 D3-D6 最好。这指向两个 long-only
"买顶层"之外的方向:
  H1 市场中性多空(long top / short bottom): 之前在【带bug数据】上是负alpha,
     但空头侧(D0)正是除权伪信号聚集处,修正后可能不同。dollar-neutral→beta≈0,
     收益≈纯alpha,对收益序列直接做 NW-t。
  H2 long-only 剔除底部(持有非-D0 等权): 数据指向"避开最差"而非"精选最好"。

⚠️诚实标注: 分层结论是在同一段 OOS 上看出的,据此选策略规则有轻度 snooping;
"避开最差decile"算较自然的先验规则,但仍应视为探索性,不是独立样本外证据。
⚠️用已缓存的 corrected ic_linear OOS(wholemkt_oos.parquet),纯仓位+回测,不训练。
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import config as C
from walkforward import _with_ids
from strategy import generate_positions, _long_short_topk
from backtest import run_backtest_periodic
from measure_alpha import _newey_west_alpha_t

oos = pd.read_parquet("data_cache/wholemkt_panel.parquet", columns=["date","symbol","amount"]) \
    if False else None
pred = pd.read_parquet("data_cache/wholemkt_oos.parquet")
amt = pd.read_parquet("data_cache/wholemkt_panel.parquet", columns=["date","symbol","amount"])
pred = pred.merge(amt, on=["date","symbol"], how="left")
print(f"corrected ic_linear OOS: {len(pred):,} rows, {pred['symbol'].nunique()} stocks, {pred['date'].nunique()} days")
ids = _with_ids(pred)
ids["amount"] = pred["amount"].to_numpy()
p = pred["prediction"].to_numpy()
bench = pred.groupby("date")["return_1d"].mean().sort_index().to_numpy()
REB = C.REBALANCE_DAYS

def report(tag, pos, neutral):
    res = run_backtest_periodic(ids, p, pos, rebalance_days=REB, pred_in_return_space=True)
    m = res["metrics"]; net = res["curve"]["net"].to_numpy()
    n = min(len(net), len(bench))
    if neutral:  # dollar-neutral: return IS alpha, test return series vs 0
        a_d, beta, t_nw, _ = _newey_west_alpha_t(net[:n], np.zeros(n)+1e-12, REB)
        # NW-t on the raw return series mean: use intercept-only
        mu = net.mean(); sd_nw = _nw_se(net, REB)
        t = mu/sd_nw if sd_nw>1e-18 else 0.0
        cum = np.prod(1+net)-1
        print(f"  {tag}: 年化={mu*252:+.2%} Sharpe={m['sharpe']:+.2f} 累计={cum:+.1%} "
              f"NW-t(收益≠0)={t:+.2f} {'显著✓' if abs(t)>=2 else '不显著'} 成本佣金={m['total_cost_commission']:.4f} 冲击={m['total_cost_impact']:.4f}")
    else:
        a_d, beta, t_nw, _ = _newey_west_alpha_t(net[:n], bench[:n], REB)
        cum = np.prod(1+net)-1; bcum=np.prod(1+bench[:n])-1
        print(f"  {tag}: 年化Alpha={a_d*252:+.2%} Beta={beta:+.2f} Sharpe={m['sharpe']:+.2f} "
              f"累计={cum:+.1%}(基准{bcum:+.1%}) NW-t={t_nw:+.2f} {'显著✓' if abs(t_nw)>=2 else '不显著'}")

def _nw_se(x, lags):
    x = np.asarray(x,float); n=len(x); mu=x.mean(); e=x-mu
    g0=(e@e)/n; s=g0
    for L in range(1,lags+1):
        w=1-L/(lags+1); s+=2*w*(e[L:]@e[:-L])/n
    return np.sqrt(s/n)

print("\n=== H1: 市场中性多空 long_short_topk (dollar-neutral, 收益≈纯alpha) ===")
for tk in [0.05, 0.10, 0.20]:
    C.TOPK_RATIO = tk
    pos = generate_positions(ids, p, strategy="long_short_topk")
    report(f"top/bottom {int(tk*100)}% (含冲击成本)", pos, neutral=True)

print("\n  对照: 关闭冲击成本(毛,看信号本身)")
C.USE_IMPACT_COST = False
for tk in [0.10]:
    C.TOPK_RATIO = tk
    pos = generate_positions(ids, p, strategy="long_short_topk")
    report(f"top/bottom {int(tk*100)}% (无冲击成本)", pos, neutral=True)
C.USE_IMPACT_COST = True

print("\n=== H2: long-only 剔除底部 decile (持有非-D0 等权) vs 等权全市场 ===")
def exclude_bottom_pos(ids, pred_arr, drop_frac):
    df = pd.DataFrame({"date": ids[C.DATE_COL].to_numpy(), "p": pred_arr})
    rank = df.groupby("date")["p"].rank(pct=True, method="first")
    keep = (rank > drop_frac).to_numpy().astype(float)
    # equal weight among kept, per date, sum=1
    w = np.zeros(len(df))
    for d in np.unique(df["date"].to_numpy()):
        mask = df["date"].to_numpy()==d
        k = keep[mask]; s=k.sum()
        if s>0: w[mask] = k/s
    return w
for drop in [0.10, 0.20, 0.30]:
    pos = exclude_bottom_pos(ids, p, drop)
    report(f"剔除底部 {int(drop*100)}% (持有其余等权)", pos, neutral=False)
print("\n判读: H1 dollar-neutral 下 NW-t≥2 才算真信号; H2 是 long-only 可吃的'剔除式'。")
print("⚠️规则受同段OOS分层启发, 轻度snooping; 真正确认需独立样本/前瞻规则。")

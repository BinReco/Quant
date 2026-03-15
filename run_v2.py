"""
小市值 + 量价因子策略 V2 - 完整回测

数据源: baostock (免费无限流)
策略: 小市值 + 放量突破 + 大盘择时 + 移动止盈止损
调仓: 周度调仓（每周一）

流程:
1. 批量下载全市场股票日线数据
2. 每个调仓日：
   a. 判断大盘状态 → 决定目标仓位数
   b. 检查持仓止损/止盈/到期
   c. 从全市场选股，买入新标的
3. 输出绩效
"""

import sys
import os
import pickle
import time
from dataclasses import dataclass, field

import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import INITIAL_CAPITAL, MAX_POSITIONS

# ============ 参数 ============
BT_START = "2025-08-01"
BT_END = "2026-02-26"
DATA_START = "2025-05-01"  # 预热数据
MAX_HOLD_DAYS = 10
STOP_LOSS = -0.05          # 固定止损 -5%
TRAIL_STOP_RATIO = 0.5     # 从最高盈利回撤 50% 止盈
REBALANCE_DAY = 0          # 周一调仓 (0=Monday)
CACHE_FILE = "cache/all_stocks_v2.pkl"


# ============ 数据获取 ============

def download_all_stocks():
    """下载全市场股票日线数据（逐只缓存 + 断点续传 + 超时重连）"""
    # 如果已有完整缓存，直接加载
    if os.path.exists(CACHE_FILE):
        print("  从完整缓存加载...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    stock_cache_dir = "cache/stocks"
    os.makedirs(stock_cache_dir, exist_ok=True)

    lg = bs.login()

    # 获取全部 A 股列表
    rs = bs.query_stock_basic(code_name="", code="")
    stock_list = rs.get_data()
    stock_list = stock_list[stock_list["type"] == "1"]
    codes = stock_list["code"].tolist()
    print(f"  全 A 股数量: {len(codes)}")

    # 检查已有缓存
    cached = set()
    for f in os.listdir(stock_cache_dir):
        if f.endswith(".pkl"):
            cached.add(f.replace(".pkl", "").replace("_", "."))
    print(f"  已缓存: {len(cached)}, 待下载: {len(codes) - len(cached)}")

    fail_count = 0
    max_consecutive_fails = 20  # 连续失败超过此数则重连

    for i, code in enumerate(tqdm(codes, desc="  下载股票数据")):
        # 已缓存则跳过
        if code in cached:
            continue

        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg,isST",
                start_date=DATA_START,
                end_date=BT_END,
                frequency="d",
                adjustflag="2",
            )
            df = rs.get_data()

            if df.empty or len(df) < 20:
                fail_count = 0
                continue

            for col in ["open", "high", "low", "close", "volume", "amount",
                         "turn", "pctChg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"])

            if len(df) < 20:
                fail_count = 0
                continue

            # 逐只保存
            cache_name = code.replace(".", "_")
            with open(f"{stock_cache_dir}/{cache_name}.pkl", "wb") as f:
                pickle.dump(df, f)
            cached.add(code)
            fail_count = 0

        except Exception as e:
            fail_count += 1
            if fail_count >= max_consecutive_fails:
                print(f"\n  连续失败 {fail_count} 次，重连 baostock...")
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(2)
                bs.login()
                fail_count = 0

    bs.logout()

    # 合并所有缓存为一个大文件
    print("  合并缓存文件...")
    all_data = {}
    for f in os.listdir(stock_cache_dir):
        if not f.endswith(".pkl"):
            continue
        code = f.replace(".pkl", "").replace("_", ".")
        with open(f"{stock_cache_dir}/{f}", "rb") as fh:
            all_data[code] = pickle.load(fh)

    os.makedirs("cache", exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(all_data, f)

    print(f"  有效股票: {len(all_data)}")
    return all_data


def download_index():
    """下载上证指数"""
    cache_path = "cache/sh_index_v2.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        "sh.000001",
        "date,close",
        start_date=DATA_START,
        end_date=BT_END,
        frequency="d",
    )
    df = rs.get_data()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    bs.logout()

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def download_style_indices():
    """下载沪深300和中证1000指数，用于大小盘风格判断"""
    cache_path = "cache/style_indices.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    lg = bs.login()
    indices = {}
    for code, name in [("sh.000300", "hs300"), ("sh.000852", "zz1000")]:
        rs = bs.query_history_k_data_plus(
            code, "date,close",
            start_date=DATA_START, end_date=BT_END,
            frequency="d",
        )
        df = rs.get_data()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"])
        indices[name] = df
    bs.logout()

    os.makedirs("cache", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(indices, f)
    return indices


# ============ 策略 ============

def market_regime(index_df: pd.DataFrame, date) -> str:
    """大盘择时"""
    df = index_df[index_df["date"] <= date]
    if len(df) < 60:
        return "neutral"
    close = df["close"].iloc[-1]
    ma20 = df["close"].rolling(20).mean().iloc[-1]
    ma60 = df["close"].rolling(60).mean().iloc[-1]
    if close > ma20:
        return "bull"
    elif close > ma60:
        return "neutral"
    else:
        return "bear"


def style_regime(style_indices: dict, date) -> str:
    """
    大小盘风格判断

    用 中证1000/沪深300 比值的均线交叉判断：
    - MA10 > MA30 → "small"（小盘占优）
    - MA10 < MA30 → "large"（大盘占优）
    - 差距很小    → "mixed"（不确定）
    """
    hs300 = style_indices["hs300"]
    zz1000 = style_indices["zz1000"]

    # 对齐日期
    h = hs300[hs300["date"] <= date].copy()
    z = zz1000[zz1000["date"] <= date].copy()
    if len(h) < 35 or len(z) < 35:
        return "mixed"

    merged = pd.merge(h, z, on="date", suffixes=("_300", "_1000"))
    if len(merged) < 35:
        return "mixed"

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["ratio"] = merged["close_1000"] / merged["close_300"]
    merged["ratio_ma10"] = merged["ratio"].rolling(10).mean()
    merged["ratio_ma30"] = merged["ratio"].rolling(30).mean()

    last = merged.iloc[-1]
    if pd.isna(last["ratio_ma10"]) or pd.isna(last["ratio_ma30"]):
        return "mixed"

    diff_pct = (last["ratio_ma10"] - last["ratio_ma30"]) / last["ratio_ma30"]

    if diff_pct > 0.005:     # MA10 高于 MA30 超过 0.5%
        return "small"
    elif diff_pct < -0.005:  # MA10 低于 MA30 超过 0.5%
        return "large"
    else:
        return "mixed"


def calc_market_cap_rank(all_data: dict, date) -> dict:
    """
    近似市值排名（用成交额代替，因为 baostock 免费版没有市值字段）
    成交额越大通常市值越大
    """
    cap_map = {}
    for code, df in all_data.items():
        row = df[df["date"] == date]
        if not row.empty:
            cap_map[code] = row.iloc[0]["amount"]

    if not cap_map:
        return {}

    # 排名（百分比）
    series = pd.Series(cap_map)
    ranks = series.rank(pct=True)
    return ranks.to_dict()


def select_stocks(all_data: dict, date, cap_ranks: dict,
                  exclude: set, max_picks: int = 5) -> list:
    """选股（始终选中小盘，因为量价因子在中小盘有效）"""
    cap_lo, cap_hi = 0.15, 0.65

    candidates = []

    for code, df in all_data.items():
        if code in exclude:
            continue

        hist = df[df["date"] <= date].tail(25)
        if len(hist) < 20:
            continue

        today = hist.iloc[-1]
        close = today["close"]
        pct = today["pctChg"]
        turn = today["turn"]
        vol = today["volume"]
        is_st = today["isST"]

        # ---- 过滤 ----
        if is_st == "1":
            continue
        if vol <= 0 or close <= 0:
            continue
        if pct > 9.5 and today["open"] == today["high"] == today["low"]:
            continue  # 一字涨停
        if pct < -9.5:
            continue  # 跌停
        if pct > 8:
            continue  # 涨太多
        if turn < 2 or turn > 12:
            continue

        # 市值过滤（根据风格动态调整范围）
        cap_rank = cap_ranks.get(code, 0.5)
        if cap_rank < cap_lo or cap_rank > cap_hi:
            continue

        # ---- 因子计算 ----
        closes = hist["close"].values
        vols = hist["volume"].values

        # 5日动量
        if len(closes) >= 6:
            mom_5d = closes[-1] / closes[-6] - 1
        else:
            continue
        if mom_5d < 0 or mom_5d > 0.15:
            continue

        # 20日均线
        ma20 = closes[-20:].mean()
        if closes[-1] < ma20:
            continue

        # 放量
        vol_ma5 = vols[-5:].mean() if vols[-5:].mean() > 0 else 1
        vol_ratio = vol / vol_ma5

        # 突破
        high_20 = closes[-20:].max()
        breakout = closes[-1] / high_20

        # 打分
        vol_score = min(vol_ratio / 2.0, 1.0)
        breakout_score = min(breakout, 1.0)

        if 0.03 <= mom_5d <= 0.08:
            mom_score = 1.0
        elif mom_5d < 0.03:
            mom_score = 0.5
        else:
            mom_score = 0.6

        if 3 <= turn <= 8:
            turn_score = 1.0
        elif turn < 3:
            turn_score = 0.6
        else:
            turn_score = 0.5

        score = (vol_score * 0.3 + breakout_score * 0.3
                 + mom_score * 0.25 + turn_score * 0.15)

        candidates.append({
            "code": code,
            "close": close,
            "score": score,
        })

    if not candidates:
        return []

    df = pd.DataFrame(candidates)
    df = df.sort_values("score", ascending=False)
    return df.head(max_picks).to_dict("records")


# ============ 回测引擎 ============

@dataclass
class Position:
    code: str
    buy_date: str
    buy_price: float
    amount: float
    hold_days: int = 0
    max_price: float = 0.0

    def __post_init__(self):
        self.max_price = self.buy_price


def run_backtest(all_data: dict, index_df: pd.DataFrame,
                 style_indices: dict = None):
    """执行回测"""
    cash = float(INITIAL_CAPITAL)
    positions: dict[str, Position] = {}
    trades = []
    daily_records = []

    # 交易日列表
    sample_code = list(all_data.keys())[0]
    sample_df = all_data[sample_code]
    trading_dates = sorted(
        sample_df[
            (sample_df["date"] >= BT_START) &
            (sample_df["date"] <= BT_END)
        ]["date"].unique()
    )

    for date in tqdm(trading_dates, desc="回测中"):
        date_str = str(date)[:10]
        weekday = pd.Timestamp(date).weekday()

        # 获取当日价格
        price_map = {}
        for code, df in all_data.items():
            row = df[df["date"] == date]
            if not row.empty:
                price_map[code] = {
                    "close": row.iloc[0]["close"],
                    "high": row.iloc[0]["high"],
                    "low": row.iloc[0]["low"],
                    "pctChg": row.iloc[0]["pctChg"],
                }

        # 1. 更新持仓
        for code, pos in positions.items():
            if pos.buy_date != date_str:
                pos.hold_days += 1
            p = price_map.get(code)
            if p:
                pos.max_price = max(pos.max_price, p["high"])

        # 2. 检查卖出（每天都检查止损止盈）
        to_sell = []
        for code, pos in positions.items():
            if pos.buy_date == date_str:  # T+1
                continue
            p = price_map.get(code)
            if not p:
                continue

            current = p["close"]
            pnl_pct = current / pos.buy_price - 1
            max_pnl = pos.max_price / pos.buy_price - 1

            reason = ""
            # 固定止损
            if pnl_pct <= STOP_LOSS:
                reason = "止损"
            # 移动止盈：从最高点回撤超过50%
            elif max_pnl > 0.03 and pnl_pct < max_pnl * TRAIL_STOP_RATIO:
                reason = "移动止盈"
            # 到期
            elif pos.hold_days >= MAX_HOLD_DAYS:
                reason = "到期"

            if reason:
                to_sell.append((code, current, pnl_pct, reason))

        for code, price, pnl_pct, reason in to_sell:
            pos = positions[code]
            sell_value = pos.amount * (1 + pnl_pct)
            comm = max(sell_value * 0.0002, 5)
            tax = sell_value * 0.001
            pnl = sell_value - pos.amount - comm - tax
            cash += sell_value - comm - tax

            trades.append({
                "date": date_str, "code": code, "direction": "SELL",
                "price": price, "pnl": pnl, "pnl_pct": pnl_pct,
                "reason": reason, "hold_days": pos.hold_days,
            })
            del positions[code]

        # 3. 调仓日选股买入（每周一，或上周一休市则周二）
        is_rebalance = (weekday == REBALANCE_DAY) or (
            weekday == 1 and date == trading_dates[0]
        )
        # 也可以在有空仓时随时补仓
        empty_slots = MAX_POSITIONS - len(positions)

        if is_rebalance and empty_slots > 0:
            regime = market_regime(index_df, date)

            if regime == "bull":
                target_positions = MAX_POSITIONS
            elif regime == "neutral":
                target_positions = MAX_POSITIONS // 2
            else:
                target_positions = 0

            slots = min(empty_slots, target_positions - len(positions))
            if slots > 0:
                cap_ranks = calc_market_cap_rank(all_data, date)
                exclude = set(positions.keys())
                picks = select_stocks(
                    all_data, date, cap_ranks, exclude,
                    max_picks=slots,
                )

                for pick in picks:
                    code = pick["code"]
                    buy_price = pick["close"]
                    alloc = min(INITIAL_CAPITAL / MAX_POSITIONS, cash)
                    if alloc < 1000:
                        break
                    comm = max(alloc * 0.0002, 5)
                    invest = alloc - comm
                    cash -= alloc

                    positions[code] = Position(
                        code=code, buy_date=date_str,
                        buy_price=buy_price, amount=invest,
                    )
                    trades.append({
                        "date": date_str, "code": code,
                        "direction": "BUY", "price": buy_price,
                        "pnl": 0, "pnl_pct": 0,
                        "reason": f"选股({regime})",
                        "hold_days": 0,
                    })

        # 4. 记录每日净值
        mv = 0
        for code, pos in positions.items():
            p = price_map.get(code)
            if p:
                pnl_pct = p["close"] / pos.buy_price - 1
                mv += pos.amount * (1 + pnl_pct)
            else:
                mv += pos.amount

        total = cash + mv
        idx_row = index_df[index_df["date"] == date]
        bench = idx_row.iloc[0]["close"] if not idx_row.empty else 0

        daily_records.append({
            "date": date_str,
            "total_value": total,
            "cash": cash,
            "positions": len(positions),
            "benchmark": bench,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_records)


# ============ 主程序 ============

def main():
    print("=" * 55)
    print("  小市值+量价因子策略 V2 - 10只持仓")
    print("=" * 55)

    print("\n[1/4] 下载全市场股票数据...")
    all_data = download_all_stocks()
    print(f"  有效股票: {len(all_data)}")

    print("\n[2/4] 下载上证指数...")
    index_df = download_index()

    print("\n[3/4] 下载风格指数（沪深300 + 中证1000）...")
    style_indices = download_style_indices()
    print(f"  沪深300: {len(style_indices['hs300'])} 条")
    print(f"  中证1000: {len(style_indices['zz1000'])} 条")

    print("\n[4/4] 执行回测...")
    trades_df, daily_df = run_backtest(all_data, index_df, style_indices)

    # ---- 计算指标 ----
    final = daily_df["total_value"].iloc[-1]
    total_ret = final / INITIAL_CAPITAL - 1
    n_days = len(daily_df)
    annual_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

    peak = daily_df["total_value"].cummax()
    dd = ((daily_df["total_value"] - peak) / peak)
    max_dd = dd.min()

    daily_df["daily_ret"] = daily_df["total_value"].pct_change()
    rf = 0.02 / 252
    excess = daily_df["daily_ret"].dropna() - rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    sells = trades_df[trades_df["direction"] == "SELL"]
    if not sells.empty:
        win = sells[sells["pnl"] > 0]
        win_rate = len(win) / len(sells)
        avg_win = win["pnl"].mean() if len(win) > 0 else 0
        avg_loss = sells[sells["pnl"] <= 0]["pnl"].mean() if len(sells[sells["pnl"] <= 0]) > 0 else 0
        total_profit = win["pnl"].sum()
        total_loss = sells[sells["pnl"] <= 0]["pnl"].sum()
        pf = abs(total_profit / total_loss) if total_loss != 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = total_profit = total_loss = pf = 0

    # 基准收益
    bench_start = daily_df["benchmark"].iloc[0]
    bench_end = daily_df["benchmark"].iloc[-1]
    bench_ret = bench_end / bench_start - 1 if bench_start > 0 else 0

    print("\n" + "=" * 55)
    print("      小市值+量价因子 10只持仓 - 回测结果")
    print("=" * 55)
    print(f"  {'回测区间':>12s}: {BT_START} ~ {BT_END}")
    print(f"  {'初始资金':>12s}: {INITIAL_CAPITAL:,.0f}")
    print(f"  {'最终资金':>12s}: {final:,.0f}")
    print(f"  {'策略收益率':>12s}: {total_ret:.2%}")
    print(f"  {'基准收益率':>12s}: {bench_ret:.2%}")
    print(f"  {'超额收益':>12s}: {total_ret - bench_ret:.2%}")
    print(f"  {'年化收益率':>12s}: {annual_ret:.2%}")
    print(f"  {'最大回撤':>12s}: {max_dd:.2%}")
    print(f"  {'夏普比率':>12s}: {sharpe:.2f}")
    print(f"  {'交易次数':>12s}: {len(trades_df)}")
    print(f"  {'卖出次数':>12s}: {len(sells)}")
    print(f"  {'胜率':>12s}: {win_rate:.2%}")
    print(f"  {'盈亏比':>12s}: {pf:.2f}")
    print(f"  {'平均盈利':>12s}: {avg_win:,.0f}")
    print(f"  {'平均亏损':>12s}: {avg_loss:,.0f}")
    print("=" * 55)

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams["axes.unicode_minus"] = False
    for fn in ["Arial Unicode MS", "Heiti TC", "Hiragino Sans", "PingFang SC", "SimHei"]:
        try:
            font_manager.findfont(fn, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [fn]
            break
        except Exception:
            continue

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    daily_df["date_dt"] = pd.to_datetime(daily_df["date"])

    # 资金曲线 vs 基准
    ax1 = axes[0]
    ax1.plot(daily_df["date_dt"], daily_df["total_value"],
             label="策略", linewidth=1.5)
    if bench_start > 0:
        bench_norm = daily_df["benchmark"] / bench_start * INITIAL_CAPITAL
        ax1.plot(daily_df["date_dt"], bench_norm,
                 label="上证指数", linewidth=1, alpha=0.7, color="gray")
    ax1.legend()
    ax1.set_ylabel("资金")
    ax1.set_title(f"小市值+量价因子(10只) | 收益 {total_ret:.1%} vs 基准 {bench_ret:.1%} | 超额 {total_ret-bench_ret:+.1%}")
    ax1.grid(True, alpha=0.3)

    # 回撤
    ax2 = axes[1]
    ax2.fill_between(daily_df["date_dt"], dd * 100, 0, alpha=0.3, color="red")
    ax2.set_ylabel("回撤(%)")
    ax2.grid(True, alpha=0.3)

    # 持仓
    ax3 = axes[2]
    ax3.bar(daily_df["date_dt"], daily_df["positions"], alpha=0.5, color="steelblue")
    ax3.set_ylabel("持仓数")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/equity_v2.png", dpi=150)
    plt.close()

    # 交易统计
    if not sells.empty:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        colors = ["green" if x > 0 else "red" for x in sells["pnl"]]
        axes2[0].bar(range(len(sells)), sells["pnl"], color=colors, alpha=0.7)
        axes2[0].axhline(0, color="black", linewidth=0.5)
        axes2[0].set_title("每笔交易盈亏")
        axes2[0].set_ylabel("元")

        # 按卖出原因统计
        reason_stats = sells.groupby("reason").agg(
            count=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
        ).sort_values("total_pnl")
        colors2 = ["green" if x > 0 else "red" for x in reason_stats["total_pnl"]]
        axes2[1].barh(reason_stats.index, reason_stats["total_pnl"], color=colors2)
        axes2[1].set_title("按卖出原因统计盈亏")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/trades_v2.png", dpi=150)
        plt.close()

    # 保存数据
    trades_df.to_csv(f"{output_dir}/trades_v2.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(f"{output_dir}/daily_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\n输出已保存到 {output_dir}/")


if __name__ == "__main__":
    main()

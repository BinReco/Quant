"""
板块强度轮动策略 - 板块指数回测

使用板块指数直接作为交易标的，快速验证策略整体表现。
后续可替换为个股选择模式。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from dataclasses import dataclass

from config.settings import (
    INITIAL_CAPITAL, MAX_POSITIONS, COMMISSION_RATE,
    STAMP_TAX_RATE, MIN_COMMISSION, MAX_HOLD_DAYS, STOP_LOSS, TAKE_PROFIT,
    SECTOR_TOP_N,
)
from data.fetcher import get_all_sector_daily, get_market_daily
from strategy.sector_strength import compute_all_sectors_strength, rank_sectors
from analysis.report import plot_equity_curve, plot_trade_distribution, print_summary


@dataclass
class SectorPosition:
    code: str
    name: str
    buy_date: str
    buy_price: float
    amount: float  # 投入金额
    hold_days: int = 0


@dataclass
class SectorTrade:
    date: str
    code: str
    name: str
    sector: str
    direction: str
    price: float
    shares: int
    amount: float
    commission: float
    tax: float
    pnl: float = 0.0


def main():
    print("=" * 55)
    print("  板块强度轮动策略 - 板块指数回测")
    print("=" * 55)

    # 数据范围（预热 + 回测）
    data_start = "20250401"
    bt_start = "20250801"
    bt_end = "20260226"

    # 获取数据
    print("\n[1/3] 获取板块日线数据...")
    all_sector = get_all_sector_daily(data_start, bt_end)
    n_sectors = all_sector["板块代码"].nunique()
    print(f"  共 {n_sectors} 个板块")

    # 获取大盘
    print("[2/3] 获取大盘基准...")
    market = get_market_daily(data_start, bt_end)
    market_close = {}
    if not market.empty:
        market_close = dict(zip(
            market["date"].dt.strftime("%Y-%m-%d"),
            market["close"]
        ))

    # 计算强度
    print("[3/3] 计算板块强度分数...")
    strength = compute_all_sectors_strength(all_sector)

    # 构建板块价格查询表: {(板块代码, 日期str) -> 收盘价}
    sector_price = {}
    for _, row in strength.iterrows():
        key = (row["板块代码"], row["日期"].strftime("%Y-%m-%d"))
        sector_price[key] = row["收盘"]

    # 交易日列表
    start_dt = pd.to_datetime(bt_start)
    end_dt = pd.to_datetime(bt_end)
    trading_dates = sorted(
        strength[
            (strength["日期"] >= start_dt) & (strength["日期"] <= end_dt)
        ]["日期"].unique()
    )
    print(f"\n回测区间: {bt_start} ~ {bt_end}, 共 {len(trading_dates)} 个交易日")

    # ---- 回测 ----
    cash = float(INITIAL_CAPITAL)
    positions: dict[str, SectorPosition] = {}
    trades = []
    daily_records = []

    for i, date in enumerate(trading_dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        # 1. 更新持仓天数
        for pos in positions.values():
            if pos.buy_date != date_str:
                pos.hold_days += 1

        # 2. 检查卖出条件
        to_sell = []
        for code, pos in positions.items():
            if pos.buy_date == date_str:  # T+1
                continue
            price = sector_price.get((code, date_str))
            if price is None:
                continue
            pnl_pct = price / pos.buy_price - 1

            if pnl_pct <= STOP_LOSS or pnl_pct >= TAKE_PROFIT or pos.hold_days >= MAX_HOLD_DAYS:
                to_sell.append((code, price, pnl_pct))

        # 执行卖出
        for code, price, pnl_pct in to_sell:
            pos = positions[code]
            sell_value = pos.amount * (1 + pnl_pct)
            commission = max(sell_value * COMMISSION_RATE, MIN_COMMISSION)
            tax = sell_value * STAMP_TAX_RATE
            pnl = sell_value - pos.amount - commission - tax
            cash += sell_value - commission - tax

            trades.append(SectorTrade(
                date=date_str, code=code, name=pos.name,
                sector=pos.name, direction="SELL",
                price=price, shares=0, amount=sell_value,
                commission=commission, tax=tax, pnl=pnl,
            ))
            del positions[code]

        # 3. 选板块买入
        empty_slots = MAX_POSITIONS - len(positions)
        if empty_slots > 0:
            top = rank_sectors(strength, date_str, top_n=SECTOR_TOP_N)
            if not top.empty:
                for _, sector_row in top.iterrows():
                    if empty_slots <= 0:
                        break
                    scode = sector_row["板块代码"]
                    sname = sector_row["板块名称"]

                    if scode in positions:
                        continue

                    buy_price = sector_price.get((scode, date_str))
                    if buy_price is None:
                        continue

                    alloc = min(INITIAL_CAPITAL / MAX_POSITIONS, cash)
                    if alloc < 1000:
                        continue

                    commission = max(alloc * COMMISSION_RATE, MIN_COMMISSION)
                    invest = alloc - commission
                    cash -= alloc

                    positions[scode] = SectorPosition(
                        code=scode, name=sname,
                        buy_date=date_str, buy_price=buy_price,
                        amount=invest,
                    )
                    trades.append(SectorTrade(
                        date=date_str, code=scode, name=sname,
                        sector=sname, direction="BUY",
                        price=buy_price, shares=0, amount=invest,
                        commission=commission, tax=0,
                    ))
                    empty_slots -= 1

        # 4. 记录每日净值
        market_value = 0.0
        for code, pos in positions.items():
            price = sector_price.get((code, date_str))
            if price is not None:
                pnl_pct = price / pos.buy_price - 1
                market_value += pos.amount * (1 + pnl_pct)
            else:
                market_value += pos.amount

        total = cash + market_value
        benchmark = market_close.get(date_str, 0)
        daily_records.append({
            "date": date_str,
            "cash": cash,
            "market_value": market_value,
            "total_value": total,
            "positions": len(positions),
            "benchmark": benchmark,
        })

    # ---- 输出结果 ----
    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame([vars(t) for t in trades])

    # 计算指标
    total_return = daily_df["total_value"].iloc[-1] / INITIAL_CAPITAL - 1
    n_days = len(daily_df)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    peak = daily_df["total_value"].cummax()
    dd = (daily_df["total_value"] - peak) / peak
    max_dd = dd.min()

    daily_df["daily_ret"] = daily_df["total_value"].pct_change()
    rf = 0.02 / 252
    excess = daily_df["daily_ret"].dropna() - rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    sells = trades_df[trades_df["direction"] == "SELL"] if not trades_df.empty else pd.DataFrame()
    if not sells.empty:
        win_trades = sells[sells["pnl"] > 0]
        win_rate = len(win_trades) / len(sells)
        total_profit = win_trades["pnl"].sum()
        total_loss = sells[sells["pnl"] <= 0]["pnl"].sum()
        pf = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        avg_win = win_trades["pnl"].mean() if len(win_trades) > 0 else 0
        avg_loss = sells[sells["pnl"] <= 0]["pnl"].mean() if len(sells[sells["pnl"] <= 0]) > 0 else 0
    else:
        win_rate = total_profit = total_loss = pf = avg_win = avg_loss = 0

    results = {
        "初始资金": f"{INITIAL_CAPITAL:,.0f}",
        "最终资金": f"{daily_df['total_value'].iloc[-1]:,.0f}",
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{annual_return:.2%}",
        "最大回撤": f"{max_dd:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "总交易次数": len(trades),
        "卖出次数": len(sells),
        "胜率": f"{win_rate:.2%}",
        "盈亏比": f"{pf:.2f}",
        "平均盈利": f"{avg_win:,.0f}",
        "平均亏损": f"{avg_loss:,.0f}",
        "交易天数": n_days,
    }
    print_summary(results)

    # 保存图表和数据
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    plot_equity_curve(daily_df, os.path.join(output_dir, "equity.png"))
    plot_trade_distribution(trades_df, os.path.join(output_dir, "trades.png"))
    trades_df.to_csv(os.path.join(output_dir, "trades.csv"),
                     index=False, encoding="utf-8-sig")
    daily_df.to_csv(os.path.join(output_dir, "daily.csv"),
                    index=False, encoding="utf-8-sig")

    # 打印最近交易
    if not sells.empty:
        print("\n最近10笔卖出交易:")
        for _, t in sells.tail(10).iterrows():
            pnl_str = f"+{t['pnl']:,.0f}" if t["pnl"] > 0 else f"{t['pnl']:,.0f}"
            print(f"  {t['date']} | {t['name']:>6s} | {pnl_str}")

    print(f"\n输出已保存到 {output_dir}/")


if __name__ == "__main__":
    main()

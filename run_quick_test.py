"""
快速验证：板块强度模型是否能捕捉到轮动机会

不做个股选择，直接看强势板块后续的表现
验证逻辑：如果今天选出的 top3 板块，后续 3-5 天平均跑赢大盘，说明模型有效
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from data.fetcher import get_all_sector_daily, get_market_daily
from strategy.sector_strength import compute_all_sectors_strength, rank_sectors
from config.settings import SECTOR_TOP_N


def main():
    print("=" * 50)
    print("  板块强度模型 - 快速验证")
    print("=" * 50)

    # 获取数据
    print("\n加载板块数据...")
    all_sector = get_all_sector_daily("20250601", "20260228")
    print(f"板块数: {all_sector['板块代码'].nunique()}")

    # 计算强度
    print("计算板块强度...")
    strength = compute_all_sectors_strength(all_sector)

    # 获取大盘数据
    market = get_market_daily("20250601", "20260228")
    if not market.empty:
        market["daily_return"] = market["close"].pct_change()
        market_ret_map = dict(zip(
            market["date"].dt.strftime("%Y-%m-%d"),
            market["daily_return"]
        ))

    # 获取交易日列表（从7月开始，给强度计算留预热期）
    dates = sorted(strength[
        strength["日期"] >= pd.to_datetime("20250801")
    ]["日期"].unique())

    # 预计算每个板块每天的收益率
    sector_returns = {}
    for code, group in strength.groupby("板块代码"):
        g = group.sort_values("日期").copy()
        g["daily_return"] = g["收盘"].pct_change()
        # 未来 N 天收益
        g["fwd_1d"] = g["收盘"].shift(-1) / g["收盘"] - 1
        g["fwd_3d"] = g["收盘"].shift(-3) / g["收盘"] - 1
        g["fwd_5d"] = g["收盘"].shift(-5) / g["收盘"] - 1
        sector_returns[code] = g.set_index("日期")

    # 逐日选板块，看后续表现
    results = []
    for date in dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        top = rank_sectors(strength, date_str, top_n=SECTOR_TOP_N)
        if top.empty:
            continue

        for _, row in top.iterrows():
            code = row["板块代码"]
            name = row["板块名称"]
            score = row["strength"]

            if code in sector_returns:
                sr = sector_returns[code]
                if date in sr.index:
                    r = sr.loc[date]
                    results.append({
                        "date": date_str,
                        "sector": name,
                        "strength": score,
                        "fwd_1d": r.get("fwd_1d", np.nan),
                        "fwd_3d": r.get("fwd_3d", np.nan),
                        "fwd_5d": r.get("fwd_5d", np.nan),
                    })

    if not results:
        print("没有选出任何板块，检查过滤条件")
        return

    df = pd.DataFrame(results)
    print(f"\n共选出 {len(df)} 次板块信号\n")

    # 统计选出板块的后续收益
    print("=" * 50)
    print("选出的强势板块 后续平均收益率:")
    print("-" * 50)
    print(f"  后1日平均收益: {df['fwd_1d'].mean():.4%}")
    print(f"  后3日平均收益: {df['fwd_3d'].mean():.4%}")
    print(f"  后5日平均收益: {df['fwd_5d'].mean():.4%}")
    print()
    print(f"  后1日胜率: {(df['fwd_1d'] > 0).mean():.2%}")
    print(f"  后3日胜率: {(df['fwd_3d'] > 0).mean():.2%}")
    print(f"  后5日胜率: {(df['fwd_5d'] > 0).mean():.2%}")
    print()

    # 按月统计
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        signals=("fwd_1d", "count"),
        avg_1d=("fwd_1d", "mean"),
        avg_3d=("fwd_3d", "mean"),
        avg_5d=("fwd_5d", "mean"),
        win_rate_3d=("fwd_3d", lambda x: (x > 0).mean()),
    )
    print("按月统计:")
    print(monthly.to_string())

    # 最近选出的板块
    print(f"\n最近5日选出的板块:")
    recent = df.sort_values("date").tail(15)
    for _, r in recent.iterrows():
        print(f"  {r['date']} | {r['sector']:>6s} | "
              f"强度:{r['strength']:.4f} | "
              f"后1日:{r['fwd_1d']:+.2%} | "
              f"后3日:{r['fwd_3d']:+.2%}")


if __name__ == "__main__":
    main()

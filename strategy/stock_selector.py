"""
个股选择模块

从强势板块中挑选最佳交易标的。
选股条件：
1. 非 ST / 非停牌
2. 非一字涨停（买不进去）
3. 成交量放大（有资金参与）
4. 涨幅适中（不追太高，留有空间）
5. 流通市值适中（太小流动性差，太大弹性小）
"""

from typing import Optional

import pandas as pd
import numpy as np
import time

from data.fetcher import get_sector_cons, get_stock_daily


def select_stocks_from_sector(sector_code: str, date: str,
                              start: str, end: str,
                              max_picks: int = 2) -> list:
    """
    从指定板块中选出候选股票

    参数:
        sector_code: 板块代码
        date: 选股日期 "YYYYMMDD"
        start/end: 个股数据起止日期
        max_picks: 最多选几只

    返回:
        [(股票代码, 股票名称, 得分), ...]
    """
    # 获取板块成分股
    cons = get_sector_cons(sector_code)
    if cons.empty:
        return []

    candidates = []
    target_date = pd.to_datetime(date)

    for _, row in cons.iterrows():
        code = row["证券代码"]
        name = row["证券名称"]

        # 跳过 ST 股
        if "ST" in name or "st" in name:
            continue

        stock_df = get_stock_daily(code, start, end)
        if stock_df.empty:
            continue

        stock_df["日期"] = pd.to_datetime(stock_df["日期"])
        day_row = stock_df[stock_df["日期"] == target_date]

        if day_row.empty:
            continue

        day = day_row.iloc[0]
        score = _score_stock(stock_df, day, target_date)

        if score is not None:
            candidates.append((code, name, score))

        time.sleep(0.05)

    # 按得分降序排列
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:max_picks]


def _score_stock(stock_df: pd.DataFrame, day: pd.Series,
                 target_date) -> Optional[float]:
    """
    给个股打分

    返回 None 表示不符合条件，应该跳过
    """
    # ---- 过滤条件 ----

    open_price = day["开盘"]
    close_price = day["收盘"]
    high = day["最高"]
    low = day["最低"]
    pct_change = day["涨跌幅"]
    turnover = day["换手率"]

    # 一字涨停：开盘 = 收盘 = 最高 = 最低，且涨幅接近涨停
    if high == low and pct_change > 9.5:
        return None

    # 涨幅已经太高（追高风险）
    if pct_change > 7:
        return None

    # 跌太多的不选（可能有利空）
    if pct_change < -5:
        return None

    # 换手率太低（流动性差）
    if turnover < 1.0:
        return None

    # ---- 评分因子 ----

    # 近期数据
    recent = stock_df[stock_df["日期"] <= target_date].tail(10)
    if len(recent) < 5:
        return None

    # 1. 当日涨幅适中（2-5% 最佳）
    if 2 <= pct_change <= 5:
        price_score = 1.0
    elif 0 < pct_change < 2:
        price_score = 0.5
    elif 5 < pct_change <= 7:
        price_score = 0.3
    else:
        price_score = 0.1

    # 2. 量能放大
    vol_today = day["成交量"]
    vol_avg = recent["成交量"].mean()
    if vol_avg > 0:
        vol_ratio = vol_today / vol_avg
        vol_score = min(vol_ratio / 2.0, 1.0)  # 放量2倍得满分
    else:
        vol_score = 0.0

    # 3. 换手率适中（3-8% 最佳）
    if 3 <= turnover <= 8:
        turnover_score = 1.0
    elif 1 <= turnover < 3:
        turnover_score = 0.5
    elif 8 < turnover <= 15:
        turnover_score = 0.6
    else:
        turnover_score = 0.2

    # 4. 近5日累计涨幅（不能已经涨太多）
    if len(recent) >= 5:
        recent_5d_return = (recent["收盘"].iloc[-1] /
                            recent["收盘"].iloc[-5] - 1)
        if recent_5d_return > 0.15:  # 5天涨了15%以上，太高
            return None
        position_score = max(0, 1.0 - recent_5d_return * 5)
    else:
        position_score = 0.5

    # 综合得分
    total = (price_score * 0.3 + vol_score * 0.3
             + turnover_score * 0.2 + position_score * 0.2)
    return total


def select_stocks_for_day(top_sectors: pd.DataFrame, date: str,
                          start: str, end: str,
                          max_total: int = 5) -> list:
    """
    从当日排名靠前的板块中选出候选股票

    参数:
        top_sectors: rank_sectors() 的返回值
        date: 选股日期 "YYYYMMDD"
        max_total: 总共最多选几只

    返回:
        [(股票代码, 股票名称, 板块名称, 得分), ...]
    """
    all_picks = []
    picks_per_sector = max(1, max_total // len(top_sectors)
                           if len(top_sectors) > 0 else max_total)

    for _, sector in top_sectors.iterrows():
        code = sector["板块代码"]
        name = sector["板块名称"]

        picks = select_stocks_from_sector(
            code, date, start, end, max_picks=picks_per_sector
        )
        for stock_code, stock_name, score in picks:
            all_picks.append((stock_code, stock_name, name, score))

    # 按得分排序，取 top
    all_picks.sort(key=lambda x: x[3], reverse=True)
    return all_picks[:max_total]

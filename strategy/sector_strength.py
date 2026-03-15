"""
板块强度评分模型

对每个板块计算综合强度分数，用于识别轮动初期的板块。
基于申万二级行业指数日线数据。

强度分数由以下因子加权得出：
1. 动量因子 (momentum)    - 近 N 日涨幅
2. 加速因子 (acceleration) - 近期涨幅 vs 更早期涨幅
3. 量能因子 (volume_surge) - 成交量放大倍数
4. 趋势因子 (trend)       - 收盘价相对均线位置
"""

import pandas as pd
import numpy as np

from config.settings import STRENGTH_LOOKBACK


def calc_sector_strength(sector_df: pd.DataFrame,
                         lookback: int = STRENGTH_LOOKBACK) -> pd.DataFrame:
    """
    计算单个板块的每日强度分数

    参数:
        sector_df: 板块日线数据，需包含 日期/收盘/成交量 列
        lookback: 回看天数

    返回:
        添加了强度因子列的 DataFrame
    """
    df = sector_df.copy()
    df = df.sort_values("日期").reset_index(drop=True)

    # 1. 动量因子: N日收益率
    df["momentum"] = df["收盘"].pct_change(lookback)

    # 2. 加速因子: 近期动量 vs 更早期动量
    df["mom_short"] = df["收盘"].pct_change(lookback // 2 + 1)
    df["mom_long"] = df["收盘"].pct_change(lookback)
    df["acceleration"] = df["mom_short"] - (df["mom_long"] - df["mom_short"])

    # 3. 量能因子: 近N日平均成交量 / 前一段平均成交量
    df["vol_ma_short"] = df["成交量"].rolling(lookback).mean()
    df["vol_ma_long"] = df["成交量"].rolling(lookback * 4).mean()
    df["volume_surge"] = df["vol_ma_short"] / df["vol_ma_long"]

    # 4. 趋势因子: 收盘价 / 20日均线 - 1
    df["ma20"] = df["收盘"].rolling(20).mean()
    df["trend"] = df["收盘"] / df["ma20"] - 1

    # 5. 波动率因子: 近期波动率（用于过滤暴涨暴跌板块）
    df["volatility"] = df["收盘"].pct_change().rolling(lookback * 2).std()

    # 综合强度分数 (各因子统一做滚动 z-score 标准化后加权)
    window = lookback * 4
    df["strength"] = (
        _zscore(df["momentum"], window) * 0.3
        + _zscore(df["acceleration"], window) * 0.25
        + _zscore(df["volume_surge"], window) * 0.25
        + _zscore(df["trend"], window) * 0.2
    )

    # 强度变化率：今天 vs 昨天，用于捕捉"突然变强"
    df["strength_delta"] = df["strength"] - df["strength"].shift(1)

    return df


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动 z-score 标准化，使各因子量纲一致"""
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std()
    std = std.replace(0, np.nan)
    return (series - mean) / std.fillna(1.0)


def rank_sectors(all_sector_df: pd.DataFrame, date: str,
                 top_n: int = 3) -> pd.DataFrame:
    """
    对指定日期的所有板块按强度排名

    参数:
        all_sector_df: 所有板块的强度数据
        date: 目标日期, "YYYY-MM-DD"
        top_n: 返回前 N 个板块

    返回:
        排名前 N 的板块 DataFrame
    """
    target = pd.to_datetime(date)
    day_data = all_sector_df[all_sector_df["日期"] == target].copy()

    if day_data.empty:
        return pd.DataFrame()

    # 按强度分数降序排名
    day_data = day_data.sort_values("strength", ascending=False)

    # 过滤条件:
    # 1. 强度分数 > 0
    # 2. 量能放大
    # 3. 波动率不能太高（过滤暴涨暴跌的妖板块）
    # 4. 动量为正（趋势向上）
    vol_median = day_data["volatility"].median()
    day_data = day_data[
        (day_data["strength"] > 0)
        & (day_data["volume_surge"] > 1.0)
        & (day_data["volatility"] < vol_median * 2.5)
        & (day_data["momentum"] > 0)
    ]

    return day_data.head(top_n)


def compute_all_sectors_strength(all_sector_daily: pd.DataFrame,
                                 lookback: int = STRENGTH_LOOKBACK
                                 ) -> pd.DataFrame:
    """
    批量计算所有板块的强度分数

    参数:
        all_sector_daily: get_all_sector_daily() 的返回值
        lookback: 回看天数

    返回:
        所有板块所有日期的强度数据
    """
    results = []
    for code, group in all_sector_daily.groupby("板块代码"):
        name = group["板块名称"].iloc[0]
        scored = calc_sector_strength(group, lookback)
        scored["板块代码"] = code
        scored["板块名称"] = name
        results.append(scored)

    return pd.concat(results, ignore_index=True)

"""
小市值 + 量价因子选股策略

核心逻辑：
- 在中小盘股票池中，用量价因子筛选出启动信号明确的标的
- 配合大盘择时，弱市减仓或空仓
- 严格止损 + 移动止盈

选股因子：
1. 市值因子：中小盘（流通市值排名后 30%-80%，避开微盘垃圾股）
2. 放量突破：成交量相对放大 + 价格突破近期高点
3. 换手率：适中（2%-10%），太低没人买，太高可能是出货
4. 短期动量：近5-10日涨幅为正但不超过15%（启动但未过热）
5. 价格位置：股价在20日均线上方（趋势向上）

大盘择时：
- 上证指数在20日均线上方 → 满仓操作（5只）
- 上证指数在20日均线下方但在60日均线上方 → 半仓（2-3只）
- 上证指数在60日均线下方 → 空仓

止损止盈：
- 固定止损 -5%
- 移动止盈：从最高盈利回撤 50% 则卖出
- 最大持仓 10 个交易日
"""

import pandas as pd
import numpy as np


def market_regime(market_df: pd.DataFrame, date) -> str:
    """
    判断大盘状态

    返回: "bull"（满仓）/ "neutral"（半仓）/ "bear"（空仓）
    """
    df = market_df[market_df["日期"] <= date].copy()
    if len(df) < 60:
        return "neutral"

    close = df["收盘"].iloc[-1]
    ma20 = df["收盘"].rolling(20).mean().iloc[-1]
    ma60 = df["收盘"].rolling(60).mean().iloc[-1]

    if close > ma20:
        return "bull"
    elif close > ma60:
        return "neutral"
    else:
        return "bear"


def select_stocks(all_stocks_day: pd.DataFrame,
                  all_stocks_hist: dict[str, pd.DataFrame],
                  date, max_picks: int = 5) -> list[dict]:
    """
    从全市场当日截面中选股

    参数:
        all_stocks_day: 当日所有股票截面数据
        all_stocks_hist: {股票代码: 历史日线DataFrame} 的字典
        date: 当前日期
        max_picks: 最多选几只

    返回:
        [{"code": ..., "name": ..., "score": ..., "close": ...}, ...]
    """
    candidates = []

    for _, row in all_stocks_day.iterrows():
        code = row["股票代码"]
        name = row.get("名称", code)
        close = row["收盘"]
        pct = row["涨跌幅"]
        turnover = row["换手率"]
        volume = row["成交量"]
        market_cap = row.get("流通市值", 0)

        # ---- 基础过滤 ----

        # 排除 ST
        if isinstance(name, str) and ("ST" in name or "退" in name):
            continue

        # 排除停牌（成交量为0）
        if volume <= 0:
            continue

        # 排除一字涨停（买不进）
        if row["开盘"] == row["最高"] == row["最低"] and pct > 9.5:
            continue

        # 排除当日跌停
        if pct < -9.5:
            continue

        # 排除涨幅已经太大的
        if pct > 8:
            continue

        # ---- 市值过滤 ----
        # 流通市值适中（不要太大也不要太小）
        # 这个在截面排名时处理

        # ---- 换手率过滤 ----
        if turnover < 2 or turnover > 12:
            continue

        # ---- 历史数据计算 ----
        hist = all_stocks_hist.get(code)
        if hist is None or len(hist) < 25:
            continue

        recent = hist[hist["日期"] <= date].tail(25)
        if len(recent) < 20:
            continue

        close_arr = recent["收盘"].values
        vol_arr = recent["成交量"].values

        # 5日动量
        if len(close_arr) >= 6:
            mom_5d = close_arr[-1] / close_arr[-6] - 1
        else:
            continue

        # 10日动量
        if len(close_arr) >= 11:
            mom_10d = close_arr[-1] / close_arr[-11] - 1
        else:
            mom_10d = mom_5d

        # 过滤：短期涨幅不能太大（已经过热）
        if mom_5d > 0.15 or mom_10d > 0.25:
            continue

        # 过滤：必须有正动量（趋势向上）
        if mom_5d < 0:
            continue

        # 20日均线上方
        ma20 = close_arr[-20:].mean()
        if close_arr[-1] < ma20:
            continue

        # ---- 量价因子打分 ----

        # 1. 放量程度：今日成交量 / 5日平均成交量
        vol_ma5 = vol_arr[-5:].mean() if vol_arr[-5:].mean() > 0 else 1
        vol_ratio = volume / vol_ma5
        vol_score = min(vol_ratio / 2.0, 1.0)  # 放量2倍得满分

        # 2. 价格突破：收盘价接近或突破近20日高点
        high_20 = close_arr[-20:].max()
        breakout_score = min(close_arr[-1] / high_20, 1.0)

        # 3. 动量得分：5日涨幅在合理区间得高分
        if 0.03 <= mom_5d <= 0.08:
            mom_score = 1.0
        elif 0 < mom_5d < 0.03:
            mom_score = 0.5
        elif 0.08 < mom_5d <= 0.15:
            mom_score = 0.6
        else:
            mom_score = 0.2

        # 4. 换手率适中度
        if 3 <= turnover <= 8:
            turn_score = 1.0
        elif 2 <= turnover < 3:
            turn_score = 0.6
        elif 8 < turnover <= 12:
            turn_score = 0.5
        else:
            turn_score = 0.3

        # 综合得分
        total_score = (vol_score * 0.3 + breakout_score * 0.3
                       + mom_score * 0.25 + turn_score * 0.15)

        candidates.append({
            "code": code,
            "name": name,
            "close": close,
            "score": total_score,
            "market_cap": market_cap,
            "mom_5d": mom_5d,
            "vol_ratio": vol_ratio,
            "turnover": turnover,
        })

    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    # 市值排名过滤：取后30%-80%（中小盘）
    if len(df) > 20 and df["market_cap"].sum() > 0:
        df["cap_rank"] = df["market_cap"].rank(pct=True)
        df = df[(df["cap_rank"] >= 0.2) & (df["cap_rank"] <= 0.7)]

    if df.empty:
        return []

    # 按综合得分排名
    df = df.sort_values("score", ascending=False)

    return df.head(max_picks).to_dict("records")

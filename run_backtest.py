"""
板块强度轮动策略 - 回测主程序（baostock 数据源）

优化：
- 用 baostock 批量下载全市场数据（比 akshare 快 10 倍+）
- 板块用申万行业指数（仍从 akshare 获取，有缓存）
- 个股选股全部从内存查，回测循环零 API 调用
"""

import sys
import os
import pickle
import time

import baostock as bs
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    BACKTEST_START, BACKTEST_END, SECTOR_TOP_N, MAX_POSITIONS,
    MARKET_TIMING, MARKET_MA_SHORT, MARKET_MA_LONG,
)
from data.fetcher import (
    get_all_sector_daily, get_market_daily,
    get_sector_list, get_sector_cons,
)
from strategy.sector_strength import (
    compute_all_sectors_strength, rank_sectors,
)
from backtest.engine import BacktestEngine
from analysis.report import (
    plot_equity_curve, plot_trade_distribution, plot_monthly_returns, print_summary,
)


# ============ 数据下载（baostock） ============

def _bs_download_all_stocks(data_start, data_end):
    """用 baostock 批量下载全市场股票日线，逐只缓存 + 断点续传"""
    cache_file = f"cache/bs_all_stocks_{data_start}_{data_end}.pkl"
    if os.path.exists(cache_file):
        print("  从缓存加载全市场数据...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    stock_cache_dir = "cache/bs_stocks"
    os.makedirs(stock_cache_dir, exist_ok=True)

    lg = bs.login()

    # 获取全 A 股列表
    rs = bs.query_stock_basic(code_name="", code="")
    stock_list = rs.get_data()
    stock_list = stock_list[stock_list["type"] == "1"]
    codes = stock_list["code"].tolist()
    print(f"  全 A 股数量: {len(codes)}")

    # 检查已缓存
    cached = set()
    for f in os.listdir(stock_cache_dir):
        if f.endswith(".pkl"):
            cached.add(f.replace(".pkl", "").replace("_", "."))
    print(f"  已缓存: {len(cached)}, 待下载: {len(codes) - len(cached)}")

    import signal

    class TimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError("下载超时")

    fail_count = 0
    for code in tqdm(codes, desc="  下载股票数据"):
        if code in cached:
            continue
        try:
            # 设置 30 秒超时，防止卡死
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)

            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg,isST",
                start_date=data_start, end_date=data_end,
                frequency="d", adjustflag="2",
            )
            df = rs.get_data()

            signal.alarm(0)  # 取消超时

            if df.empty or len(df) < 20:
                fail_count = 0
                continue

            for col in ["open", "high", "low", "close", "volume",
                        "amount", "turn", "pctChg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"])

            if len(df) < 20:
                continue

            cache_name = code.replace(".", "_")
            with open(f"{stock_cache_dir}/{cache_name}.pkl", "wb") as f:
                pickle.dump(df, f)
            cached.add(code)
            fail_count = 0

        except (Exception, TimeoutError):
            signal.alarm(0)
            fail_count += 1
            if fail_count >= 10:
                print(f"\n  连续失败 {fail_count} 次，重连 baostock...")
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(3)
                bs.login()
                fail_count = 0

    signal.alarm(0)
    bs.logout()

    # 合并
    print("  合并缓存...")
    all_data = {}
    for f in os.listdir(stock_cache_dir):
        if not f.endswith(".pkl"):
            continue
        code = f.replace(".pkl", "").replace("_", ".")
        with open(f"{stock_cache_dir}/{f}", "rb") as fh:
            all_data[code] = pickle.load(fh)

    os.makedirs("cache", exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(all_data, f)

    print(f"  有效股票: {len(all_data)}")
    return all_data


def _bs_download_index(data_start, data_end):
    """下载上证指数"""
    cache_path = f"cache/bs_index_{data_start}_{data_end}.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        "sh.000001", "date,close",
        start_date=data_start, end_date=data_end,
        frequency="d",
    )
    df = rs.get_data()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    bs.logout()

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


# ============ 板块→个股映射 ============

def _build_sector_stock_map():
    """构建 板块代码 → [baostock 股票代码] 的映射"""
    sectors = get_sector_list()
    sector_map = {}

    for _, row in sectors.iterrows():
        sw_code = row["行业代码"].replace(".SI", "")
        cons = get_sector_cons(sw_code)
        if cons.empty:
            continue

        bs_codes = []
        for _, c in cons.iterrows():
            stock_code = c["证券代码"]
            name = c["证券名称"]
            if "ST" in str(name) or "st" in str(name):
                continue
            # 转换为 baostock 格式: 000001 -> sz.000001
            if stock_code.startswith("6"):
                bs_codes.append(f"sh.{stock_code}")
            else:
                bs_codes.append(f"sz.{stock_code}")
        sector_map[sw_code] = bs_codes

    return sector_map


# ============ 选股（从内存数据） ============

def _select_stocks(all_data, sector_codes, date, max_picks=5):
    """
    选股逻辑（优化版）：选"趋势确立但尚未过热"的个股
    - 要求站上20日均线（趋势确认）
    - 5日涨幅适中（0~10%，非追高）
    - 量能温和放大（非爆量见顶）
    - 10日涨幅不能太大（排除已暴涨的）
    """
    candidates = []
    target_date = pd.to_datetime(date)

    for code in sector_codes:
        df = all_data.get(code)
        if df is None or df.empty:
            continue

        hist = df[df["date"] <= target_date].tail(30)
        if len(hist) < 20:
            continue

        today = hist.iloc[-1]
        if today["date"] != target_date:
            continue

        close = today["close"]
        pct = today["pctChg"]
        turn = today["turn"]
        vol = today["volume"]
        is_st = today.get("isST", "0")

        # 基础过滤
        if is_st == "1":
            continue
        if vol <= 0 or close <= 0:
            continue
        if pct > 9.5 and today["open"] == today["high"] == today["low"]:
            continue  # 一字涨停买不进
        if pct < -3 or pct > 7:
            continue  # 当日涨幅异常的不要
        if turn < 1 or turn > 20:
            continue

        closes = hist["close"].values
        vols = hist["volume"].values

        # 趋势确认：站上20日均线
        ma20 = closes[-20:].mean()
        if close < ma20:
            continue

        # 10日均线也要向上
        ma10 = closes[-10:].mean()
        if ma10 < closes[-20:-10].mean():
            continue  # 10日均线走平或下行，趋势未确立

        # 5日动量：适中（0~10%）
        mom_5d = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else 0
        if mom_5d < 0 or mom_5d > 0.10:
            continue

        # 10日涨幅不能太大（排除已暴涨）
        mom_10d = closes[-1] / closes[-11] - 1 if len(closes) >= 11 else 0
        if mom_10d > 0.20:
            continue

        # 量能温和放大（1~3倍，非爆量）
        vol_ma10 = vols[-10:].mean() if vols[-10:].mean() > 0 else 1
        vol_ratio = vol / vol_ma10
        if vol_ratio > 4.0:
            continue  # 爆量通常是见顶信号

        # === 评分 ===
        # 趋势强度：离20日均线的距离（越近越好，说明刚启动）
        ma_dist = close / ma20 - 1
        if ma_dist < 0.02:
            trend_score = 1.0  # 刚站上均线
        elif ma_dist < 0.05:
            trend_score = 0.8
        elif ma_dist < 0.10:
            trend_score = 0.5
        else:
            trend_score = 0.2  # 离均线太远，追高风险大

        # 量能评分
        if 1.2 <= vol_ratio <= 2.5:
            vol_score = 1.0  # 温和放量最佳
        elif 1.0 <= vol_ratio < 1.2:
            vol_score = 0.5
        else:
            vol_score = 0.3

        # 涨幅评分
        if 1 <= pct <= 4:
            price_score = 1.0
        elif 0 < pct < 1:
            price_score = 0.6
        elif 4 < pct <= 7:
            price_score = 0.4
        else:
            price_score = 0.2

        # 位置评分：5日涨幅越小越好（启动初期）
        position_score = max(0, 1.0 - mom_5d * 10)

        score = (trend_score * 0.35 + vol_score * 0.25
                 + price_score * 0.2 + position_score * 0.2)

        candidates.append({
            "code": code,
            "close": close,
            "score": score,
        })

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_picks]


# ============ 大盘择时 ============

def _market_regime(index_df, date):
    """
    增强版大盘择时：
    - bull: 价格在短期均线上方 且 短期均线在长期均线上方
    - neutral: 价格在长期均线上方 但短期趋势不明
    - bear: 价格在长期均线下方 或 短期均线在长期均线下方持续下行
    """
    df = index_df[index_df["date"] <= pd.to_datetime(date)]
    if len(df) < MARKET_MA_LONG + 10:
        return "neutral"

    close = df["close"].iloc[-1]
    closes = df["close"].values
    ma_s = pd.Series(closes).rolling(MARKET_MA_SHORT).mean().iloc[-1]
    ma_l = pd.Series(closes).rolling(MARKET_MA_LONG).mean().iloc[-1]

    # 长期均线斜率（最近10天）
    ma_l_series = pd.Series(closes).rolling(MARKET_MA_LONG).mean()
    ma_l_slope = (ma_l_series.iloc[-1] / ma_l_series.iloc[-10] - 1) if len(ma_l_series) > 10 else 0

    # 20日收益率
    ret_20d = closes[-1] / closes[-20] - 1 if len(closes) >= 20 else 0

    if close > ma_s and ma_s > ma_l and ma_l_slope > -0.005:
        return "bull"
    elif close > ma_l and ret_20d > -0.05:
        return "neutral"
    else:
        return "bear"


# ============ 主程序 ============

def main():
    print("=" * 55)
    print("  板块强度轮动策略 - 5年回测")
    print("=" * 55)

    # 数据区间（预热60天）
    bt_start_dt = pd.to_datetime(BACKTEST_START)
    data_start = (bt_start_dt - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    data_end = pd.to_datetime(BACKTEST_END).strftime("%Y-%m-%d")
    data_start_compact = data_start.replace("-", "")
    data_end_compact = data_end.replace("-", "")

    # ---- 第1步：下载全市场股票数据（baostock） ----
    print(f"\n[1/6] 下载全市场股票数据 ({data_start} ~ {data_end})...")
    all_stock_data = _bs_download_all_stocks(data_start, data_end)
    print(f"  有效股票: {len(all_stock_data)}")

    # ---- 第2步：下载上证指数 ----
    print("\n[2/6] 下载上证指数...")
    index_df = _bs_download_index(data_start, data_end)

    # ---- 第3步：获取板块日线数据（akshare，有缓存） ----
    print("\n[3/6] 获取板块日线数据...")
    all_sector_daily = get_all_sector_daily(data_start_compact, data_end_compact)
    n_sectors = all_sector_daily["板块代码"].nunique()
    print(f"  共 {n_sectors} 个板块")

    # ---- 第4步：计算板块强度 ----
    print("\n[4/6] 计算板块强度分数...")
    strength_df = compute_all_sectors_strength(all_sector_daily)

    # ---- 第5步：构建板块→个股映射 ----
    print("\n[5/6] 构建板块成分股映射...")
    sector_stock_map = _build_sector_stock_map()
    print(f"  {len(sector_stock_map)} 个板块映射完成")

    # ---- 第6步：逐日回测 ----
    print("\n[6/6] 开始逐日回测...")
    engine = BacktestEngine()

    trading_dates = sorted(
        strength_df[
            (strength_df["日期"] >= bt_start_dt) &
            (strength_df["日期"] <= pd.to_datetime(BACKTEST_END))
        ]["日期"].unique()
    )
    trading_dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d")
                         for d in trading_dates]
    print(f"  交易日数: {len(trading_dates)}")

    # 构建 baostock 价格查询表
    price_lookup = {}
    for code, df in all_stock_data.items():
        for _, row in df.iterrows():
            price_lookup[(code, row["date"].strftime("%Y-%m-%d"))] = row["close"]

    # 大盘基准
    market_map = {}
    if not index_df.empty:
        market_map = dict(zip(
            index_df["date"].dt.strftime("%Y-%m-%d"),
            index_df["close"]
        ))

    for date in tqdm(trading_dates, desc="回测进度"):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")

        # 当日持仓价格
        price_map = {}
        for code in list(engine.positions.keys()):
            p = price_lookup.get((code, date_str))
            if p is not None:
                price_map[code] = p

        # 1. 检查卖出
        to_sell = engine.check_exits(date_str, price_map,
                                     trading_dates=trading_dates_str)
        for code in to_sell:
            if code in price_map:
                engine.sell(code, price_map[code], date_str)

        # 2. 大盘择时
        if MARKET_TIMING:
            regime = _market_regime(index_df, date)
            if regime == "bull":
                target_pos = MAX_POSITIONS
            elif regime == "neutral":
                target_pos = max(1, MAX_POSITIONS // 2)
            else:
                target_pos = 0
        else:
            target_pos = MAX_POSITIONS

        # 3. 选股买入（降低频率：每次最多买入2只）
        empty_slots = target_pos - len(engine.positions)
        max_buy_per_day = 2  # 每天最多新建2个仓位，避免集中买入
        buy_slots = min(empty_slots, max_buy_per_day)

        if buy_slots > 0:
            top_sectors = rank_sectors(strength_df, date_str,
                                       top_n=SECTOR_TOP_N)
            if not top_sectors.empty:
                all_picks = []
                picks_per_sector = max(1, buy_slots)

                for _, sr in top_sectors.iterrows():
                    scode = sr["板块代码"]
                    sname = sr["板块名称"]
                    sector_codes = sector_stock_map.get(scode, [])
                    if not sector_codes:
                        continue

                    picks = _select_stocks(
                        all_stock_data, sector_codes, date_str,
                        max_picks=picks_per_sector,
                    )
                    for p in picks:
                        all_picks.append({**p, "sector": sname})

                all_picks.sort(key=lambda x: x["score"], reverse=True)

                bought = 0
                for pick in all_picks:
                    if bought >= buy_slots:
                        break
                    code = pick["code"]
                    if code in engine.positions:
                        continue
                    buy_price = price_lookup.get((code, date_str))
                    if buy_price is not None:
                        if engine.buy(code, code, pick["sector"],
                                      buy_price, date_str):
                            bought += 1

        # 补充新买入股票的价格
        for code in list(engine.positions.keys()):
            if code not in price_map:
                p = price_lookup.get((code, date_str))
                if p is not None:
                    price_map[code] = p

        benchmark = market_map.get(date_str, 0)
        engine.update_daily(date_str, price_map, benchmark)

    # ---- 输出结果 ----
    results = engine.get_results()
    print_summary(results)

    daily_df = engine.get_daily_df()
    trades_df = engine.get_trades_df()

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    plot_equity_curve(daily_df, os.path.join(output_dir, "equity.png"))
    plot_trade_distribution(trades_df, os.path.join(output_dir, "trades.png"))
    plot_monthly_returns(daily_df, os.path.join(output_dir, "monthly.png"))

    trades_df.to_csv(os.path.join(output_dir, "trades.csv"),
                     index=False, encoding="utf-8-sig")
    daily_df.to_csv(os.path.join(output_dir, "daily.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"\n详细数据已保存到 {output_dir}/")


if __name__ == "__main__":
    main()

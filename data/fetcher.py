"""
数据获取模块 - 使用 akshare 获取 A 股板块和个股数据
使用申万二级行业分类 + 个股日线数据
"""

import os
import pickle
import time
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from tqdm import tqdm

from config.settings import CACHE_DIR


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def _load_cache(name: str, max_age_hours: int = 12):
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(hours=max_age_hours):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(name: str, data):
    path = _cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ============ 板块数据 ============

def get_sector_list() -> pd.DataFrame:
    """获取申万二级行业板块列表"""
    cached = _load_cache("sector_list", max_age_hours=168)
    if cached is not None:
        return cached
    df = ak.sw_index_second_info()
    _save_cache("sector_list", df)
    return df


def get_sector_daily(sector_code: str) -> pd.DataFrame:
    """
    获取申万板块指数日线 (全历史)
    sector_code: 如 "801016"
    """
    cache_name = f"sw_hist_{sector_code}"
    cached = _load_cache(cache_name, max_age_hours=12)
    if cached is not None:
        return cached
    try:
        df = ak.index_hist_sw(symbol=sector_code, period="day")
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)
        _save_cache(cache_name, df)
        return df
    except Exception as e:
        print(f"获取板块 {sector_code} 数据失败: {e}")
        return pd.DataFrame()


def get_sector_cons(sector_code: str) -> pd.DataFrame:
    """获取申万二级行业成分股"""
    cache_name = f"sw_cons_{sector_code}"
    cached = _load_cache(cache_name, max_age_hours=168)
    if cached is not None:
        return cached
    try:
        df = ak.index_component_sw(symbol=sector_code)
        _save_cache(cache_name, df)
        return df
    except Exception as e:
        print(f"获取板块 {sector_code} 成分股失败: {e}")
        return pd.DataFrame()


def get_all_sector_daily(start: str, end: str) -> pd.DataFrame:
    """
    获取所有申万二级行业指数的日线数据
    返回: DataFrame，包含 板块代码、板块名称、日期、OHLCV
    """
    cache_name = f"all_sector_daily_{start}_{end}"
    cached = _load_cache(cache_name, max_age_hours=12)
    if cached is not None:
        return cached

    sectors = get_sector_list()
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    all_data = []
    for _, row in tqdm(sectors.iterrows(), total=len(sectors),
                       desc="获取板块日线"):
        code = row["行业代码"].replace(".SI", "")
        name = row["行业名称"]
        df = get_sector_daily(code)
        if df.empty:
            continue
        df = df[(df["日期"] >= start_dt) & (df["日期"] <= end_dt)].copy()
        df["板块名称"] = name
        df["板块代码"] = code
        all_data.append(df)
        time.sleep(0.3)  # 避免限流

    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    _save_cache(cache_name, result)
    return result


# ============ 个股数据 ============

def get_stock_daily(symbol: str, start: str, end: str,
                    adjust: str = "qfq",
                    max_retries: int = 3) -> pd.DataFrame:
    """获取个股日线行情（前复权），带自动重试"""
    cache_name = f"stock_{symbol}_{start}_{end}_{adjust}"
    cached = _load_cache(cache_name, max_age_hours=12)
    if cached is not None:
        return cached
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust=adjust,
            )
            _save_cache(cache_name, df)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 2)  # 递增等待
            else:
                print(f"获取股票 {symbol} 数据失败(重试{max_retries}次): {e}")
                return pd.DataFrame()


def get_stock_list() -> pd.DataFrame:
    """获取全 A 股实时行情列表（用于获取股票代码和名称）"""
    cached = _load_cache("stock_list_spot", max_age_hours=12)
    if cached is not None:
        return cached
    df = ak.stock_zh_a_spot_em()
    _save_cache("stock_list_spot", df)
    return df


def batch_get_stock_daily(codes: list, start: str, end: str,
                          adjust: str = "qfq") -> pd.DataFrame:
    """批量获取个股日线，带进度条和限流"""
    import hashlib
    codes_hash = hashlib.md5("_".join(sorted(codes)).encode()).hexdigest()[:8]
    cache_name = f"batch_stocks_{start}_{end}_{codes_hash}"
    cached = _load_cache(cache_name, max_age_hours=12)
    if cached is not None:
        return cached

    all_data = []
    for code in tqdm(codes, desc="获取个股日线"):
        df = get_stock_daily(code, start, end, adjust)
        if df.empty:
            continue
        all_data.append(df)
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()
    result = pd.concat(all_data, ignore_index=True)
    _save_cache(cache_name, result)
    return result


# ============ 大盘基准 ============

def get_market_daily(start: str, end: str) -> pd.DataFrame:
    """获取上证指数日线作为大盘基准"""
    cache_name = f"market_sh_{start}_{end}"
    cached = _load_cache(cache_name, max_age_hours=12)
    if cached is not None:
        return cached
    try:
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
        df = df.reset_index(drop=True)
        _save_cache(cache_name, df)
        return df
    except Exception as e:
        print(f"获取大盘数据失败: {e}")
        return pd.DataFrame()

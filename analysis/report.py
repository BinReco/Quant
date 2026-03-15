"""
绩效分析与可视化
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 尝试设置中文字体
plt.rcParams["axes.unicode_minus"] = False
for font_name in ["Arial Unicode MS", "Heiti TC", "Hiragino Sans",
                   "PingFang SC", "SimHei"]:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name]
        break
    except Exception:
        continue


def plot_equity_curve(daily_df: pd.DataFrame, save_path: str = "equity.png"):
    """绘制资金曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    daily_df["date"] = pd.to_datetime(daily_df["date"])

    # 1. 资金曲线
    ax1 = axes[0]
    ax1.plot(daily_df["date"], daily_df["total_value"],
             label="策略净值", linewidth=1.5)
    if "benchmark" in daily_df.columns and daily_df["benchmark"].sum() > 0:
        bench_norm = (daily_df["benchmark"]
                      / daily_df["benchmark"].iloc[0]
                      * daily_df["total_value"].iloc[0])
        ax1.plot(daily_df["date"], bench_norm,
                 label="沪指基准", linewidth=1, alpha=0.7)
    ax1.set_ylabel("资金")
    ax1.legend()
    ax1.set_title("策略回测 - 资金曲线")
    ax1.grid(True, alpha=0.3)

    # 2. 回撤曲线
    ax2 = axes[1]
    peak = daily_df["total_value"].cummax()
    drawdown = (daily_df["total_value"] - peak) / peak * 100
    ax2.fill_between(daily_df["date"], drawdown, 0,
                     alpha=0.3, color="red")
    ax2.set_ylabel("回撤 (%)")
    ax2.set_title("回撤")
    ax2.grid(True, alpha=0.3)

    # 3. 持仓数量
    ax3 = axes[2]
    ax3.bar(daily_df["date"], daily_df["positions"],
            alpha=0.5, color="steelblue")
    ax3.set_ylabel("持仓数")
    ax3.set_title("每日持仓数量")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"资金曲线已保存到 {save_path}")


def plot_trade_distribution(trades_df: pd.DataFrame,
                            save_path: str = "trades.png"):
    """绘制交易盈亏分布"""
    sells = trades_df[trades_df["direction"] == "SELL"].copy()
    if sells.empty:
        print("没有卖出交易，跳过分布图")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 盈亏分布
    ax1 = axes[0]
    colors = ["green" if x > 0 else "red" for x in sells["pnl"]]
    ax1.bar(range(len(sells)), sells["pnl"], color=colors, alpha=0.7)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_xlabel("交易序号")
    ax1.set_ylabel("盈亏 (元)")
    ax1.set_title("每笔交易盈亏")
    ax1.grid(True, alpha=0.3)

    # 持仓天数分布
    ax2 = axes[1]
    if "hold_days" in sells.columns:
        ax2.hist(sells["hold_days"], bins=range(1, 8),
                 alpha=0.7, color="steelblue", edgecolor="white")
        ax2.set_xlabel("持仓天数")
        ax2.set_ylabel("次数")
        ax2.set_title("持仓天数分布")
    else:
        # 按板块统计胜率
        sector_stats = sells.groupby("sector").agg(
            count=("pnl", "count"),
            win=("pnl", lambda x: (x > 0).sum()),
            total_pnl=("pnl", "sum"),
        )
        sector_stats["win_rate"] = sector_stats["win"] / sector_stats["count"]
        sector_stats = sector_stats.sort_values("total_pnl", ascending=True)
        colors = ["green" if x > 0 else "red"
                  for x in sector_stats["total_pnl"]]
        ax2.barh(sector_stats.index, sector_stats["total_pnl"], color=colors)
        ax2.set_title("各板块盈亏")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"交易分布图已保存到 {save_path}")


def plot_monthly_returns(daily_df: pd.DataFrame,
                         save_path: str = "monthly.png"):
    """绘制月度收益热力图"""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["daily_ret"] = df["total_value"].pct_change()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 计算月度收益率
    monthly = df.groupby(["year", "month"])["daily_ret"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    monthly.columns = ["year", "month", "return"]

    # 透视表
    pivot = monthly.pivot(index="year", columns="month", values="return")
    pivot.columns = [f"{m}月" for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 1.2)))

    # 绘制热力图
    data = pivot.values * 100  # 转为百分比
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # 在格子中标注数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 5 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    ax.set_title("月度收益率 (%)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"月度收益图已保存到 {save_path}")


def print_summary(results: dict):
    """打印回测结果摘要"""
    print("\n" + "=" * 50)
    print("           板块轮动策略 - 回测结果")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:>12s}: {v}")
    print("=" * 50)

"""
回测引擎

模拟交易执行，遵守 A 股规则：
- T+1：买入当天不能卖出
- 涨跌停限制
- 佣金 + 印花税
- 资金管理（等权分配）
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

from config.settings import (
    INITIAL_CAPITAL, MAX_POSITIONS, COMMISSION_RATE,
    STAMP_TAX_RATE, MIN_COMMISSION, MAX_HOLD_DAYS, STOP_LOSS, TAKE_PROFIT,
    TRAIL_STOP_PROFIT, TRAIL_STOP_RATIO,
)


@dataclass
class Position:
    code: str
    name: str
    sector: str
    buy_date: str
    buy_price: float
    shares: int
    hold_days: int = 0
    max_price: float = 0.0  # 持仓期间最高价（用于移动止盈）

    def __post_init__(self):
        if self.max_price == 0.0:
            self.max_price = self.buy_price


@dataclass
class Trade:
    date: str
    code: str
    name: str
    sector: str
    direction: str  # "BUY" / "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    tax: float
    pnl: float = 0.0  # 仅卖出时有值


@dataclass
class DailyRecord:
    date: str
    cash: float
    market_value: float
    total_value: float
    positions: int
    benchmark: float = 0.0


class BacktestEngine:
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions: dict[str, Position] = {}  # code -> Position
        self.trades: list[Trade] = []
        self.daily_records: list[DailyRecord] = []
        self.current_date = ""

    def _calc_commission(self, amount: float) -> float:
        comm = amount * COMMISSION_RATE
        return max(comm, MIN_COMMISSION)

    def _calc_tax(self, amount: float) -> float:
        return amount * STAMP_TAX_RATE

    def _current_total_value(self) -> float:
        """当前总资产（现金 + 持仓市值估算）"""
        if self.daily_records:
            return self.daily_records[-1].total_value
        return self.cash

    def buy(self, code: str, name: str, sector: str,
            price: float, date: str) -> bool:
        """
        买入股票

        返回 True 表示成交，False 表示无法买入
        """
        if code in self.positions:
            return False

        if len(self.positions) >= MAX_POSITIONS:
            return False

        # 动态仓位：按当前总资产等权分配
        total_value = self._current_total_value()
        target_amount = total_value / MAX_POSITIONS
        shares = int(target_amount / price / 100) * 100  # 整手（100股）

        if shares <= 0:
            return False

        amount = shares * price
        commission = self._calc_commission(amount)
        total_cost = amount + commission

        if total_cost > self.cash:
            # 资金不足，减少买入数量
            shares = int(self.cash / price / 100) * 100
            if shares <= 0:
                return False
            amount = shares * price
            commission = self._calc_commission(amount)
            total_cost = amount + commission
            if total_cost > self.cash:
                return False

        self.cash -= total_cost
        self.positions[code] = Position(
            code=code, name=name, sector=sector,
            buy_date=date, buy_price=price, shares=shares,
        )
        self.trades.append(Trade(
            date=date, code=code, name=name, sector=sector,
            direction="BUY", price=price, shares=shares,
            amount=amount, commission=commission, tax=0,
        ))
        return True

    def sell(self, code: str, price: float, date: str,
             reason: str = "") -> bool:
        """卖出股票"""
        if code not in self.positions:
            return False

        pos = self.positions[code]

        # T+1 检查
        if pos.buy_date == date:
            return False

        amount = pos.shares * price
        commission = self._calc_commission(amount)
        tax = self._calc_tax(amount)
        net = amount - commission - tax

        # 盈亏 = 卖出净收入 - 买入总成本（含买入佣金）
        buy_amount = pos.shares * pos.buy_price
        buy_commission = self._calc_commission(buy_amount)
        pnl = net - buy_amount - buy_commission

        self.cash += net
        del self.positions[code]

        self.trades.append(Trade(
            date=date, code=code, name=pos.name, sector=pos.sector,
            direction="SELL", price=price, shares=pos.shares,
            amount=amount, commission=commission, tax=tax, pnl=pnl,
        ))
        return True

    def update_daily(self, date: str,
                     price_map: dict[str, float],
                     benchmark_value: float = 0.0):
        """
        每日更新：记录持仓市值、更新最高价

        price_map: {code: 收盘价}
        """
        self.current_date = date

        # 计算持仓市值 & 更新最高价
        market_value = 0.0
        for code, pos in self.positions.items():
            current_price = price_map.get(code, pos.buy_price)
            pos.max_price = max(pos.max_price, current_price)
            market_value += pos.shares * current_price

        total = self.cash + market_value
        self.daily_records.append(DailyRecord(
            date=date,
            cash=self.cash,
            market_value=market_value,
            total_value=total,
            positions=len(self.positions),
            benchmark=benchmark_value,
        ))

    def check_exits(self, date: str,
                    price_map: dict[str, float],
                    trading_dates: list = None) -> list[str]:
        """
        检查需要卖出的股票（止损/移动止盈/固定止盈/到期）

        返回需要卖出的股票代码列表
        """
        to_sell = []
        for code, pos in list(self.positions.items()):
            if pos.buy_date == date:  # T+1
                continue

            current_price = price_map.get(code)
            if current_price is None:
                continue

            pnl_pct = current_price / pos.buy_price - 1
            max_pnl = pos.max_price / pos.buy_price - 1

            # 止损
            if pnl_pct <= STOP_LOSS:
                to_sell.append(code)
                continue

            # 移动止盈：盈利超过阈值后，从最高点回撤超过比例则卖出
            if max_pnl > TRAIL_STOP_PROFIT and pnl_pct < max_pnl * TRAIL_STOP_RATIO:
                to_sell.append(code)
                continue

            # 固定止盈
            if pnl_pct >= TAKE_PROFIT:
                to_sell.append(code)
                continue

            # 持仓到期（用交易日列表精确计算）
            hold_days = self._calc_hold_days(pos.buy_date, date, trading_dates)
            if hold_days >= MAX_HOLD_DAYS:
                to_sell.append(code)
                continue

        return to_sell

    @staticmethod
    def _calc_hold_days(buy_date: str, current_date: str,
                        trading_dates: list = None) -> int:
        """计算持仓交易日天数"""
        if trading_dates is not None:
            buy_dt = pd.to_datetime(buy_date)
            cur_dt = pd.to_datetime(current_date)
            return sum(1 for d in trading_dates if buy_dt < pd.to_datetime(d) <= cur_dt)
        # fallback: 用自然日近似
        delta = (pd.to_datetime(current_date) - pd.to_datetime(buy_date)).days
        return int(delta * 5 / 7)  # 粗略换算交易日

    def get_results(self) -> dict:
        """返回回测结果摘要"""
        if not self.daily_records:
            return {}

        df = pd.DataFrame([vars(r) for r in self.daily_records])
        df["daily_return"] = df["total_value"].pct_change()
        daily_ret = df["daily_return"].dropna()

        total_return = df["total_value"].iloc[-1] / self.initial_capital - 1
        trading_days = len(df)
        annual_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1

        # 最大回撤
        peak = df["total_value"].cummax()
        drawdown = (df["total_value"] - peak) / peak
        max_drawdown = drawdown.min()

        # 夏普比率 (无风险利率按 2%)
        risk_free = 0.02 / 252
        excess = daily_ret - risk_free
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)
                  if excess.std() > 0 else 0)

        # Sortino 比率（只用下行波动率）
        downside = excess[excess < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = (excess.mean() / downside_std * np.sqrt(252)
                   if downside_std > 0 else 0)

        # Calmar 比率（年化收益 / 最大回撤）
        calmar = (abs(annual_return / max_drawdown)
                  if max_drawdown != 0 else 0)

        # 交易统计
        sell_trades = [t for t in self.trades if t.direction == "SELL"]
        win_trades = [t for t in sell_trades if t.pnl > 0]
        win_rate = len(win_trades) / max(len(sell_trades), 1)

        total_profit = sum(t.pnl for t in win_trades)
        total_loss = sum(t.pnl for t in sell_trades if t.pnl <= 0)
        profit_factor = (abs(total_profit / total_loss)
                         if total_loss != 0 else float('inf'))

        avg_win = total_profit / len(win_trades) if win_trades else 0
        lose_trades = [t for t in sell_trades if t.pnl <= 0]
        avg_loss = total_loss / len(lose_trades) if lose_trades else 0

        return {
            "初始资金": f"{self.initial_capital:,.0f}",
            "最终资金": f"{df['total_value'].iloc[-1]:,.0f}",
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{annual_return:.2%}",
            "最大回撤": f"{max_drawdown:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Calmar": f"{calmar:.2f}",
            "总交易次数": len(self.trades),
            "卖出次数": len(sell_trades),
            "胜率": f"{win_rate:.2%}",
            "盈亏比": f"{profit_factor:.2f}",
            "平均盈利": f"{avg_win:,.0f}",
            "平均亏损": f"{avg_loss:,.0f}",
            "总盈利": f"{total_profit:,.0f}",
            "总亏损": f"{total_loss:,.0f}",
            "交易天数": trading_days,
        }

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([vars(t) for t in self.trades])

    def get_daily_df(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.daily_records])

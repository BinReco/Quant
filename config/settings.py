"""策略参数配置"""

# === 资金管理 ===
INITIAL_CAPITAL = 300_000  # 初始本金 30万
MAX_POSITIONS = 5          # 最大持仓数
POSITION_SIZE = INITIAL_CAPITAL / MAX_POSITIONS  # 单票仓位 6万

# === 交易成本 ===
COMMISSION_RATE = 0.0002   # 佣金万二（买卖双向）
STAMP_TAX_RATE = 0.001     # 印花税千一（卖出单边）
MIN_COMMISSION = 5.0       # 最低佣金5元

# === 策略参数 ===
SECTOR_TOP_N = 3           # 每日选取强度排名前N的板块
MAX_HOLD_DAYS = 15         # 最大持仓天数（5→15，让盈利奔跑）
STOP_LOSS = -0.07          # 止损线 -7%（-3%→-7%，减少被洗出）
TAKE_PROFIT = 0.15         # 止盈线 +15%（8%→15%，让利润奔跑）
TRAIL_STOP_PROFIT = 0.06   # 移动止盈启动阈值（3%→6%）
TRAIL_STOP_RATIO = 0.4     # 移动止盈回撤比例（从最高盈利回撤40%则卖出）

# === 大盘择时 ===
MARKET_TIMING = True       # 是否开启大盘择时
MARKET_MA_SHORT = 10       # 短期均线天数（20→10，更灵敏）
MARKET_MA_LONG = 60        # 长期均线天数

# === 板块强度参数 ===
STRENGTH_LOOKBACK = 20     # 强度计算回看天数（5→20，识别中期趋势而非短期脉冲）
VOLUME_SURGE_RATIO = 1.5   # 成交量放大倍数阈值
BREADTH_THRESHOLD = 0.6    # 板块内上涨个股比例阈值

# === 数据参数 ===
CACHE_DIR = "cache"        # 数据缓存目录
BACKTEST_START = "20210401" # 回测起始日期
BACKTEST_END = "20260315"   # 回测结束日期

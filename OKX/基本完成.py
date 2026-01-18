import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import re
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta
import urllib3
import sys
import numpy as np
import concurrent.futures
from typing import Dict, List, Optional
import math

PROXY_HOST = '127.0.0.1'
PROXY_PORT = 10809

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置系统代理
os.environ['http_proxy'] = f'http://{PROXY_HOST}:{PROXY_PORT}'
os.environ['https_proxy'] = f'http://{PROXY_HOST}:{PROXY_PORT}'

# 创建带代理的session函数
def create_proxy_session():
    """创建带代理的requests session"""
    session = requests.Session()
    session.proxies = {
        'http': f'http://{PROXY_HOST}:{PROXY_PORT}',
        'https': f'http://{PROXY_HOST}:{PROXY_PORT}'
    }
    session.verify = False  # 忽略SSL验证（仅用于测试）
    return session

# 加载环境变量
load_dotenv()

# ==============================================
# 检查必需的环境变量
# ==============================================
def check_required_env_vars():
    """检查必需的环境变量"""
    required_vars = {
        'DEEPSEEK_API_KEY': 'DeepSeek API密钥',
        'OKX_API_KEY': 'OKX API密钥',
        'OKX_SECRET': 'OKX API密钥',
        'OKX_PASSWORD': 'OKX交易密码'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        print("❌ 缺少必需的环境变量:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n请按照以下步骤设置:")
        print("1. 在脚本同目录创建 .env 文件")
        print("2. 添加以下内容:")
        print("   DEEPSEEK_API_KEY=你的DeepSeek API密钥")
        print("   OKX_API_KEY=你的OKX API密钥")
        print("   OKX_SECRET=你的OKX API密钥")
        print("   OKX_PASSWORD=你的OKX交易密码")
        print("\n或者在系统环境变量中设置这些值")
        return False
    
    return True

# 检查环境变量
if not check_required_env_vars():
    print("程序退出")
    sys.exit(1)

# 获取API密钥
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
okx_api_key = os.getenv('OKX_API_KEY')
okx_secret = os.getenv('OKX_SECRET')
okx_password = os.getenv('OKX_PASSWORD')

# ==============================================
# 初始化客户端
# ==============================================
try:
    # 初始化DeepSeek客户端
    deepseek_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )
    
    print(f"✅ DeepSeek客户端初始化成功")
    
except Exception as e:
    print(f"❌ DeepSeek客户端初始化失败: {e}")
    sys.exit(1)

# 创建代理session
proxy_session = create_proxy_session()

# 初始化OKX交易所
try:
    exchange = ccxt.okx({
        'options': {
            'defaultType': 'swap',  # OKX使用swap表示永续合约
        },
        'apiKey': okx_api_key,
        'secret': okx_secret,
        'password': okx_password,  # OKX需要交易密码
        'session': proxy_session,  # 添加代理session
        'enableRateLimit': True,    # 添加限速
        'timeout': 30000,           # 添加超时
    })
    
    print(f"✅ OKX交易所客户端初始化成功")
    
except Exception as e:
    print(f"❌ OKX交易所客户端初始化失败: {e}")
    sys.exit(1)

# ==============================================
# 多币种配置 - 简化版（仅保留基本信息）
# ==============================================
TRADE_SYMBOLS = [
    {
        'symbol': 'BTC/USDT:USDT',
        'display_name': 'BTC',
        'contract_size': 0.01,  # 合约乘数
        'min_position': 0.01,   # 最小交易量
        'max_position': 10,     # 最大交易量
        'default_leverage': 10  # 默认杠杆（仅作备用）
    },
    {
        'symbol': 'ETH/USDT:USDT',
        'display_name': 'ETH',
        'contract_size': 0.1,
        'min_position': 0.1,
        'max_position': 50,
        'default_leverage': 15
    },
    {
        'symbol': 'BNB/USDT:USDT',
        'display_name': 'BNB',
        'contract_size': 0.01,
        'min_position': 0.1,
        'max_position': 100,
        'default_leverage': 20
    }
]

# 全局配置
GLOBAL_CONFIG = {
    'test_mode': False,
    'max_total_exposure': 30,  # 最大总风险暴露百分比
    'enable_safety_limits': True,  # 启用安全限制
    'parallel_fetch': True,
    'max_leverage': 25,  # 最大允许杠杆
    'min_leverage': 1,   # 最小允许杠杆
    'max_risk_per_trade': 5.0,  # 单笔交易最大风险资金比例
    'min_balance_for_trade': 10.0  # 最小交易余额（USDT）
}

# ==============================================
# 全局变量存储历史数据
# ==============================================
price_history = {}
signal_history = {}
positions = {}
symbol_configs = {}  # 存储每个币种的配置信息

# ==============================================
# 原有函数（需要保持的部分）
# ==============================================
# 以下是您原有脚本中的函数，需要保留：
# 1. calculate_technical_indicators
# 2. get_support_resistance_levels  
# 3. get_market_trend
# 4. get_symbol_ohlcv_enhanced
# 5. fetch_all_symbols_data_parallel
# 6. generate_technical_analysis_text
# 7. get_current_position
# 8. get_sentiment_indicators
# 9. safe_json_parse
# 10. display_deepseek_analysis_results
# 11. wait_for_next_period

# 注意：由于我们重写了分析函数，所以不需要原来的 analyze_with_deepseek_for_symbol
# 但需要保留上面列出的其他技术分析函数

# ==============================================
# 简化的技术指标计算（保留核心函数）
# ==============================================
def calculate_technical_indicators(df):
    """计算技术指标"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df

def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]
        
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'rsi': df['rsi'].iloc[-1]
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}

def get_symbol_ohlcv_enhanced(symbol_config):
    """获取指定币种的K线数据"""
    try:
        symbol = symbol_config['symbol']
        timeframe = '15m'  # 固定为15分钟
        data_points = 96   # 固定数据点数
        
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=data_points)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data
        
        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        
        return {
            'symbol': symbol,
            'display_name': symbol_config['display_name'],
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': timeframe,
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_dict('records'),
            'technical_data': {
                'sma_5': current_data.get('sma_5', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_50': current_data.get('sma_50', 0),
                'rsi': current_data.get('rsi', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0)
            },
            'trend_analysis': trend_analysis,
            'full_data': df
        }
    except Exception as e:
        print(f"❌ 获取 {symbol_config['display_name']} K线数据失败: {e}")
        return None

def fetch_all_symbols_data_parallel():
    """并行获取所有币种数据"""
    price_data_dict = {}
    
    if GLOBAL_CONFIG['parallel_fetch']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_symbol = {
                executor.submit(get_symbol_ohlcv_enhanced, symbol_config): symbol_config 
                for symbol_config in TRADE_SYMBOLS
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol_config = future_to_symbol[future]
                try:
                    price_data = future.result()
                    if price_data:
                        price_data_dict[symbol_config['symbol']] = price_data
                except Exception as e:
                    print(f"❌ {symbol_config['display_name']} 数据获取失败: {e}")
    else:
        # 串行获取
        for symbol_config in TRADE_SYMBOLS:
            price_data = get_symbol_ohlcv_enhanced(symbol_config)
            if price_data:
                price_data_dict[symbol_config['symbol']] = price_data
    
    return price_data_dict

def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    if 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})

    # 检查数据有效性
    def safe_float(value, default=0):
        return float(value) if value is not None and pd.notna(value) else default

    analysis_text = f"""
    【技术指标分析】
    
    📈 移动平均线:
    - 5周期: {safe_float(tech['sma_5']):.2f} 
    - 20周期: {safe_float(tech['sma_20']):.2f}
    - 50周期: {safe_float(tech['sma_50']):.2f}
    
    📊 动量指标:
    - RSI: {safe_float(tech['rsi']):.1f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f} ({'金叉' if safe_float(tech['macd']) > safe_float(tech['macd_signal']) else '死叉'})
    
    🎚️ 布林带:
    - 上轨: {safe_float(tech['bb_upper']):.2f}
    - 下轨: {safe_float(tech['bb_lower']):.2f}
    - 当前价格相对位置: {(price_data['price'] - safe_float(tech['bb_lower'])) / (safe_float(tech['bb_upper']) - safe_float(tech['bb_lower'])):.2%}
    
    📈 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    """
    return analysis_text

def get_current_position(symbol):
    """获取当前持仓情况 - 指定币种"""
    try:
        positions_list = exchange.fetch_positions([symbol])
        
        for pos in positions_list:
            if pos['symbol'] == symbol:
                contracts = float(pos['contracts']) if pos['contracts'] else 0
                
                if contracts > 0:
                    return {
                        'side': pos['side'],
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else None,
                        'symbol': pos['symbol']
                    }
        
        return None
        
    except Exception as e:
        print(f"❌ 获取 {symbol} 持仓失败: {e}")
        return None

def get_sentiment_indicators():
    """获取情绪指标 - 简化版本"""
    try:
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = "7ad48a56-8730-4238-a714-eebc30834e3e"

        # 获取最近4小时数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": ["BTC"]
        }

        headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
        response = proxy_session.post(API_URL, json=request_body, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200 and data.get("data"):
                time_periods = data["data"][0]["timePeriods"]

                for period in time_periods:
                    period_data = period.get("data", [])
                    sentiment = {}
                    valid_data_found = False

                    for item in period_data:
                        endpoint = item.get("endpoint")
                        value = item.get("value", "").strip()

                        if value:
                            try:
                                if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                    sentiment[endpoint] = float(value)
                                    valid_data_found = True
                            except (ValueError, TypeError):
                                continue

                    if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                        positive = sentiment['CO-A-02-01']
                        negative = sentiment['CO-A-02-02']
                        net_sentiment = positive - negative
                        
                        data_delay = int((datetime.now() - datetime.strptime(
                            period['startTime'], '%Y-%m-%d %H:%M:%S')).total_seconds() // 60)

                        print(f"✅ 使用情绪数据时间: {period['startTime']} (延迟: {data_delay}分钟)")

                        return {
                            'positive_ratio': positive,
                            'negative_ratio': negative,
                            'net_sentiment': net_sentiment,
                            'data_time': period['startTime'],
                            'data_delay_minutes': data_delay
                        }

                print("❌ 所有时间段数据都为空")
                return None

        return None
    except Exception as e:
        print(f"情绪指标获取失败: {e}")
        return None

def safe_json_parse(json_str):
    """安全解析JSON"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return None

def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（00, 15, 30, 45分钟）
    next_period_minute = ((current_minute // 15) + 1) * 15
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return seconds_to_wait

# ==============================================
# 以下是全权决策的核心函数
# ==============================================

def analyze_with_deepseek_full_control(price_data, symbol_config, account_info):
    """让DeepSeek全权负责交易决策"""
    
    # 获取市场数据
    technical_analysis = generate_technical_analysis_text(price_data)
    
    # 获取持仓信息
    current_position = get_current_position(price_data['symbol'])
    position_text = "无持仓" if not current_position else f"{current_position['side']}仓, 数量: {current_position['size']:.2f}张, 盈亏: {current_position['unrealized_pnl']:+.2f}USDT"
    
    # 获取情绪数据
    sentiment_data = get_sentiment_indicators()
    sentiment_text = "【市场情绪】数据暂不可用"
    if sentiment_data:
        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        sentiment_text = f"【市场情绪】乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
    
    # 构建完整的账户信息
    account_summary = f"""
    【账户信息】
    - 总资产: {account_info['total']:.2f} USDT
    - 可用余额: {account_info['free']:.2f} USDT
    - 当前持仓价值: {account_info['position_value']:.2f} USDT
    - 风险暴露: {account_info['exposure_pct']:.1f}%
    - 已用保证金: {account_info['used_margin']:.2f} USDT
    """
    
    # 构建K线数据
    kline_text = f"【{symbol_config['display_name']} 最近5根K线数据】\n"
    for i, kline in enumerate(price_data['kline_data'][-5:]):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"
    
    # 完整的提示词
    prompt = f"""
    你是一个专业的加密货币全权交易员。现在请基于以下信息做出完整的交易决策：

    {account_summary}

    {kline_text}

    {technical_analysis}

    {sentiment_text}

    【当前行情】
    - 币种: {symbol_config['display_name']} ({price_data['symbol']})
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}
    - 合约乘数: {symbol_config['contract_size']}
    - 最小交易量: {symbol_config['min_position']}张
    - 最大交易量: {symbol_config['max_position']}张
    - 可用最大杠杆: {GLOBAL_CONFIG['max_leverage']}x

    【交易参数约束】
    1. 杠杆范围: {GLOBAL_CONFIG['min_leverage']}-{GLOBAL_CONFIG['max_leverage']}倍
    2. 仓位大小: {symbol_config['min_position']}-{symbol_config['max_position']}张合约
    3. 单笔最大风险: {GLOBAL_CONFIG['max_risk_per_trade']}%的账户资产

    【你的决策权限】
    作为全权交易员，你需要决定：
    1. 是否交易（BUY/SELL/HOLD）
    2. 使用多少杠杆
    3. 开仓多少张合约
    4. 止盈止损价格
    5. 订单类型（市价单/限价单）

    【风险管理要求】
    1. 高波动市场应降低杠杆和仓位
    2. 低信心时使用小仓位或观望

    【必须遵守的JSON格式】
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "你的详细分析理由，至少3点",
        "stop_loss": 具体数字,
        "take_profit": 具体数字,
        "confidence": "HIGH|MEDIUM|LOW",
        "position_size": 具体数字（合约张数，如0.25）,
        "leverage": 具体数字（如12）,
        "risk_percentage": 具体数字（如2.5，表示风险资金比例）,
        "order_type": "market|limit",
        "limit_price": 具体数字（如果order_type为limit）
    }}
    """
    
    try:
        print(f"🤖 正在调用DeepSeek API进行全权决策...")
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "你是一位经验丰富的全权交易员，拥有完整的交易决策权。你需要综合考虑技术分析、市场情绪、账户风险和资金管理，做出最优的交易决策。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        print(f"✅ DeepSeek全权决策API调用成功")
        
        # 提取JSON部分
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)
            
            if signal_data:
                # 添加验证和修正
                signal_data = validate_and_correct_decision(signal_data, symbol_config, account_info, price_data)
                signal_data['timestamp'] = price_data['timestamp']
                signal_data['price'] = price_data['price']
                
                return signal_data
        
        print("⚠️ 无法解析JSON，使用保守备用策略")
        return create_conservative_fallback(price_data, account_info)
        
    except Exception as e:
        print(f"❌ DeepSeek全权决策失败: {e}")
        return create_conservative_fallback(price_data, account_info)

def safe_json_parse_full(json_str):
    """安全解析完整的交易决策JSON"""
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None

def validate_and_correct_decision(signal_data, symbol_config, account_info, price_data):
    """验证并修正DeepSeek的决策"""
    
    corrected = signal_data.copy()
    current_price = price_data['price']
    
    # 1. 验证杠杆范围
    min_leverage = GLOBAL_CONFIG['min_leverage']
    max_leverage = GLOBAL_CONFIG['max_leverage']
    if 'leverage' not in corrected:
        corrected['leverage'] = symbol_config['default_leverage']
    elif corrected['leverage'] < min_leverage:
        corrected['leverage'] = min_leverage
    elif corrected['leverage'] > max_leverage:
        corrected['leverage'] = max_leverage
    
    # 2. 验证仓位大小
    min_position = symbol_config['min_position']
    max_position = symbol_config['max_position']
    if 'position_size' not in corrected:
        corrected['position_size'] = min_position
    elif corrected['position_size'] < min_position:
        corrected['position_size'] = min_position
    elif corrected['position_size'] > max_position:
        corrected['position_size'] = max_position
    
    # 3. 验证风险比例
    max_risk = GLOBAL_CONFIG['max_risk_per_trade']
    if 'risk_percentage' not in corrected:
        corrected['risk_percentage'] = 1.0
    elif corrected['risk_percentage'] > max_risk:
        corrected['risk_percentage'] = max_risk
    
    # 4. 验证订单类型
    if 'order_type' not in corrected:
        corrected['order_type'] = 'market'
    
    # 5. 验证止损和止盈价格
    if 'stop_loss' not in corrected or corrected['stop_loss'] is None or corrected['stop_loss'] <= 0:
        corrected['stop_loss'] = current_price * 0.98 if current_price > 0 else 0
    
    if 'take_profit' not in corrected or corrected['take_profit'] is None or corrected['take_profit'] <= 0:
        corrected['take_profit'] = current_price * 1.02 if current_price > 0 else 0
    
    return corrected

def create_conservative_fallback(price_data, account_info):
    """创建保守的备用决策"""
    current_price = price_data['price']
    
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守观望策略",
        "stop_loss": current_price * 0.98 if current_price > 0 else 0,
        "take_profit": current_price * 1.02 if current_price > 0 else 0,
        "confidence": "LOW",
        "position_size": 0,
        "leverage": 1,
        "risk_percentage": 0,
        "order_type": "market",
        "is_fallback": True
    }

def get_account_info(symbol_config, current_price):
    """获取完整的账户信息"""
    try:
        balance = exchange.fetch_balance()
        
        # 计算持仓价值
        position = get_current_position(symbol_config['symbol'])
        position_value = 0
        if position:
            position_value = abs(position['size'] * symbol_config['contract_size'] * position.get('entry_price', current_price))
        
        total_balance = balance['USDT']['total'] if 'USDT' in balance and 'total' in balance['USDT'] else 0
        free_balance = balance['USDT']['free'] if 'USDT' in balance and 'free' in balance['USDT'] else 0
        
        # 计算风险暴露
        exposure_pct = (position_value / total_balance * 100) if total_balance > 0 else 0
        
        # 简化计算：已用保证金 ≈ 持仓价值 / 平均杠杆
        used_margin = position_value / 10 if position_value > 0 else 0
        
        return {
            'total': total_balance,
            'free': free_balance,
            'position_value': position_value,
            'exposure_pct': exposure_pct,
            'used_margin': used_margin,
            'current_price': current_price
        }
    except Exception as e:
        print(f"❌ 获取账户信息失败: {e}")
        return {
            'total': 0,
            'free': 0,
            'position_value': 0,
            'exposure_pct': 0,
            'used_margin': 0,
            'current_price': current_price
        }

def display_full_control_results(symbol_name, signal_data, price_data, account_info):
    """显示全权决策结果"""
    print(f"\n{'='*60}")
    print(f"🤖 {symbol_name} DeepSeek全权决策引擎")
    print(f"{'='*60}")
    
    print(f"\n📊 【账户状况】")
    print(f"   - 总资产: ${account_info['total']:,.2f}")
    print(f"   - 可用余额: ${account_info['free']:,.2f}")
    print(f"   - 持仓价值: ${account_info['position_value']:,.2f}")
    print(f"   - 风险暴露: {account_info['exposure_pct']:.1f}%")
    
    print(f"\n🎯 【交易决策】")
    signal_icon = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal_data['signal'], '⚪')
    print(f"   {signal_icon} 交易方向: {signal_data['signal']}")
    
    confidence_icon = {'HIGH': '⭐⭐⭐', 'MEDIUM': '⭐⭐', 'LOW': '⭐'}.get(signal_data['confidence'], '?')
    print(f"   {confidence_icon} 信心程度: {signal_data['confidence']}")
    
    print(f"\n⚙️ 【仓位管理】")
    print(f"   - 合约张数: {signal_data['position_size']:.2f}张")
    print(f"   - 杠杆倍数: {signal_data['leverage']}x")
    print(f"   - 风险资金: {signal_data['risk_percentage']:.1f}%")
    print(f"   - 订单类型: {signal_data['order_type']}")
    
    print(f"\n⚠️ 【风险管理】")
    current_price = price_data['price']
    
    # 安全计算止损和止盈百分比
    try:
        if signal_data['stop_loss'] is not None and current_price > 0:
            sl_pct = (signal_data['stop_loss'] - current_price) / current_price * 100
            print(f"   - 止损价格: ${signal_data['stop_loss']:,.2f} ({sl_pct:+.2f}%)")
        else:
            print(f"   - 止损价格: ${0:,.2f} (N/A)")
        
        if signal_data['take_profit'] is not None and current_price > 0:
            tp_pct = (signal_data['take_profit'] - current_price) / current_price * 100
            print(f"   - 止盈价格: ${signal_data['take_profit']:,.2f} ({tp_pct:+.2f}%)")
        else:
            print(f"   - 止盈价格: ${0:,.2f} (N/A)")
    except Exception as e:
        print(f"   - 风险管理计算错误: {e}")
    
    print(f"\n📝 【分析理由】")
    reason = signal_data['reason']
    # 简化显示理由
    lines = reason.split('.')
    for i, line in enumerate(lines[:3]):  # 只显示前3点
        if line.strip():
            print(f"   • {line.strip()}.")
    
    if signal_data.get('is_fallback', False):
        print(f"\n⚠️ 【备用策略】")
        print(f"   使用保守备用策略")
    
    print("=" * 60)

def perform_safety_check(signal_data, account_info):
    """执行安全检查"""
    
    # 1. 检查总风险暴露
    if account_info['exposure_pct'] > GLOBAL_CONFIG['max_total_exposure']:
        print(f"🚨 总风险暴露{account_info['exposure_pct']:.1f}%超过限制{GLOBAL_CONFIG['max_total_exposure']}%")
        return False
    
    # 2. 检查单笔风险
    if signal_data['risk_percentage'] > GLOBAL_CONFIG['max_risk_per_trade']:
        print(f"🚨 单笔风险{signal_data['risk_percentage']:.1f}%超过限制{GLOBAL_CONFIG['max_risk_per_trade']}%")
        return False
    
    return True

def check_balance_for_trade(signal_data, account_info):
    """检查余额是否足够执行交易"""
    
    # 获取最小交易余额要求
    min_balance = GLOBAL_CONFIG.get('min_balance_for_trade', 10.0)
    
    # 检查可用余额是否大于最小要求
    if account_info['free'] < min_balance:
        print(f"💰 余额不足: 可用余额${account_info['free']:.2f} < 最小要求${min_balance:.2f}")
        return False
    
    # 计算所需保证金
    if signal_data['signal'] != 'HOLD' and signal_data['position_size'] > 0:
        # 简化计算所需保证金
        position_value = signal_data['position_size'] * 0.01 * account_info['current_price']
        required_margin = position_value / signal_data['leverage']
        
        # 加上风险资金
        risk_amount = account_info['total'] * (signal_data['risk_percentage'] / 100)
        total_required = required_margin + risk_amount
        
        if account_info['free'] < total_required:
            print(f"💰 保证金不足: 所需${total_required:.2f} > 可用${account_info['free']:.2f}")
            return False
    
    return True

def calculate_required_margin(signal_data, current_price, total_balance):
    """计算所需保证金"""
    # 简化计算：仓位价值 / 杠杆
    position_value = signal_data['position_size'] * symbol_config.get('contract_size', 0.01) * current_price
    required_margin = position_value / signal_data['leverage']
    
    # 加上风险资金
    risk_amount = total_balance * (signal_data['risk_percentage'] / 100)
    required_margin += risk_amount
    
    return required_margin

def execute_full_control_trade(symbol_config, signal_data, price_data, account_info):
    """执行DeepSeek全权决策的交易"""
    
    symbol = symbol_config['symbol']
    display_name = symbol_config['display_name']
    
    print(f"\n🎯 {display_name} 执行全权决策交易")
    print(f"📊 决策详情:")
    print(f"   - 信号: {signal_data['signal']}")
    print(f"   - 信心: {signal_data['confidence']}")
    print(f"   - 仓位: {signal_data['position_size']:.2f}张")
    print(f"   - 杠杆: {signal_data['leverage']}x")
    print(f"   - 风险资金: {signal_data['risk_percentage']:.1f}%")
    print(f"   - 止损: ${signal_data.get('stop_loss', 0):.2f}")
    print(f"   - 止盈: ${signal_data.get('take_profit', 0):.2f}")
    
    if signal_data.get('is_fallback', False):
        print(f"⚠️ 使用备用保守策略，不执行交易")
        return
    
    # 检查余额是否足够
    if not check_balance_for_trade(signal_data, account_info):
        print(f"💰 余额不足，跳过当前开单，继续执行脚本循环")
        return
    
    if GLOBAL_CONFIG['test_mode']:
        print(f"🔧 测试模式 - 仅模拟交易")
        return
    
    # 安全检查
    if not perform_safety_check(signal_data, account_info):
        print(f"🚨 安全检查失败，取消交易")
        return
    
    try:
        # 1. 设置杠杆
        print(f"⚙️ 设置杠杆为{signal_data['leverage']}x...")
        try:
            exchange.set_leverage(
                signal_data['leverage'],
                symbol,
                {'mgnMode': 'cross'}
            )
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ 设置杠杆失败: {e}")
            
        # 2. 获取当前持仓
        current_position = get_current_position(symbol)
        
        # 3. 根据信号执行交易
        if signal_data['signal'] == 'BUY':
            execute_simple_buy_trade(symbol_config, signal_data, current_position)
        elif signal_data['signal'] == 'SELL':
            execute_simple_sell_trade(symbol_config, signal_data, current_position)
        elif signal_data['signal'] == 'HOLD':
            print("🤚 观望策略，不执行交易")
        
        print(f"✅ {display_name} 全权决策交易执行完成")
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ {display_name} 交易执行失败: {e}")

def execute_simple_buy_trade(symbol_config, signal_data, current_position):
    """执行简化的买入交易"""
    symbol = symbol_config['symbol']
    position_size = signal_data['position_size']
    
    if current_position and current_position['side'] == 'short':
        # 平空仓
        if current_position['size'] > 0:
            exchange.create_market_order(
                symbol,
                'buy',
                current_position['size'],
                params={'reduceOnly': True, 'tag': 'DeepSeek_FC'}
            )
            time.sleep(1)
        
        # 开多仓
        if position_size > 0:
            exchange.create_market_order(
                symbol,
                'buy',
                position_size,
                params={'tag': 'DeepSeek_FC'}
            )
    
    elif current_position and current_position['side'] == 'long':
        # 调整多仓
        size_diff = position_size - current_position['size']
        if abs(size_diff) >= symbol_config['min_position']:
            if size_diff > 0:
                # 加仓
                exchange.create_market_order(
                    symbol,
                    'buy',
                    size_diff,
                    params={'tag': 'DeepSeek_FC'}
                )
            else:
                # 减仓
                exchange.create_market_order(
                    symbol,
                    'sell',
                    abs(size_diff),
                    params={'reduceOnly': True, 'tag': 'DeepSeek_FC'}
                )
        else:
            print(f"📊 仓位合适，保持现状")
    
    else:
        # 新开多仓
        if position_size > 0:
            exchange.create_market_order(
                symbol,
                'buy',
                position_size,
                params={'tag': 'DeepSeek_FC'}
            )

def execute_simple_sell_trade(symbol_config, signal_data, current_position):
    """执行简化的卖出交易"""
    symbol = symbol_config['symbol']
    position_size = signal_data['position_size']
    
    if current_position and current_position['side'] == 'long':
        # 平多仓
        if current_position['size'] > 0:
            exchange.create_market_order(
                symbol,
                'sell',
                current_position['size'],
                params={'reduceOnly': True, 'tag': 'DeepSeek_FC'}
            )
            time.sleep(1)
        
        # 开空仓
        if position_size > 0:
            exchange.create_market_order(
                symbol,
                'sell',
                position_size,
                params={'tag': 'DeepSeek_FC'}
            )
    
    elif current_position and current_position['side'] == 'short':
        # 调整空仓
        size_diff = position_size - current_position['size']
        if abs(size_diff) >= symbol_config['min_position']:
            if size_diff > 0:
                # 加仓
                exchange.create_market_order(
                    symbol,
                    'sell',
                    size_diff,
                    params={'tag': 'DeepSeek_FC'}
                )
            else:
                # 减仓
                exchange.create_market_order(
                    symbol,
                    'buy',
                    abs(size_diff),
                    params={'reduceOnly': True, 'tag': 'DeepSeek_FC'}
                )
        else:
            print(f"📊 仓位合适，保持现状")
    
    else:
        # 新开空仓
        if position_size > 0:
            exchange.create_market_order(
                symbol,
                'sell',
                position_size,
                params={'tag': 'DeepSeek_FC'}
            )

def multi_symbol_full_control_bot():
    """多币种全权决策交易机器人"""
    
    # 等待到整点
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    
    print(f"\n{'='*60}")
    print(f"DeepSeek全权决策交易 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 获取所有币种数据
    print(f"📊 获取{len(TRADE_SYMBOLS)}个币种数据...")
    price_data_dict = fetch_all_symbols_data_parallel()
    
    if not price_data_dict:
        print("❌ 未获取到任何币种数据")
        return
    
    print(f"✅ 成功获取{len(price_data_dict)}个币种数据")
    
    # 逐个分析并交易
    for symbol_config in TRADE_SYMBOLS:
        symbol = symbol_config['symbol']
        
        if symbol in price_data_dict:
            price_data = price_data_dict[symbol]
            current_price = price_data['price']
            
            print(f"\n{'━'*40}")
            print(f"📈 分析 {symbol_config['display_name']} ({symbol})")
            print(f"当前价格: ${current_price:,.2f}")
            
            # 获取账户信息
            account_info = get_account_info(symbol_config, current_price)
            
            # 让DeepSeek全权决策
            signal_data = analyze_with_deepseek_full_control(price_data, symbol_config, account_info)
            
            # 保存信号历史
            if symbol not in signal_history:
                signal_history[symbol] = []
            signal_history[symbol].append(signal_data)
            if len(signal_history[symbol]) > 50:
                signal_history[symbol].pop(0)
            
            # 显示结果
            display_full_control_results(symbol_config['display_name'], signal_data, price_data, account_info)
            
            # 执行交易
            execute_full_control_trade(symbol_config, signal_data, price_data, account_info)
        else:
            print(f"❌ {symbol_config['display_name']} 数据获取失败，跳过")

def main():
    """主函数"""
    print("=" * 60)
    print("DeepSeek全权决策交易系统启动成功！")
    print("🎯 决策权限：DeepSeek 100%控制")
    print(f"支持币种: {len(TRADE_SYMBOLS)}个")
    print("=" * 60)
    
    if GLOBAL_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，DeepSeek全权决策！")
        print("⚠️ 注意：所有交易决策均由AI自动执行")
    
    # 显示配置
    print("\n📋 系统配置:")
    print(f"   - 最大杠杆: {GLOBAL_CONFIG['max_leverage']}x")
    print(f"   - 单笔最大风险: {GLOBAL_CONFIG['max_risk_per_trade']}%")
    print(f"   - 总风险限制: {GLOBAL_CONFIG['max_total_exposure']}%")
    print(f"   - 最小交易余额: ${GLOBAL_CONFIG['min_balance_for_trade']}")
    
    print("\n🎯 交易币种:")
    for i, symbol_config in enumerate(TRADE_SYMBOLS, 1):
        print(f"  {i}. {symbol_config['display_name']} ({symbol_config['symbol']})")
        print(f"     合约乘数: {symbol_config['contract_size']}")
        print(f"     仓位范围: {symbol_config['min_position']}-{symbol_config['max_position']}张")
    
    # 初始化交易所（简化设置）
    try:
        exchange.load_markets()
        print("✅ 交易所初始化成功")
    except Exception as e:
        print(f"❌ 交易所初始化失败: {e}")
        return
    
    print(f"\n🔄 执行频率: 每15分钟整点执行")
    print(f"🤖 AI权限: 全权控制交易方向、仓位、杠杆、风险")
    
    # 主循环
    while True:
        try:
            multi_symbol_full_control_bot()
            time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            print("\n👋 用户中断，程序退出")
            break
        except Exception as e:
            print(f"❌ 主循环异常: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(300)

if __name__ == "__main__":
    main()

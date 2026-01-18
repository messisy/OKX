import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
from dotenv import load_dotenv
import urllib3
import requests

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================
# 代理配置
# ==============================================
PROXY_HOST = '127.0.0.1'
PROXY_PORT = 10809

# 设置系统代理
os.environ['http_proxy'] = f'http://{PROXY_HOST}:{PROXY_PORT}'
os.environ['https_proxy'] = f'http://{PROXY_HOST}:{PROXY_PORT}'

load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 创建带代理的session
def create_proxy_session():
    """创建带代理的requests session"""
    session = requests.Session()
    session.proxies = {
        'http': f'http://{PROXY_HOST}:{PROXY_PORT}',
        'https': f'http://{PROXY_HOST}:{PROXY_PORT}'
    }
    session.verify = False  # 忽略SSL验证（仅用于测试）
    return session

# 初始化OKX交易所（带代理）
proxy_session = create_proxy_session()
exchange = ccxt.okx({
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {
        'defaultType': 'swap',
    },
    'session': proxy_session,  # 使用带代理的session
})

# 交易参数配置
TRADE_CONFIG = {
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'amount': 0.001,  # 交易数量 (BTC) - 从0.01改为0.001，更安全
    'leverage': 3,    # 杠杆倍数 - 从10改为3，降低风险
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': True,   # 先使用测试模式!!!
}

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

def setup_exchange():
    """设置交易所参数"""
    try:
        print("正在初始化交易所连接...")
        
        # 测试连接
        ticker = exchange.fetch_ticker(TRADE_CONFIG['symbol'])
        print(f"连接成功！BTC当前价格: ${ticker['last']:,.2f}")
        
        # 获取余额
        balance = exchange.fetch_balance()
        usdt_total = balance.get('USDT', {}).get('total', 0)
        usdt_free = balance.get('USDT', {}).get('free', 0)
        print(f"账户余额 - 总额: {usdt_total:.2f} USDT, 可用: {usdt_free:.2f} USDT")
        
        # 设置杠杆（如果需要）
        if not TRADE_CONFIG['test_mode']:
            try:
                exchange.set_leverage(
                    TRADE_CONFIG['leverage'],
                    TRADE_CONFIG['symbol'],
                    {'mgnMode': 'cross'}
                )
                print(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")
            except Exception as e:
                print(f"设置杠杆失败（可能已设置）: {e}")
        
        # 检查是否为模拟账户
        if usdt_total == 0 and usdt_free == 0:
            print("⚠ 警告：账户USDT余额为0，请确认：")
            print("  1. 是否为模拟/测试账户")
            print("  2. 是否已充值USDT")
            print("  3. 当前模式: {'实盘' if not TRADE_CONFIG['test_mode'] else '模拟'}")
        
        return True
    except Exception as e:
        print(f"交易所设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_btc_ohlcv():
    """获取BTC/USDT的K线数据"""
    try:
        # 获取最近10根K线
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'], limit=10)

        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_dict('records')
        }
    except Exception as e:
        print(f"获取K线数据失败: {e}")
        return None

def get_current_position():
    """获取当前持仓情况"""
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    return {
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
                    }

        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号"""

    # 添加当前价格到历史记录
    price_history.append(price_data)
    if len(price_history) > 20:
        price_history.pop(0)

    # 构建K线数据文本
    kline_text = f"【最近5根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data']):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

    # 构建技术指标文本
    if len(price_history) >= 5:
        closes = [data['price'] for data in price_history[-5:]]
        sma_5 = sum(closes) / len(closes)
        price_vs_sma = ((price_data['price'] - sma_5) / sma_5) * 100

        indicator_text = f"【技术指标】\n5周期均价: {sma_5:.2f}\n当前价格相对于均线: {price_vs_sma:+.2f}%"
    else:
        indicator_text = "【技术指标】\n数据不足计算技术指标"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下BTC/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {indicator_text}

    {signal_text}

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} BTC
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}

    【分析要求】
    1. 基于{TRADE_CONFIG['timeframe']}K线趋势和技术指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（考虑趋势连续性、支撑阻力、成交量等因素）
    3. 基于技术分析建议合理的止损价位
    4. 基于技术分析建议合理的止盈价位
    5. 评估信号信心程度

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格,
        "take_profit": 具体价格,
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f"您是一位急需资金为母亲治疗癌症的交易员，你现在需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = json.loads(json_str)
        else:
            print(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        print(f"DeepSeek分析失败: {e}")
        return None

def execute_trade(signal_data, price_data):
    """执行交易"""
    global position

    current_position = get_current_position()

    print(f"\n交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"理由: {signal_data['reason']}")
    print(f"止损: ${signal_data['stop_loss']:,.2f}")
    print(f"止盈: ${signal_data['take_profit']:,.2f}")
    print(f"当前持仓: {current_position}")

    if TRADE_CONFIG['test_mode']:
        print("🔧 测试模式 - 仅模拟交易，不下单")
        
        # 模拟交易逻辑
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("模拟：平空仓并开多仓")
            elif not current_position:
                print("模拟：开多仓")
            else:
                print("模拟：已持有多仓，无需操作")
                
        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("模拟：平多仓并开空仓")
            elif not current_position:
                print("模拟：开空仓")
            else:
                print("模拟：已持有空仓，无需操作")
                
        elif signal_data['signal'] == 'HOLD':
            print("模拟：建议观望，不执行交易")
            
        return

    try:
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("平空仓并开多仓...")
                # 平空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'auto_trade'}
                )
                time.sleep(1)
                # 开多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'auto_trade'}
                )
            elif not current_position:
                print("开多仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'auto_trade'}
                )
            else:
                print("已持有多仓，无需操作")

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("平多仓并开空仓...")
                # 平多仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    current_position['size'],
                    params={'reduceOnly': True, 'tag': 'auto_trade'}
                )
                time.sleep(1)
                # 开空仓
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'auto_trade'}
                )
            elif not current_position:
                print("开空仓...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    TRADE_CONFIG['amount'],
                    params={'tag': 'auto_trade'}
                )
            else:
                print("已持有空仓，无需操作")

        elif signal_data['signal'] == 'HOLD':
            print("建议观望，不执行交易")
            return

        print("订单执行成功")
        # 更新持仓信息
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")

    except Exception as e:
        print(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()

def trading_bot():
    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取K线数据
    price_data = get_btc_ohlcv()
    if not price_data:
        print("获取价格数据失败")
        return

    print(f"BTC当前价格: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析
    print("\n正在使用DeepSeek分析市场...")
    signal_data = analyze_with_deepseek(price_data)
    if not signal_data:
        print("DeepSeek分析失败")
        return

    # 3. 执行交易
    execute_trade(signal_data, price_data)

def main():
    """主函数"""
    print("=" * 60)
    print("BTC/USDT OKX自动交易机器人")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"代理设置: {PROXY_HOST}:{PROXY_PORT}")
    
    if TRADE_CONFIG['test_mode']:
        print("🔧 当前为模拟模式，不会真实下单")
    else:
        print("⚠ 实盘交易模式，请谨慎操作！")
    
    print(f"交易周期: {TRADE_CONFIG['timeframe']}")
    print(f"交易数量: {TRADE_CONFIG['amount']} BTC")
    print(f"杠杆倍数: {TRADE_CONFIG['leverage']}x")
    print("已启用K线数据分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    # 根据时间周期设置执行频率
    if TRADE_CONFIG['timeframe'] == '1h':
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")
    elif TRADE_CONFIG['timeframe'] == '15m':
        schedule.every(15).minutes.do(trading_bot)
        print("执行频率: 每15分钟一次")
    else:
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")

    print("\n" + "=" * 60)
    print("开始运行...")
    print("=" * 60)

    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback
        traceback.print_exc()

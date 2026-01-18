import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional, Tuple
import json

# ========== 配置区 (请务必填写你的API密钥) ==========
CONFIG = {
    "okx_api_base": "https://www.okx.com",
    "cmc_api_key": "YOUR_COINMARKETCAP_API_KEY",  # 用于获取社交媒体数据
    "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",  # 可选：用于监控社群
}

# ========== 1. 数据获取模块 (升级版) ==========
def fetch_top_gainers(limit: int = 30, min_gain: float = 60.0) -> List[Dict]:
    """
    获取涨幅榜，并应用60%涨幅过滤。
    返回: [{'instId': 'BTC-USDT-SWAP', 'gain': 75.5}, ...]
    """
    url = f"{CONFIG['okx_api_base']}/api/v5/market/tickers"
    params = {"instType": "SWAP"}
    
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        if resp['code'] != '0':
            print("获取行情失败:", resp['msg'])
            return []
        
        all_tickers = resp['data']
        # 提取USDT永续合约，并计算24小时涨跌幅 (假设字段为 'sod24hPx'，需根据实际API调整)
        gainers = []
        for ticker in all_tickers:
            if ticker['instId'].endswith('-USDT-SWAP'):
                try:
                    # 注意：这里需要根据OKX返回的实际字段名修改 'sod24hPx'
                    gain_pct = float(ticker.get('sod24hPx', '0'))
                    if gain_pct >= min_gain:
                        gainers.append({
                            'instId': ticker['instId'],
                            'gain': gain_pct,
                            'last': float(ticker.get('last', '0'))
                        })
                except ValueError:
                    continue
        
        # 按涨幅排序，取前 limit 名
        sorted_gainers = sorted(gainers, key=lambda x: x['gain'], reverse=True)[:limit]
        print(f"找到 {len(sorted_gainers)} 个涨幅 ≥ {min_gain}% 的币种")
        return sorted_gainers
        
    except Exception as e:
        print(f"获取涨幅榜异常: {e}")
        return []

# ========== 2. 市场情绪量化模块 ==========
def quantify_market_sentiment(instId: str) -> Dict:
    """
    量化市场情绪指标。
    返回包含社交热度分数和多空比的数据字典。
    """
    sentiment = {'social_score': 0.5, 'long_short_ratio': 1.0, 'sentiment': 'neutral'}
    
    # 示例：通过CoinMarketCap获取社交媒体数据 (需配置API)
    coin_symbol = instId.split('-')[0]  # 简单提取币种符号，如 'BTC'
    try:
        # 注意：CoinMarketCap API 端点可能需要调整
        url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/info"
        headers = {'X-CMC_PRO_API_KEY': CONFIG['cmc_api_key']}
        params = {'symbol': coin_symbol}
        resp = requests.get(url, headers=headers, params=params, timeout=10).json()
        
        # 简化处理：假设返回数据中有社交媒体相关指标
        # 实际应根据API文档解析，此处为示例逻辑
        if 'data' in resp and coin_symbol in resp['data']:
            data = resp['data'][coin_symbol][0]
            # 示例：结合Twitter粉丝增长、Reddit帖子活跃度等（需根据实际API字段调整）
            twitter_followers = data.get('twitter_followers', 0)
            reddit_active_users = data.get('reddit_active_users', 0)
            # 简单计算一个0-1的分数
            sentiment['social_score'] = min(1.0, (np.log1p(twitter_followers) / 15 + np.log1p(reddit_active_users) / 10) / 2)
    except Exception as e:
        print(f"获取 {instId} 社交数据失败: {e}")
    
    # 获取多空人数比 (通过OKX未平仓合约估算)
    try:
        oi_url = f"{CONFIG['okx_api_base']}/api/v5/public/open-interest"
        oi_params = {"instId": instId}
        oi_resp = requests.get(oi_url, params=oi_params, timeout=5).json()
        if oi_resp['code'] == '0':
            # 注意：这是一个简化的估算，OKX未直接提供多空人数比
            oi_data = oi_resp['data'][0]
            long_oi = float(oi_data.get('longOpenInterest', 0))
            short_oi = float(oi_data.get('shortOpenInterest', 0))
            if short_oi > 0:
                sentiment['long_short_ratio'] = long_oi / short_oi
    except Exception as e:
        print(f"获取 {instId} 多空数据失败: {e}")
    
    # 判断情绪
    if sentiment['social_score'] > 0.7 and sentiment['long_short_ratio'] > 1.5:
        sentiment['sentiment'] = 'extremely_greedy'
    elif sentiment['social_score'] > 0.6:
        sentiment['sentiment'] = 'greedy'
    elif sentiment['social_score'] < 0.4 and sentiment['long_short_ratio'] < 0.8:
        sentiment['sentiment'] = 'fearful'
    
    return sentiment

# ========== 3. 背离检测模块 ==========
def detect_divergence(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    检测价格与RSI、成交量的顶背离。
    返回包含背离类型和强度的字典。
    """
    divergence = {'type': 'none', 'strength': 0}
    
    if len(df) < lookback:
        return divergence
    
    # 提取最近 lookback 期内的价格高点和RSI
    price_highs = df['high'].iloc[-lookback:].values
    rsi_values = df['rsi'].iloc[-lookback:].values
    volume_values = df['volume'].iloc[-lookback:].values
    
    # 寻找价格高点 (简化：寻找局部峰值)
    from scipy.signal import find_peaks
    price_peak_indices, _ = find_peaks(price_highs, distance=10, prominence=0.01)
    if len(price_peak_indices) < 2:
        return divergence
    
    # 取最近两个价格峰值
    recent_peaks = sorted(price_peak_indices[-2:])
    peak1_idx, peak2_idx = recent_peaks[0], recent_peaks[1]
    
    price1, price2 = price_highs[peak1_idx], price_highs[peak2_idx]
    rsi1, rsi2 = rsi_values[peak1_idx], rsi_values[peak2_idx]
    vol1, vol2 = volume_values[peak1_idx], volume_values[peak2_idx]
    
    # 判断顶背离条件
    is_price_higher = price2 > price1
    is_rsi_lower = rsi2 < rsi1
    is_volume_lower = vol2 < vol1
    
    if is_price_higher and is_rsi_lower:
        divergence['type'] = 'price_rsi_divergence'
        divergence['strength'] += 30
        print(f"    检测到价格-RSI顶背离: 价格 {price1:.4f}->{price2:.4f}, RSI {rsi1:.1f}->{rsi2:.1f}")
    
    if is_price_higher and is_volume_lower:
        divergence['type'] = 'price_volume_divergence' if divergence['type'] == 'none' else 'multiple_divergence'
        divergence['strength'] += 20
        print(f"    检测到价格-成交量顶背离: 价格 {price1:.4f}->{price2:.4f}, 成交量 {vol1:.0f}->{vol2:.0f}")
    
    return divergence

# ========== 4. 综合分析与决策模块 (整合所有条件) ==========
def comprehensive_short_analysis(instId: str) -> Optional[Dict]:
    """
    执行完整的做空分析流程。
    """
    print(f"\n🔍 深度分析 {instId}")
    
    # 4.1 应用基础过滤 (历史价格位置、币龄、资金费率)
    passed, filter_reason = apply_filters(instId)  # 复用之前定义的过滤函数
    if not passed:
        print(f"   ⛔ 基础过滤未通过: {filter_reason}")
        return None
    
    # 4.2 获取K线数据并计算技术指标
    klines = fetch_klines(instId, bar='1H', limit=200)
    if not klines:
        return None
    df = process_klines_to_df(klines)  # 将K线转为DataFrame并计算RSI、布林带等
    
    # 4.3 检测背离
    divergence = detect_divergence(df)
    
    # 4.4 量化市场情绪
    sentiment = quantify_market_sentiment(instId)
    
    # 4.5 综合信号评分 (加权计算)
    total_score = 0
    reasons = []
    
    # 技术信号分 (40%)
    tech_score, tech_reasons = calculate_technical_score(df)
    total_score += tech_score * 0.4
    reasons.extend(tech_reasons)
    
    # 价格位置分 (20%): 历史高位加分
    price_position = calculate_price_position(instId)  # 复用之前函数
    if price_position >= 0.9:
        total_score += 20
        reasons.append(f"历史高位({price_position:.1%})")
    
    # 背离信号分 (20%)
    total_score += divergence['strength'] * 0.2
    if divergence['type'] != 'none':
        reasons.append(f"{divergence['type']}")
    
    # 情绪信号分 (20%): 市场极度贪婪时加分
    if sentiment['sentiment'] in ['greedy', 'extremely_greedy']:
        total_score += 20
        reasons.append(f"市场情绪: {sentiment['sentiment']}")
    elif sentiment['sentiment'] == 'fearful':
        total_score -= 10  # 情绪恐惧时，做空需谨慎
    
    # 4.6 最终决策
    if total_score < 70:  # 综合置信度阈值
        print(f"   ⚠️  综合置信度不足: {total_score:.1f}分")
        return None
    
    # 计算关键价位
    entry, sl, tp1, tp2 = calculate_trading_prices(df, divergence)
    
    # 4.7 整合分析报告
    return {
        'instId': instId,
        'composite_score': round(total_score, 1),
        'reasons': reasons,
        'divergence': divergence['type'],
        'sentiment': sentiment['sentiment'],
        'entry': entry,
        'stop_loss': sl,
        'take_profit_1': tp1,
        'take_profit_2': tp2,
        'risk_reward_ratio': '1:2',
        'time': datetime.now().isoformat()
    }

# ========== 主执行函数 ==========
def main():
    """主执行流程：获取榜单 -> 深度分析 -> 输出报告"""
    print("🚀 启动高级做空信号扫描系统")
    print("=" * 60)
    
    # 1. 获取高涨幅币种
    gainers = fetch_top_gainers(limit=30, min_gain=60.0)
    if not gainers:
        print("未找到符合条件的币种")
        return
    
    # 2. 对每个币种进行深度分析
    signals = []
    for i, coin in enumerate(gainers[:5]):  # 示例：先分析前5个
        print(f"\n[{i+1}/{min(5, len(gainers))}] 分析 {coin['instId']} (涨幅: {coin['gain']:.1f}%)")
        signal = comprehensive_short_analysis(coin['instId'])
        if signal:
            signals.append(signal)
        time.sleep(1)  # 控制请求频率
    
    # 3. 输出结果
    print("\n" + "=" * 60)
    print(f"📊 分析完成！发现 {len(signals)} 个高置信度做空机会:")
    
    for sig in sorted(signals, key=lambda x: x['composite_score'], reverse=True):
        print(f"\n✅ 币种: {sig['instId']}")
        print(f"   综合评分: {sig['composite_score']} | 情绪: {sig['sentiment']} | 背离: {sig['divergence']}")
        print(f"   理由: {', '.join(sig['reasons'])}")
        print(f"   操作: 做空 @ {sig['entry']:.4f}")
        print(f"   风控: 止损 {sig['stop_loss']:.4f} | 止盈 {sig['take_profit_1']:.4f}, {sig['take_profit_2']:.4f}")

if __name__ == "__main__":
    main()
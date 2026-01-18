# verify_api.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY')
print(f"API密钥: {api_key}")
print(f"密钥长度: {len(api_key)}")

# 测试API调用
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "你好，请回复'测试成功'"}],
    "stream": False,
    "max_tokens": 50
}

try:
    response = requests.post(
        'https://api.deepseek.com/chat/completions',
        headers=headers,
        json=data,
        timeout=10
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API密钥有效！")
        print(f"回复: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ API调用失败")
        print(f"错误信息: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ 网络请求错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")

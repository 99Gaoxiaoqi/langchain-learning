"""
第六课：工具 (Tools)
学习目标：
1. 理解工具的概念
2. 创建自定义工具
3. 使用 @tool 装饰器
4. 工具与 LLM 的结合
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import math

load_dotenv()
# 定义工具
@tool
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    try:
        # 安全的数学计算
        allowed_names = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def search_weather(city: str) -> str:
    """
    查询城市天气（模拟）
    Args:
        city: 城市名称
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，温度 25°C",
        "上海": "多云，温度 28°C",
        "广州": "小雨，温度 30°C",
    }
    return weather_data.get(city, f"{city}的天气数据暂不可用")

def tools_demo():
    """工具使用示例"""
    print("=== 工具信息 ===")
    print(f"工具名称: {get_current_time.name}")
    print(f"工具描述: {get_current_time.description}")
    
    print("\n=== 直接调用工具 ===")
    print(f"当前时间: {get_current_time.invoke({})}")
    print(f"计算 2+3*4: {calculate.invoke({'expression': '2+3*4'})}")
    print(f"北京天气: {search_weather.invoke({'city': '北京'})}")

def tool_calling_demo():
    """工具调用示例（需要模型支持）"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # 绑定工具到 LLM
    tools = [get_current_time, calculate, search_weather]
    llm_with_tools = llm.bind_tools(tools)
    
    # 让 LLM 决定是否调用工具
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，可以使用工具来帮助用户。"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm_with_tools
    
    # 测试
    response = chain.invoke({"question": "现在几点了？"})
    print("LLM 响应:", response)
    
    if response.tool_calls:
        print("工具调用:", response.tool_calls)
        # 执行工具
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_current_time":
                result = get_current_time.invoke({})
                print(f"工具结果: {result}")

if __name__ == "__main__":
    tools_demo()
    print("\n=== 工具调用示例 ===")
    tool_calling_demo()

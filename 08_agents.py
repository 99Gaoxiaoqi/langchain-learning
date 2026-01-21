"""
第八课：Agent 智能体
学习目标：
1. 理解 Agent 的概念
2. ReAct 模式
3. 创建简单的 Agent
4. Agent 的工具使用
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import math

load_dotenv()

# 定义工具
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。
    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"
    """
    try:
        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, 
                   "tan": math.tan, "pi": math.pi, "e": math.e, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库（模拟）。
    Args:
        query: 搜索查询
    """
    knowledge = {
        "python": "Python 是一种高级编程语言，以简洁易读著称。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架。",
        "ai": "人工智能是让机器模拟人类智能的技术。",
    }
    
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "未找到相关信息"

def simple_react_agent():
    """
    简单的 ReAct Agent 实现
    ReAct = Reasoning + Acting
    """
    llm = ChatTongyi(model="qwen-turbo")
    tools = [calculator, search_knowledge]
    
    # ReAct 提示模板
    react_prompt = """你是一个智能助手，可以使用以下工具来帮助用户：

可用工具：
1. calculator(expression) - 计算数学表达式
2. search_knowledge(query) - 搜索知识库

请按照以下格式思考和行动：

思考：分析用户的问题，决定是否需要使用工具
行动：如果需要工具，说明要使用哪个工具和参数
观察：工具返回的结果
... (可以重复思考-行动-观察)
最终答案：给用户的最终回答

用户问题：{question}

请开始："""

    prompt = ChatPromptTemplate.from_template(react_prompt)
    
    def run_agent(question: str):
        # 第一步：让 LLM 思考
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})
        print("Agent 思考过程:")
        print(response)
        
        # 简单的工具调用解析（实际应用中需要更复杂的解析）
        if "calculator" in response.lower():
            # 提取表达式（简化处理）
            import re
            match = re.search(r'calculator\(["\']?([^"\']+)["\']?\)', response)
            if match:
                expr = match.group(1)
                result = calculator.invoke({"expression": expr})
                print(f"\n工具调用结果: {result}")
        
        return response
    
    # 测试
    print("=== 测试 Agent ===\n")
    run_agent("计算 25 的平方根是多少？")

def tool_agent_demo():
    """使用 LangChain 的工具调用功能"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # 绑定工具
    tools = [calculator, search_knowledge]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，根据需要使用工具来回答问题。"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm_with_tools
    
    # 测试
    questions = [
        "计算 15 * 8 + 32",
        "什么是 Python？",
        "你好，今天天气怎么样？"
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        response = chain.invoke({"input": q})
        
        if response.tool_calls:
            print("需要调用工具:")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}: {tc['args']}")
                # 执行工具
                if tc['name'] == 'calculator':
                    result = calculator.invoke(tc['args'])
                elif tc['name'] == 'search_knowledge':
                    result = search_knowledge.invoke(tc['args'])
                print(f"  结果: {result}")
        else:
            print(f"直接回答: {response.content}")

if __name__ == "__main__":
    print("=== 简单 ReAct Agent ===")
    simple_react_agent()
    print("\n" + "="*50)
    print("\n=== 工具调用 Agent ===")
    tool_agent_demo()

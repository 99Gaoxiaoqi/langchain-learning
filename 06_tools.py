"""
第六课：工具 (Tools) - 让 LLM 与外部世界交互

学习目标：
1. 理解工具的概念和作用
2. 掌握多种工具定义方式
3. 学会工具调用的完整流程
4. 了解企业级最佳实践

核心概念：
─────────────────────────────────────────────────────────────────────────────
工具 (Tool) 是 LLM 与外部世界交互的桥梁。LLM 本身只能生成文本，
通过工具可以让 LLM：
- 获取实时信息（天气、股票、搜索）
- 执行计算和数据处理
- 操作外部系统（数据库、API、文件）
- 与其他服务集成

工具调用流程：
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 定义工具 → 2. 绑定到 LLM → 3. LLM 决定调用 → 4. 执行工具 → 5. 返回结果  │
└─────────────────────────────────────────────────────────────────────────────┘

三种工具调用方式对比：
┌────────────────────┬─────────────────────┬─────────────────────────────────┐
│ 方式               │ 适用场景            │ 特点                            │
├────────────────────┼─────────────────────┼─────────────────────────────────┤
│ 手动循环           │ 简单场景/学习理解   │ 完全控制，代码量多              │
│ create_tool_       │ 标准 Agent 场景     │ 官方推荐，自动处理循环          │
│ calling_agent      │                     │                                 │
│ LangGraph          │ 复杂/生产环境       │ 最灵活，支持状态管理和人工介入  │
└────────────────────┴─────────────────────┴─────────────────────────────────┘

企业级推荐：
- 简单工具调用：手动循环（完全控制）
- 标准 Agent：create_tool_calling_agent + AgentExecutor
- 生产环境：LangGraph（第8课详细讲解）
─────────────────────────────────────────────────────────────────────────────
"""
import asyncio
import math
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from llm_factory import get_llm

load_dotenv()
llm = get_llm()


# ============================================================
# 第一部分：工具定义方式
# ============================================================

# 方式1：@tool 装饰器（最常用）
@tool
def get_current_time() -> str:
    """获取当前时间，返回格式化的时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"、"sqrt(16)"、"sin(3.14)"
    """
    try:
        allowed = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, 
                   "tan": math.tan, "pi": math.pi, "e": math.e, "abs": abs}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_weather(city: str) -> str:
    """
    查询城市天气
    
    Args:
        city: 城市名称，如 "北京"、"上海"
    """
    weather_data = {
        "北京": "晴天，温度 25°C，湿度 40%",
        "上海": "多云，温度 28°C，湿度 65%",
        "广州": "小雨，温度 30°C，湿度 80%",
        "深圳": "阴天，温度 29°C，湿度 70%",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


# 方式2：使用 Pydantic 定义参数（可选，用于参数验证）
# 说明：大多数情况下不需要，只有需要验证规则时才用
class SearchInput(BaseModel):
    """Pydantic 模型可以添加验证规则"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=100)  # ge=1, le=100 表示值必须在 1-100 之间


@tool(args_schema=SearchInput)
def validated_search(query: str, max_results: int = 5) -> str:
    """带参数验证的搜索（如果 max_results 不在 1-100 之间会报错）"""
    return f"搜索 '{query}'，返回 {max_results} 条结果"


# 💡 推荐：大多数情况下直接这样写就够了，不需要 args_schema
@tool
def web_search(query: str, max_results: int = 5, language: str = "zh") -> str:
    """
    模拟网络搜索
    
    Args:
        query: 搜索关键词
        max_results: 最大返回结果数
        language: 语言，zh中文/en英文
    """
    return f"搜索 '{query}'，语言={language}，返回 {max_results} 条结果"


# 方式3：StructuredTool.from_function（更多控制）
def send_email_func(to: str, subject: str, body: str) -> str:
    """发送邮件的实际实现"""
    return f"邮件已发送到 {to}，主题：{subject}"


send_email = StructuredTool.from_function(
    func=send_email_func,
    name="send_email",
    description="发送电子邮件",
    return_direct=False,  # 是否直接返回结果给用户
)


def demo_tool_definitions():
    """1.1 工具定义方式演示"""
    print("=" * 60)
    print("1.1 工具定义方式")
    print("=" * 60)
    
    tools = [get_current_time, calculate, search_weather, web_search, send_email]
    
    for t in tools:
        print(f"\n📦 {t.name}")
        print(f"   描述: {t.description[:50]}...")
        print(f"   参数: {t.args}")
    print()


def demo_direct_invoke():
    """1.2 直接调用工具"""
    print("=" * 60)
    print("1.2 直接调用工具（不经过 LLM）")
    print("=" * 60)
    
    # 直接调用
    print(f"当前时间: {get_current_time.invoke({})}")
    print(f"计算 sqrt(16) + 2*3: {calculate.invoke({'expression': 'sqrt(16) + 2*3'})}")
    print(f"北京天气: {search_weather.invoke({'city': '北京'})}")
    print(f"网络搜索: {web_search.invoke({'query': 'LangChain', 'max_results': 3})}")
    print()


# ============================================================
# 第二部分：工具绑定与调用
# ============================================================

def demo_bind_tools():
    """2.1 将工具绑定到 LLM"""
    print("=" * 60)
    print("2.1 将工具绑定到 LLM (bind_tools)")
    print("=" * 60)
    
    tools = [get_current_time, calculate, search_weather]
    
    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # LLM 会根据问题决定是否调用工具
    response = llm_with_tools.invoke("现在几点了？")
    
    print(f"LLM 响应类型: {type(response).__name__}")
    print(f"内容: {response.content}")
    print(f"工具调用: {response.tool_calls}")
    
    if response.tool_calls:
        print("\n📞 LLM 决定调用工具:")
        for tc in response.tool_calls:
            print(f"   工具: {tc['name']}")
            print(f"   参数: {tc['args']}")
            print(f"   ID: {tc['id']}")
    print()


def demo_tool_choice():
    """2.2 控制工具选择"""
    print("=" * 60)
    print("2.2 控制工具选择 (tool_choice)")
    print("=" * 60)
    
    tools = [get_current_time, calculate, search_weather]
    
    # 强制使用特定工具
    print("--- 强制使用 calculate 工具 ---")
    llm_forced = llm.bind_tools(tools, tool_choice="calculate")
    response = llm_forced.invoke("你好")  # 即使问候也会调用计算工具
    print(f"工具调用: {response.tool_calls}")
    
    # 禁止并行调用（如果模型支持）
    print("\n--- 禁止并行工具调用 ---")
    try:
        llm_no_parallel = llm.bind_tools(tools, parallel_tool_calls=False)
        response = llm_no_parallel.invoke("北京和上海的天气怎么样？")
        print(f"工具调用数量: {len(response.tool_calls)}")
    except Exception as e:
        print(f"当前模型可能不支持此参数: {e}")
    print()


# ============================================================
# 第三部分：手动工具调用循环（完全控制）
# ============================================================

def demo_manual_tool_loop():
    """3.1 手动工具调用循环 - 完全控制流程"""
    print("=" * 60)
    print("3.1 手动工具调用循环（企业级推荐方式之一）")
    print("=" * 60)
    
    tools = [get_current_time, calculate, search_weather]
    tools_map = {t.name: t for t in tools}
    
    llm_with_tools = llm.bind_tools(tools)
    
    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以使用工具来帮助用户。"},
        {"role": "user", "content": "现在几点了？北京天气怎么样？"}
    ]
    
    print(f"用户: {messages[-1]['content']}")
    
    # 第一次调用：LLM 决定调用哪些工具
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)
    
    print(f"\nLLM 决定调用 {len(ai_response.tool_calls)} 个工具:")
    
    # 执行工具调用
    if ai_response.tool_calls:
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"  - 执行 {tool_name}({tool_args})")
            
            # 执行工具
            tool_result = tools_map[tool_name].invoke(tool_args)
            print(f"    结果: {tool_result}")
            
            # 将结果作为 ToolMessage 添加到消息列表
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_id
            ))
        
        # 第二次调用：LLM 根据工具结果生成最终回复
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终回复: {final_response.content}")
    else:
        print(f"LLM 直接回复: {ai_response.content}")
    print()


async def demo_parallel_tool_execution():
    """3.2 并行执行工具调用"""
    print("=" * 60)
    print("3.2 并行执行工具调用（提高效率）")
    print("=" * 60)
    
    tools = [get_current_time, calculate, search_weather]
    tools_map = {t.name: t for t in tools}
    
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [HumanMessage(content="北京、上海、广州的天气分别怎么样？")]
    ai_response = await llm_with_tools.ainvoke(messages)
    
    if ai_response.tool_calls:
        print(f"需要执行 {len(ai_response.tool_calls)} 个工具调用")
        
        # 并行执行所有工具调用
        async def execute_tool(tc):
            tool = tools_map[tc["name"]]
            result = tool.invoke(tc["args"])  # 工具本身可能不支持异步
            return ToolMessage(content=str(result), tool_call_id=tc["id"])
        
        # 并发执行
        tool_messages = await asyncio.gather(
            *[execute_tool(tc) for tc in ai_response.tool_calls]
        )
        
        for tm in tool_messages:
            print(f"  工具结果: {tm.content}")
        
        # 获取最终回复
        messages.append(ai_response)
        messages.extend(tool_messages)
        final = await llm_with_tools.ainvoke(messages)
        print(f"\n最终回复: {final.content}")
    print()


# ============================================================
# 第四部分：使用 Agent（自动处理工具循环）
# ============================================================

def demo_tool_calling_agent():
    """4.1 使用 create_tool_calling_agent（官方推荐）"""
    print("=" * 60)
    print("4.1 create_tool_calling_agent（官方推荐方式）")
    print("=" * 60)
    
    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        
        tools = [get_current_time, calculate, search_weather]
        
        # 创建 Agent 提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手，可以使用工具来帮助用户。"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # 创建 AgentExecutor（负责执行循环）
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # 显示执行过程
            max_iterations=5,  # 最大迭代次数
            handle_parsing_errors=True,  # 处理解析错误
        )
        
        # 执行
        result = agent_executor.invoke({
            "input": "计算 sqrt(144) + 10，然后告诉我北京的天气"
        })
        
        print(f"\n最终结果: {result['output']}")
        
    except ImportError:
        print("需要安装 langchain: pip install langchain")
    print()


def demo_agent_with_memory():
    """4.2 带记忆的 Agent"""
    print("=" * 60)
    print("4.2 带记忆的 Agent")
    print("=" * 60)
    
    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory
        
        tools = [get_current_time, calculate, search_weather]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        
        # 添加记忆
        store = {}
        def get_history(session_id):
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]
        
        agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        config = {"configurable": {"session_id": "demo"}}
        
        # 多轮对话
        print("用户: 我叫小明")
        r1 = agent_with_memory.invoke({"input": "我叫小明"}, config=config)
        print(f"AI: {r1['output']}")
        
        print("\n用户: 现在几点了？")
        r2 = agent_with_memory.invoke({"input": "现在几点了？"}, config=config)
        print(f"AI: {r2['output']}")
        
        print("\n用户: 你还记得我叫什么吗？")
        r3 = agent_with_memory.invoke({"input": "你还记得我叫什么吗？"}, config=config)
        print(f"AI: {r3['output']}")
        
    except ImportError:
        print("需要安装 langchain")
    print()


# ============================================================
# 第五部分：高级工具技巧
# ============================================================

def demo_tool_error_handling():
    """5.1 工具错误处理"""
    print("=" * 60)
    print("5.1 工具错误处理")
    print("=" * 60)
    
    @tool
    def risky_tool(value: int) -> str:
        """一个可能出错的工具"""
        if value < 0:
            raise ValueError("值不能为负数")
        return f"处理结果: {value * 2}"
    
    # 方式1：在工具内部处理错误
    @tool
    def safe_tool(value: int) -> str:
        """带错误处理的工具"""
        try:
            if value < 0:
                return "错误：值不能为负数，请提供正数"
            return f"处理结果: {value * 2}"
        except Exception as e:
            return f"处理失败: {str(e)}"
    
    print("安全工具调用 (value=5):", safe_tool.invoke({"value": 5}))
    print("安全工具调用 (value=-1):", safe_tool.invoke({"value": -1}))
    print("\n💡 建议：在工具内部处理错误，返回友好的错误信息")
    print()


def demo_tool_with_context():
    """5.2 工具访问上下文"""
    print("=" * 60)
    print("5.2 工具访问运行时配置")
    print("=" * 60)
    
    from langchain_core.runnables import RunnableConfig
    
    @tool
    def get_user_info(
        query: str,
        config: RunnableConfig  # 自动注入配置
    ) -> str:
        """获取用户信息，可以访问运行时配置"""
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        return f"用户 {user_id} 查询: {query}"
    
    # 调用时传入配置
    result = get_user_info.invoke(
        {"query": "我的订单"},
        config={"configurable": {"user_id": "user_123"}}
    )
    print(f"结果: {result}")
    print("\n💡 通过 RunnableConfig 可以传递用户ID、会话ID等上下文信息")
    print()


def demo_async_tool():
    """5.3 异步工具"""
    print("=" * 60)
    print("5.3 异步工具定义")
    print("=" * 60)
    
    @tool
    async def async_search(query: str) -> str:
        """异步搜索工具"""
        await asyncio.sleep(0.1)  # 模拟异步操作
        return f"异步搜索 '{query}' 的结果..."
    
    # 异步调用
    async def run():
        result = await async_search.ainvoke({"query": "LangChain"})
        print(f"异步结果: {result}")
    
    asyncio.run(run())
    print("\n💡 对于 I/O 密集型操作（API调用、数据库查询），使用异步工具可提高性能")
    print()


# ============================================================
# 第六部分：企业级最佳实践
# ============================================================

def print_best_practices():
    """打印企业级最佳实践"""
    print("=" * 60)
    print("📊 企业级最佳实践")
    print("=" * 60)
    print("""
    工具定义最佳实践：
    ─────────────────────────────────────────────────────────────
    1. 清晰的描述：工具描述要准确，LLM 依赖描述来决定是否调用
    2. 参数验证：使用 Pydantic 定义复杂参数，自动验证
    3. 错误处理：在工具内部处理错误，返回友好信息
    4. 幂等性：工具应该是幂等的，多次调用结果一致
    5. 超时控制：对外部 API 调用设置超时
    
    工具调用方式选择：
    ─────────────────────────────────────────────────────────────
    ┌─────────────────────┬─────────────────────────────────────┐
    │ 场景                │ 推荐方式                            │
    ├─────────────────────┼─────────────────────────────────────┤
    │ 简单单次调用        │ 手动循环（完全控制）                │
    │ 标准多轮 Agent      │ create_tool_calling_agent           │
    │ 需要人工审批        │ LangGraph（interrupt_before）       │
    │ 复杂状态管理        │ LangGraph                           │
    │ 生产环境            │ LangGraph（推荐）                   │
    └─────────────────────┴─────────────────────────────────────┘
    
    安全注意事项：
    ─────────────────────────────────────────────────────────────
    1. 不要在工具中执行任意代码（如 eval 用户输入）
    2. 对敏感操作（删除、支付）添加确认机制
    3. 限制工具的权限范围
    4. 记录工具调用日志用于审计
    5. 设置调用频率限制
    """)


# ============================================================
# 主函数
# ============================================================

async def run_async_demos():
    """运行异步示例"""
    await demo_parallel_tool_execution()


def main():
    print("\n🔧 第六课：工具 (Tools) - 让 LLM 与外部世界交互\n")
    
    print("\n" + "=" * 60)
    print("📚 第一部分：工具定义方式")
    print("=" * 60)
    demo_tool_definitions()
    demo_direct_invoke()
    
    print("\n" + "=" * 60)
    print("📚 第二部分：工具绑定与调用")
    print("=" * 60)
    demo_bind_tools()
    demo_tool_choice()
    
    print("\n" + "=" * 60)
    print("📚 第三部分：手动工具调用循环")
    print("=" * 60)
    demo_manual_tool_loop()
    asyncio.run(run_async_demos())
    
    print("\n" + "=" * 60)
    print("📚 第四部分：使用 Agent")
    print("=" * 60)
    demo_tool_calling_agent()
    demo_agent_with_memory()
    
    print("\n" + "=" * 60)
    print("📚 第五部分：高级工具技巧")
    print("=" * 60)
    demo_tool_error_handling()
    demo_tool_with_context()
    demo_async_tool()
    
    print_best_practices()
    
    print("\n" + "=" * 60)
    print("📌 第六课总结")
    print("=" * 60)
    print("""
    工具定义方式              适用场景
    ─────────────────────────────────────────────────────────
    @tool 装饰器              简单工具（最常用）
    @tool + Pydantic          复杂参数验证
    StructuredTool            需要更多控制
    BaseTool 子类             完全自定义
    
    工具调用流程
    ─────────────────────────────────────────────────────────
    1. 定义工具（@tool 装饰器）
    2. 绑定工具（llm.bind_tools()）
    3. LLM 返回 tool_calls
    4. 执行工具，获取结果
    5. 将 ToolMessage 传回 LLM
    6. LLM 生成最终回复
    
    企业级推荐
    ─────────────────────────────────────────────────────────
    - 简单场景：手动循环（完全控制）
    - 标准 Agent：create_tool_calling_agent
    - 生产环境：LangGraph（第8课详解）
    """)


if __name__ == "__main__":
    main()

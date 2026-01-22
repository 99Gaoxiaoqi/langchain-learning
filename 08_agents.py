"""
第八课：Agent 智能体 - 让 AI 自主决策和行动

学习目标：
1. 理解 Agent 的核心概念和工作原理
2. 掌握多种 Agent 实现方式
3. 学会使用 LangGraph 构建生产级 Agent
4. 了解 Agent 的高级特性（记忆、人工介入等）

核心概念：
─────────────────────────────────────────────────────────────────────────────
Agent（智能体）是能够自主决策、使用工具、完成复杂任务的 AI 系统。

与普通 Chain 的区别：
┌────────────────────┬─────────────────────┬─────────────────────────────────┐
│ 特性               │ Chain               │ Agent                           │
├────────────────────┼─────────────────────┼─────────────────────────────────┤
│ 执行流程           │ 固定、预定义        │ 动态、根据情况决定              │
│ 工具使用           │ 按顺序调用          │ 按需选择调用                    │
│ 循环能力           │ 无                  │ 可以循环直到完成任务            │
│ 适用场景           │ 简单、确定性任务    │ 复杂、需要推理的任务            │
└────────────────────┴─────────────────────┴─────────────────────────────────┘

Agent 工作流程（ReAct 模式）：
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   用户输入 → 思考(Thought) → 行动(Action) → 观察(Observation) → 循环...    │
│                    ↑                              │                         │
│                    └──────────────────────────────┘                         │
│                                                                             │
│   直到 Agent 认为任务完成，输出最终答案                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

本课涵盖的 Agent 实现方式：
┌────────────────────┬─────────────────────┬─────────────────────────────────┐
│ 方式               │ 适用场景            │ 特点                            │
├────────────────────┼─────────────────────┼─────────────────────────────────┤
│ 手动工具循环       │ 学习理解/简单场景   │ 完全控制，代码量多              │
│ create_tool_       │ 标准 Agent 场景     │ 官方封装，快速上手              │
│ calling_agent      │                     │                                 │
│ LangGraph          │ 生产环境（推荐）    │ 最灵活，支持状态/记忆/人工介入  │
└────────────────────┴─────────────────────┴─────────────────────────────────┘
─────────────────────────────────────────────────────────────────────────────
"""
import asyncio
import math
from datetime import datetime
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_factory import get_llm

load_dotenv()
llm = get_llm()


# ============================================================
# 工具定义（供所有示例使用）
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"、"sqrt(16)"、"pow(2, 10)"
    """
    try:
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "pi": math.pi, "e": math.e,
            "pow": pow, "abs": abs, "log": math.log
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def search_weather(city: str) -> str:
    """
    查询城市天气
    
    Args:
        city: 城市名称，如 "北京"、"上海"
    """
    weather_data = {
        "北京": "晴天，温度 -2°C，空气质量良好",
        "上海": "多云，温度 8°C，湿度 65%",
        "广州": "小雨，温度 15°C，湿度 80%",
        "深圳": "阴天，温度 18°C，湿度 70%",
        "杭州": "晴天，温度 5°C，空气质量优",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库
    
    Args:
        query: 搜索关键词
    """
    knowledge = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的开源框架，提供了丰富的组件和工具。",
        "langgraph": "LangGraph 是 LangChain 团队开发的图状态机框架，用于构建复杂的 Agent 工作流。",
        "agent": "Agent（智能体）是能够自主决策、使用工具、完成复杂任务的 AI 系统。",
        "rag": "RAG（检索增强生成）通过检索外部知识来增强 LLM 的回答能力。",
        "python": "Python 是一种高级编程语言，以简洁易读著称，广泛用于 AI 和数据科学。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到关于 '{query}' 的信息"


# 工具列表
TOOLS = [calculator, get_current_time, search_weather, search_knowledge]
TOOLS_MAP = {t.name: t for t in TOOLS}


# ============================================================
# 第一部分：手动工具调用循环（理解原理）
# ============================================================

def demo_manual_agent_loop():
    """1.1 手动实现 Agent 循环 - 理解 Agent 工作原理"""
    print("=" * 60)
    print("1.1 手动实现 Agent 循环（理解原理）")
    print("=" * 60)
    
    llm_with_tools = llm.bind_tools(TOOLS)
    
    def run_agent(user_input: str, max_iterations: int = 5):
        """运行 Agent 直到完成任务或达到最大迭代次数"""
        
        messages = [
            SystemMessage(content="你是一个智能助手，可以使用工具来帮助用户。请根据需要调用工具，然后给出最终答案。"),
            HumanMessage(content=user_input)
        ]
        
        print(f"\n用户: {user_input}")
        print("-" * 40)
        
        for i in range(max_iterations):
            # 1. 调用 LLM
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # 2. 检查是否有工具调用
            if not response.tool_calls:
                # 没有工具调用，Agent 完成任务
                print(f"\n最终答案: {response.content}")
                return response.content
            
            # 3. 执行工具调用
            print(f"\n[迭代 {i+1}] Agent 决定调用工具:")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"  📞 {tool_name}({tool_args})")
                
                # 执行工具
                tool_result = TOOLS_MAP[tool_name].invoke(tool_args)
                print(f"  📋 结果: {tool_result}")
                
                # 将结果添加到消息
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_id
                ))
        
        print("\n⚠️ 达到最大迭代次数")
        return None
    
    # 测试
    run_agent("现在几点了？北京天气怎么样？")
    print()
    run_agent("计算 (25 + 75) * 2 的结果")
    print()


# ============================================================
# 第二部分：使用 create_tool_calling_agent（官方封装）
# ============================================================

def demo_tool_calling_agent():
    """2.1 使用 create_tool_calling_agent（官方推荐的简单方式）"""
    print("=" * 60)
    print("2.1 create_tool_calling_agent（官方封装）")
    print("=" * 60)
    
    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        
        # Agent 提示词模板
        # 关键：必须包含 agent_scratchpad 占位符
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，可以使用工具来帮助用户完成任务。"),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建 Agent
        agent = create_tool_calling_agent(llm, TOOLS, prompt)
        
        # 创建 AgentExecutor（负责执行循环）
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,           # 显示执行过程
            max_iterations=5,       # 最大迭代次数
            handle_parsing_errors=True,  # 处理解析错误
        )
        
        # 测试
        print("\n--- 测试 1: 多工具调用 ---")
        result = agent_executor.invoke({
            "input": "现在几点了？然后帮我计算 sqrt(144) + 10"
        })
        print(f"结果: {result['output']}")
        
        print("\n--- 测试 2: 知识查询 ---")
        result = agent_executor.invoke({
            "input": "什么是 LangGraph？"
        })
        print(f"结果: {result['output']}")
        
    except ImportError as e:
        print(f"需要安装依赖: {e}")
    print()


def demo_agent_with_memory():
    """2.2 带记忆的 Agent"""
    print("=" * 60)
    print("2.2 带记忆的 Agent（多轮对话）")
    print("=" * 60)
    
    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，可以使用工具帮助用户。记住用户告诉你的信息。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(llm, TOOLS, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)
        
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
        conversations = [
            "我叫小明，我住在北京",
            "北京今天天气怎么样？",
            "你还记得我叫什么名字吗？住在哪里？",
        ]
        
        for user_input in conversations:
            print(f"\n用户: {user_input}")
            result = agent_with_memory.invoke({"input": user_input}, config=config)
            print(f"AI: {result['output']}")
        
    except ImportError as e:
        print(f"需要安装依赖: {e}")
    print()


# ============================================================
# 第三部分：LangGraph Agent（生产级推荐）
# ============================================================
"""
LangGraph 是构建生产级 Agent 的推荐方式：
- 图状态机：清晰的状态管理和流程控制
- 持久化：支持检查点，可恢复执行
- 人工介入：支持在关键节点暂停等待人工确认
- 流式输出：支持实时流式返回
- 可视化：可以生成流程图
"""

def demo_langgraph_basic():
    """3.1 LangGraph 基础 Agent"""
    print("=" * 60)
    print("3.1 LangGraph 基础 Agent（推荐方式）")
    print("=" * 60)
    
    try:
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode
        from langgraph.graph.message import add_messages
        
        # 1. 定义状态
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
        
        # 2. 绑定工具到 LLM
        llm_with_tools = llm.bind_tools(TOOLS)
        
        # 3. 定义节点函数
        def call_model(state: AgentState):
            """调用 LLM"""
            messages = state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: AgentState) -> Literal["tools", END]:
            """决定是否继续（是否有工具调用）"""
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        # 4. 构建图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(TOOLS))
        
        # 添加边
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        # 5. 编译
        app = workflow.compile()
        
        # 6. 运行
        print("\n--- 测试 LangGraph Agent ---")
        result = app.invoke({
            "messages": [HumanMessage(content="计算 15 * 8，然后告诉我现在几点")]
        })
        
        # 打印对话过程
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"\n用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    print(f"Agent 调用工具: {[tc['name'] for tc in msg.tool_calls]}")
                elif msg.content:
                    print(f"Agent: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"工具结果: {msg.content}")
        
    except ImportError as e:
        print(f"需要安装 langgraph: pip install langgraph")
        print(f"错误: {e}")
    print()


def demo_langgraph_with_memory():
    """3.2 LangGraph Agent 带记忆（检查点）"""
    print("=" * 60)
    print("3.2 LangGraph Agent 带记忆")
    print("=" * 60)
    
    try:
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode
        from langgraph.graph.message import add_messages
        from langgraph.checkpoint.memory import MemorySaver
        
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
        
        llm_with_tools = llm.bind_tools(TOOLS)
        
        def call_model(state: AgentState):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def should_continue(state: AgentState) -> Literal["tools", END]:
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(TOOLS))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        # 添加检查点（记忆）
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        
        # 使用 thread_id 来维护对话
        config = {"configurable": {"thread_id": "conversation-1"}}
        
        print("\n--- 多轮对话测试 ---")
        
        # 第一轮
        result = app.invoke(
            {"messages": [HumanMessage(content="我叫张三，请记住")]},
            config=config
        )
        print(f"用户: 我叫张三，请记住")
        print(f"Agent: {result['messages'][-1].content}")
        
        # 第二轮（同一个 thread_id，有记忆）
        result = app.invoke(
            {"messages": [HumanMessage(content="我叫什么名字？")]},
            config=config
        )
        print(f"\n用户: 我叫什么名字？")
        print(f"Agent: {result['messages'][-1].content}")
        
        # 新的对话（不同 thread_id，无记忆）
        config2 = {"configurable": {"thread_id": "conversation-2"}}
        result = app.invoke(
            {"messages": [HumanMessage(content="我叫什么名字？")]},
            config=config2
        )
        print(f"\n[新对话] 用户: 我叫什么名字？")
        print(f"Agent: {result['messages'][-1].content}")
        
    except ImportError as e:
        print(f"需要安装 langgraph: {e}")
    print()


def demo_langgraph_streaming():
    """3.3 LangGraph 流式输出"""
    print("=" * 60)
    print("3.3 LangGraph 流式输出")
    print("=" * 60)
    
    try:
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode
        from langgraph.graph.message import add_messages
        
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
        
        llm_with_tools = llm.bind_tools(TOOLS)
        
        def call_model(state: AgentState):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def should_continue(state: AgentState) -> Literal["tools", END]:
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(TOOLS))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        app = workflow.compile()
        
        print("\n--- 流式输出测试 ---")
        print("用户: 杭州天气怎么样？")
        print("Agent: ", end="")
        
        # 流式输出
        for event in app.stream(
            {"messages": [HumanMessage(content="杭州天气怎么样？")]},
            stream_mode="values"
        ):
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(last_msg.content)
        
    except ImportError as e:
        print(f"需要安装 langgraph: {e}")
    print()


# ============================================================
# 第四部分：高级 Agent 模式
# ============================================================

def demo_react_agent_manual():
    """4.1 手动实现 ReAct Agent（思考-行动-观察）"""
    print("=" * 60)
    print("4.1 ReAct Agent（思考-行动-观察模式）")
    print("=" * 60)
    
    # ReAct 提示词
    react_prompt = """你是一个智能助手，使用 ReAct（推理+行动）模式来解决问题。

可用工具：
- calculator(expression): 计算数学表达式
- get_current_time(): 获取当前时间
- search_weather(city): 查询城市天气
- search_knowledge(query): 搜索知识库

请按照以下格式思考和行动：

思考：分析问题，决定下一步行动
行动：调用工具（如果需要）
观察：工具返回的结果
... (重复思考-行动-观察，直到得出答案)
最终答案：给用户的回答

用户问题：{question}

请开始你的推理："""
    
    prompt = ChatPromptTemplate.from_template(react_prompt)
    llm_with_tools = llm.bind_tools(TOOLS)
    
    def run_react_agent(question: str):
        print(f"\n用户: {question}")
        print("-" * 40)
        
        messages = [HumanMessage(content=prompt.format(question=question))]
        
        for i in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            if response.tool_calls:
                print(f"\n思考 → 行动: 调用 {[tc['name'] for tc in response.tool_calls]}")
                
                for tc in response.tool_calls:
                    result = TOOLS_MAP[tc["name"]].invoke(tc["args"])
                    print(f"观察: {result}")
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            else:
                print(f"\n最终答案: {response.content}")
                return
    
    run_react_agent("北京和上海哪个城市更冷？")
    print()


def demo_multi_step_agent():
    """4.2 多步骤任务 Agent"""
    print("=" * 60)
    print("4.2 多步骤任务 Agent")
    print("=" * 60)
    
    llm_with_tools = llm.bind_tools(TOOLS)
    
    def run_multi_step(task: str):
        messages = [
            SystemMessage(content="""你是一个任务规划和执行助手。
对于复杂任务，请：
1. 先分解任务为多个步骤
2. 逐步执行每个步骤
3. 汇总结果给出最终答案

可用工具：calculator, get_current_time, search_weather, search_knowledge"""),
            HumanMessage(content=task)
        ]
        
        print(f"\n任务: {task}")
        print("-" * 40)
        
        for i in range(10):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            if response.tool_calls:
                print(f"\n步骤 {i+1}: 调用工具")
                for tc in response.tool_calls:
                    print(f"  - {tc['name']}({tc['args']})")
                    result = TOOLS_MAP[tc["name"]].invoke(tc["args"])
                    print(f"    结果: {result}")
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            else:
                print(f"\n最终答案:\n{response.content}")
                return
    
    run_multi_step("帮我做以下事情：1) 查询北京天气 2) 计算 100 除以 4 3) 告诉我现在的时间")
    print()


# ============================================================
# 第五部分：Agent 最佳实践
# ============================================================

def print_best_practices():
    """打印 Agent 最佳实践"""
    print("=" * 60)
    print("📊 Agent 最佳实践")
    print("=" * 60)
    print("""
    Agent 实现方式选择：
    ─────────────────────────────────────────────────────────────
    ┌─────────────────────┬─────────────────────────────────────┐
    │ 场景                │ 推荐方式                            │
    ├─────────────────────┼─────────────────────────────────────┤
    │ 学习/理解原理       │ 手动工具循环                        │
    │ 快速原型            │ create_tool_calling_agent           │
    │ 生产环境            │ LangGraph（强烈推荐）               │
    │ 需要人工审批        │ LangGraph + interrupt               │
    │ 复杂工作流          │ LangGraph + 自定义状态              │
    └─────────────────────┴─────────────────────────────────────┘
    
    工具设计原则：
    ─────────────────────────────────────────────────────────────
    1. 清晰的描述：工具描述要准确，LLM 依赖描述来决定调用
    2. 单一职责：每个工具只做一件事
    3. 参数简单：参数越简单，LLM 越容易正确调用
    4. 错误处理：在工具内部处理错误，返回友好信息
    5. 幂等性：工具应该是幂等的，多次调用结果一致
    
    Agent 提示词设计：
    ─────────────────────────────────────────────────────────────
    1. 明确角色和能力
    2. 列出可用工具及其用途
    3. 说明何时使用工具、何时直接回答
    4. 提供输出格式指导
    
    生产环境注意事项：
    ─────────────────────────────────────────────────────────────
    1. 设置最大迭代次数，防止无限循环
    2. 添加超时控制
    3. 记录日志用于调试和审计
    4. 对敏感操作添加人工确认
    5. 使用检查点实现断点续传
    6. 监控 token 使用量和成本
    
    LangGraph 优势：
    ─────────────────────────────────────────────────────────────
    - 状态管理：清晰的状态定义和转换
    - 持久化：支持多种检查点存储（内存/Redis/Postgres）
    - 人工介入：interrupt 机制支持暂停等待确认
    - 可视化：可以生成流程图
    - 流式输出：支持实时流式返回
    - 可测试：易于单元测试和集成测试
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n🤖 第八课：Agent 智能体 - 让 AI 自主决策和行动\n")
    
    print("\n" + "=" * 60)
    print("📚 第一部分：手动工具循环（理解原理）")
    print("=" * 60)
    demo_manual_agent_loop()
    
    print("\n" + "=" * 60)
    print("📚 第二部分：create_tool_calling_agent")
    print("=" * 60)
    demo_tool_calling_agent()
    demo_agent_with_memory()
    
    print("\n" + "=" * 60)
    print("📚 第三部分：LangGraph Agent（生产推荐）")
    print("=" * 60)
    demo_langgraph_basic()
    demo_langgraph_with_memory()
    demo_langgraph_streaming()
    
    print("\n" + "=" * 60)
    print("📚 第四部分：高级 Agent 模式")
    print("=" * 60)
    demo_react_agent_manual()
    demo_multi_step_agent()
    
    print_best_practices()
    
    print("\n" + "=" * 60)
    print("📌 第八课总结")
    print("=" * 60)
    print("""
    Agent 核心概念
    ─────────────────────────────────────────────────────────────
    Agent = LLM + 工具 + 循环
    - LLM 负责决策：决定调用哪个工具、何时结束
    - 工具负责执行：与外部世界交互
    - 循环负责协调：思考 → 行动 → 观察 → 思考...
    
    三种实现方式
    ─────────────────────────────────────────────────────────────
    手动工具循环        : 完全控制，适合学习理解
    create_tool_calling : 官方封装，快速上手
    LangGraph           : 生产推荐，功能最强大
    
    LangGraph 核心组件
    ─────────────────────────────────────────────────────────────
    StateGraph          : 定义状态和图结构
    add_node            : 添加节点（处理函数）
    add_edge            : 添加边（节点连接）
    add_conditional_edges: 条件边（动态路由）
    MemorySaver         : 内存检查点（记忆）
    ToolNode            : 预置的工具执行节点
    
    生产环境推荐
    ─────────────────────────────────────────────────────────────
    使用 LangGraph 构建 Agent：
    - 清晰的状态管理
    - 支持持久化和恢复
    - 支持人工介入
    - 易于测试和调试
    """)


if __name__ == "__main__":
    main()

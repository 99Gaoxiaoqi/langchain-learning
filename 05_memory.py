"""
第五课：对话记忆 (Memory) - 让 AI 记住上下文

学习目标：
1. 理解为什么需要对话记忆
2. 掌握两种记忆管理方式的区别和用法
3. 学会根据场景选择合适的方案

核心概念：
- LLM 本身是无状态的，每次调用都是独立的
- Memory 组件让 AI 能"记住"之前的对话内容

两种记忆管理方式：
┌─────────────────────────────────────────────────────────────────────────────┐
│ 方式一：传统方式 (langchain_classic)                                         │
│ - 使用 ConversationChain + Memory 类                                        │
│ - 提供丰富的记忆类型：Buffer/Window/Summary/Token 等                         │
│ - 简单易用，但不支持流式输出，灵活性较低                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 方式二：LCEL 方式 (推荐)                                                     │
│ - 使用 RunnableWithMessageHistory + BaseChatMessageHistory                  │
│ - 只提供基础存储，窗口/摘要等需要自己实现或用 trim_messages                   │
│ - 支持流式、异步、批量，灵活性高                                             │
└─────────────────────────────────────────────────────────────────────────────┘

注意：传统 Memory 类不能直接和 LCEL 的 RunnableWithMessageHistory 结合使用！

关于 langchain_classic 包：
─────────────────────────────────────────────────────────────────────────────
langchain_classic 是 LangChain 官方提供的向后兼容包，包含被标记为 legacy 的组件。
这些组件（如 ConversationChain、各种 Memory 类）在新版 LangChain 中已被 LCEL 方式取代，
但为了兼容旧代码和快速原型开发，官方将它们迁移到了 langchain_classic 包中。

使用场景：
- 快速原型开发：开箱即用的 Memory 类型丰富
- 迁移旧项目：保持与旧代码兼容
- 学习理解：概念更直观易懂

生产环境建议：使用 LCEL 方式（RunnableWithMessageHistory），更灵活、支持流式/异步
─────────────────────────────────────────────────────────────────────────────
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, trim_messages

# 传统方式需要的类
# langchain_classic 是官方向后兼容包，包含 legacy 组件
# 这些类在新版 LangChain 中已被 LCEL 方式取代，但仍可用于快速原型开发
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain_classic.chains import ConversationChain

from llm_factory import get_llm

load_dotenv()
llm = get_llm()


# ============================================================
# 第一部分：理解为什么需要记忆
# ============================================================

def demo_without_memory():
    """1.1 没有记忆的对话 - 展示问题"""
    print("=" * 60)
    print("1.1 没有记忆的对话 - AI 会'失忆'")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template("{input}")
    chain = prompt | llm | StrOutputParser()

    print("用户: 我叫小明，今年25岁")
    response1 = chain.invoke({"input": "我叫小明，今年25岁"})
    print(f"AI: {response1}")

    print("\n用户: 你还记得我叫什么名字吗？")
    response2 = chain.invoke({"input": "你还记得我叫什么名字吗？"})
    print(f"AI: {response2}")
    print("\n💡 AI 不记得用户名字，因为每次调用都是独立的")
    print()


# ============================================================
# 第二部分：传统方式 - ConversationChain + Memory
# ============================================================

def demo_traditional_buffer():
    """2.1 传统方式 - ConversationBufferMemory（完整历史）"""
    print("=" * 60)
    print("2.1 传统方式 - ConversationBufferMemory")
    print("=" * 60)

    # ConversationBufferMemory: 存储完整对话历史
    # return_messages=True: 返回 Message 对象而不是字符串
    memory = ConversationBufferMemory(return_messages=True)
    
    # ConversationChain 自动管理 memory 的读写
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    print("用户: 我叫小明，是程序员")
    print(f"AI: {conversation.predict(input='我叫小明，是程序员')}")

    print("\n用户: 我喜欢用 Python")
    print(f"AI: {conversation.predict(input='我喜欢用 Python')}")

    print("\n用户: 你还记得我的名字和职业吗？")
    print(f"AI: {conversation.predict(input='你还记得我的名字和职业吗？')}")

    print(f"\n📝 Memory 中消息数: {len(memory.chat_memory.messages)}")
    print()


def demo_traditional_window():
    """2.2 传统方式 - ConversationBufferWindowMemory（窗口记忆）"""
    print("=" * 60)
    print("2.2 传统方式 - ConversationBufferWindowMemory")
    print("=" * 60)

    # k=2: 只保留最近 2 轮对话
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    print("第1轮 - 用户: 我叫小王")
    print(f"AI: {conversation.predict(input='我叫小王')[:50]}...")

    print("\n第2轮 - 用户: 我今年30岁")
    print(f"AI: {conversation.predict(input='我今年30岁')[:50]}...")

    print("\n第3轮 - 用户: 我在上海工作")
    print(f"AI: {conversation.predict(input='我在上海工作')[:50]}...")

    # 第1轮已被丢弃
    print("\n用户: 你还记得我的名字吗？（第1轮内容，已被丢弃）")
    print(f"AI: {conversation.predict(input='你还记得我的名字吗？')}")
    print()


def demo_traditional_summary():
    """2.3 传统方式 - ConversationSummaryMemory（摘要记忆）"""
    print("=" * 60)
    print("2.3 传统方式 - ConversationSummaryMemory")
    print("=" * 60)

    # 用 LLM 生成对话摘要，适合长对话
    memory = ConversationSummaryMemory(llm=llm, return_messages=True)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    conversations = [
        "我叫李华，是软件工程师",
        "我在杭州阿里巴巴工作",
        "我主要做后端开发，用 Java 和 Go",
    ]

    for msg in conversations:
        print(f"用户: {msg}")
        print(f"AI: {conversation.predict(input=msg)[:60]}...")
        print()

    print("📋 当前摘要:")
    print(f"  {memory.buffer[:100]}...")
    print()


# ============================================================
# 第三部分：LCEL 方式 - RunnableWithMessageHistory
# ============================================================

# 全局存储
session_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 获取对应的聊天历史"""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def demo_lcel_basic():
    """3.1 LCEL 方式 - 基础用法"""
    print("=" * 60)
    print("3.1 LCEL 方式 - RunnableWithMessageHistory")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    # RunnableWithMessageHistory 自动管理历史的读写
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": "lcel_demo"}}

    print("用户: 我叫张三，在北京做数据分析")
    print(f"AI: {chain_with_history.invoke({'input': '我叫张三，在北京做数据分析'}, config=config)}")

    print("\n用户: 总结一下你知道的关于我的信息")
    print(f"AI: {chain_with_history.invoke({'input': '总结一下你知道的关于我的信息'}, config=config)}")

    print(f"\n📝 存储的消息数: {len(session_store['lcel_demo'].messages)}")
    print()


def demo_lcel_multi_session():
    """3.2 LCEL 方式 - 多会话管理"""
    print("=" * 60)
    print("3.2 LCEL 方式 - 多会话独立历史")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是AI助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain, get_session_history,
        input_messages_key="input", history_messages_key="history"
    )

    # 用户 A
    config_a = {"configurable": {"session_id": "user_A"}}
    print("👤 用户A: 我喜欢吃火锅")
    chain_with_history.invoke({"input": "我喜欢吃火锅"}, config=config_a)

    # 用户 B
    config_b = {"configurable": {"session_id": "user_B"}}
    print("👤 用户B: 我喜欢吃寿司")
    chain_with_history.invoke({"input": "我喜欢吃寿司"}, config=config_b)

    # 各自询问
    print("\n👤 用户A: 我喜欢吃什么？")
    print(f"AI: {chain_with_history.invoke({'input': '我喜欢吃什么？'}, config=config_a)}")

    print("\n👤 用户B: 我喜欢吃什么？")
    print(f"AI: {chain_with_history.invoke({'input': '我喜欢吃什么？'}, config=config_b)}")

    print("\n💡 不同 session_id 的历史完全独立")
    print()


# ============================================================
# 第四部分：LCEL 方式实现窗口/摘要功能
# ============================================================

def demo_lcel_window():
    """4.1 LCEL 方式 - 实现窗口记忆"""
    print("=" * 60)
    print("4.1 LCEL 方式 - 用闭包实现窗口记忆")
    print("=" * 60)

    def create_windowed_history(k: int):
        """创建带窗口限制的历史获取函数"""
        store = {}
        max_messages = k * 2  # 每轮2条消息

        def get_history(session_id: str) -> InMemoryChatMessageHistory:
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            history = store[session_id]
            # 裁剪到最近 k 轮
            if len(history.messages) > max_messages:
                history.messages[:] = history.messages[-max_messages:]
            return history

        return get_history, store

    get_windowed_history, window_store = create_windowed_history(k=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是AI助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain, get_windowed_history,
        input_messages_key="input", history_messages_key="history"
    )

    config = {"configurable": {"session_id": "window_test"}}

    for i, msg in enumerate(["我叫小王", "我今年30岁", "我在上海工作"], 1):
        print(f"第{i}轮 - 用户: {msg}")
        chain_with_history.invoke({"input": msg}, config=config)
        get_windowed_history("window_test")  # 触发裁剪

    print(f"\n消息数: {len(window_store['window_test'].messages)} (最多4条)")
    print("\n用户: 你还记得我的名字吗？")
    print(f"AI: {chain_with_history.invoke({'input': '你还记得我的名字吗？'}, config=config)}")
    print()


def demo_lcel_trim_messages():
    """4.2 LCEL 方式 - 使用 trim_messages 工具"""
    print("=" * 60)
    print("4.2 LCEL 方式 - 使用 trim_messages")
    print("=" * 60)

    # trim_messages 是 LangChain 提供的消息裁剪工具
    messages = [
        HumanMessage(content="我叫小明"),
        AIMessage(content="你好小明"),
        HumanMessage(content="我25岁"),
        AIMessage(content="好的"),
        HumanMessage(content="我在北京"),
        AIMessage(content="北京不错"),
    ]

    print(f"原始消息数: {len(messages)}")

    # 保留最后4条
    trimmed = trim_messages(
        messages,
        max_tokens=4,
        token_counter=len,  # 简单用消息数量计数
        strategy="last"
    )

    print(f"裁剪后消息数: {len(trimmed)}")
    print("裁剪后内容:")
    for m in trimmed:
        print(f"  {type(m).__name__}: {m.content}")

    print("\n💡 trim_messages 可以在链中手动调用来控制历史长度")
    print()


# ============================================================
# 第五部分：两种方式对比
# ============================================================

def print_comparison():
    """打印两种方式的对比"""
    print("=" * 60)
    print("📊 两种方式对比")
    print("=" * 60)
    print("""
    ┌────────────────────┬─────────────────────┬─────────────────────┐
    │ 特性               │ 传统方式            │ LCEL 方式           │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ 核心组件           │ ConversationChain   │ RunnableWithMessage │
    │                    │ + Memory 类         │ History             │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ 记忆类型           │ Buffer/Window/      │ 只有基础存储        │
    │                    │ Summary/Token 等    │ 需自己实现高级功能  │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ 流式输出           │ ❌ 不支持           │ ✅ 支持             │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ 异步调用           │ ❌ 不支持           │ ✅ 支持             │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ 灵活性             │ 低                  │ 高                  │
    ├────────────────────┼─────────────────────┼─────────────────────┤
    │ 适用场景           │ 简单对话            │ 生产环境            │
    └────────────────────┴─────────────────────┴─────────────────────┘

    选择建议：
    - 快速原型/简单场景：传统方式，开箱即用
    - 生产环境/需要流式：LCEL 方式
    - 需要窗口/摘要：传统方式更方便，LCEL 需自己实现
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n💾 第五课：对话记忆 (Memory) - 让 AI 记住上下文\n")

    print("\n" + "=" * 60)
    print("📚 第一部分：理解为什么需要记忆")
    print("=" * 60)
    demo_without_memory()

    print("\n" + "=" * 60)
    print("📚 第二部分：传统方式 - ConversationChain + Memory")
    print("=" * 60)
    demo_traditional_buffer()
    demo_traditional_window()
    demo_traditional_summary()

    print("\n" + "=" * 60)
    print("📚 第三部分：LCEL 方式 - RunnableWithMessageHistory")
    print("=" * 60)
    demo_lcel_basic()
    demo_lcel_multi_session()

    print("\n" + "=" * 60)
    print("📚 第四部分：LCEL 实现窗口/摘要功能")
    print("=" * 60)
    demo_lcel_window()
    demo_lcel_trim_messages()

    print_comparison()

    print("\n" + "=" * 60)
    print("📌 第五课总结")
    print("=" * 60)
    print("""
    传统方式 (langchain_classic):
    ─────────────────────────────────────────────────────────
    ConversationBufferMemory        完整历史
    ConversationBufferWindowMemory  最近K轮
    ConversationSummaryMemory       LLM生成摘要
    ConversationTokenBufferMemory   按token限制
    
    LCEL 方式 (推荐):
    ─────────────────────────────────────────────────────────
    InMemoryChatMessageHistory      内存存储
    RunnableWithMessageHistory      自动管理历史
    trim_messages                   消息裁剪工具
    
    注意事项:
    ─────────────────────────────────────────────────────────
    1. 传统 Memory 类不能直接和 LCEL 结合使用
    2. LCEL 方式需要自己实现窗口/摘要逻辑
    3. 生产环境用 Redis/SQL 等持久化存储
    """)


if __name__ == "__main__":
    main()

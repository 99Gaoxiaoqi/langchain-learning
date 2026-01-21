"""
第一课：LangChain 基础 - 使用统一兼容层进行对话
学习目标：
1. 了解 LangChain 的基本概念
2. 使用统一的 LLM 工厂切换不同模型
3. 掌握 4 种消息格式
4. 掌握所有调用方式：invoke/stream/batch 及其异步版本
"""
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from llm_factory import get_llm, list_providers

# 使用统一工厂获取 LLM
llm = get_llm()


# ============================================================
# 第一部分：4 种消息格式
# ============================================================

def demo_message_formats():
    """演示 4 种消息输入格式"""
    print("=" * 50)
    print("📝 4 种消息格式演示")
    print("=" * 50)
    
    # 方式1：纯字符串（最简单）
    print("\n1️⃣ 字符串格式（最简单）")
    response = llm.invoke("用一句话介绍Python")
    print(f"   响应: {response.content}")
    
    # 方式2：元组列表（推荐，简洁易读）
    print("\n2️⃣ 元组列表格式（推荐）")
    response = llm.invoke([
        ("system", "你是一个友好的AI助手，用一句话回答"),
        ("human", "什么是Java？"),
    ])
    print(f"   响应: {response.content}")
    
    # 方式3：字典列表（OpenAI 原生格式）
    print("\n3️⃣ 字典列表格式（OpenAI兼容）")
    response = llm.invoke([
        {"role": "system", "content": "你是一个友好的AI助手，用一句话回答"},
        {"role": "user", "content": "什么是Go语言？"},
    ])
    print(f"   响应: {response.content}")
    
    # 方式4：Message 对象（完整控制）
    print("\n4️⃣ Message对象格式（完整控制）")
    response = llm.invoke([
        SystemMessage(content="你是一个友好的AI助手，用一句话回答"),
        HumanMessage(content="什么是Rust？"),
    ])
    print(f"   响应: {response.content}")
    
    print("\n" + "-" * 50)
    print("💡 推荐：简单场景用字符串/元组，多模态用Message对象")
    print()


# ============================================================
# 第二部分：6 种调用方式
# ============================================================

# 统一使用元组格式作为示例
messages = [
    ("system", "你是一个友好的AI助手，用一句话简洁回答"),
    ("human", "Python是什么？"),
]

batch_messages = [
    [("system", "用一句话回答"), ("human", "什么是Java？")],
    [("system", "用一句话回答"), ("human", "什么是Go？")],
    [("system", "用一句话回答"), ("human", "什么是Rust？")],
]


def demo_invoke():
    """1. invoke() - 同步调用"""
    print("=" * 50)
    print("1. invoke() - 同步调用")
    print("=" * 50)
    response = llm.invoke(messages)
    print(f"响应: {response.content}\n")


def demo_stream():
    """2. stream() - 同步流式"""
    print("=" * 50)
    print("2. stream() - 同步流式（观察逐字输出效果）")
    print("=" * 50)
    print("响应: ", end="")
    # 用更长的问题让流式效果更明显
    stream_messages = [
        ("system", "你是一个编程专家"),
        ("human", "用100字左右介绍Python的主要特点和应用场景"),
    ]
    for chunk in llm.stream(stream_messages):
        print(chunk.content, end="", flush=True)
    print("\n")


def demo_batch():
    """3. batch() - 同步批量"""
    print("=" * 50)
    print("3. batch() - 同步批量处理")
    print("=" * 50)
    responses = llm.batch(batch_messages)
    for i, resp in enumerate(responses):
        print(f"响应{i+1}: {resp.content}")
    print()


async def demo_ainvoke():
    """4. ainvoke() - 异步调用"""
    print("=" * 50)
    print("4. ainvoke() - 异步调用")
    print("=" * 50)
    response = await llm.ainvoke(messages)
    print(f"响应: {response.content}\n")


async def demo_astream():
    """5. astream() - 异步流式"""
    print("=" * 50)
    print("5. astream() - 异步流式（观察逐字输出效果）")
    print("=" * 50)
    print("响应: ", end="")
    stream_messages = [
        ("system", "你是一个编程专家"),
        ("human", "用100字左右介绍Java的主要特点和应用场景"),
    ]
    async for chunk in llm.astream(stream_messages):
        print(chunk.content, end="", flush=True)
    print("\n")


async def demo_abatch():
    """6. abatch() - 异步批量"""
    print("=" * 50)
    print("6. abatch() - 异步批量处理")
    print("=" * 50)
    responses = await llm.abatch(batch_messages)
    for i, resp in enumerate(responses):
        print(f"响应{i+1}: {resp.content}")
    print()


async def demo_concurrent():
    """7. 并发调用 - 企业级推荐"""
    print("=" * 50)
    print("7. 并发调用 - 企业级推荐")
    print("=" * 50)
    tasks = [
        llm.ainvoke("1+1=?"),
        llm.ainvoke("2+2=?"),
        llm.ainvoke("3+3=?"),
    ]
    results = await asyncio.gather(*tasks)
    for i, resp in enumerate(results):
        print(f"并发响应{i+1}: {resp.content}")
    print()


async def run_async_demos():
    await demo_ainvoke()
    await demo_astream()
    await demo_abatch()
    await demo_concurrent()


def main():
    print("\n🚀 LangChain 基础教程\n")
    
    # 显示支持的提供商
    list_providers()
    print()
    
    # 消息格式演示
    demo_message_formats()
    
    # 调用方式演示
    demo_invoke()
    demo_stream()
    demo_batch()
    asyncio.run(run_async_demos())
    
    print("=" * 50)
    print("📌 总结")
    print("=" * 50)
    print("""
    消息格式              适用场景
    ─────────────────────────────────────
    字符串                最简单的单轮问答
    元组列表              多角色对话（推荐）
    字典列表              OpenAI 格式兼容
    Message对象           多模态、需要元数据
    
    调用方式              适用场景
    ─────────────────────────────────────
    invoke()              简单脚本/测试
    stream()              用户实时交互
    batch()               批量数据处理
    ainvoke()             Web API 服务
    astream()             异步实时交互
    abatch()              异步批量处理
    asyncio.gather()      高并发场景
    """)


if __name__ == "__main__":
    main()

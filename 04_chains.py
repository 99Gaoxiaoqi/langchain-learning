"""
第四课：链 (Chains) - LCEL 完全指南
学习目标：
1. 理解 LCEL (LangChain Expression Language)
2. 掌握核心 Runnable 组件
3. 学会各种调用方式和数据流转
4. 实现错误处理和调试
"""
import asyncio
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    chain,
)
from llm_factory import get_llm

load_dotenv()

llm = get_llm()


# ============================================================
# 第一部分：LCEL 基础
# ============================================================

def demo_simple_chain():
    """1.1 简单链 - 管道操作符"""
    print("=" * 60)
    print("1.1 简单链 - 使用 | 管道符组合组件")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("用一句话解释什么是{concept}")
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"concept": "机器学习"})
    print(f"结果: {result}")
    print(f"\n链的输入 Schema: {chain.input_schema.model_json_schema()}")
    print()


def demo_chain_equivalents():
    """1.2 三种等价的链构建方式"""
    print("=" * 60)
    print("1.2 三种等价的链构建方式")
    print("=" * 60)
    
    from langchain_core.runnables import RunnableSequence
    
    prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")
    parser = StrOutputParser()
    
    # 方式1: 管道操作符（推荐）
    chain1 = prompt | llm | parser
    # 方式2: pipe() 方法
    chain2 = prompt.pipe(llm).pipe(parser)
    # 方式3: RunnableSequence（显式）
    chain3 = RunnableSequence(first=prompt, middle=[llm], last=parser)
    
    result = chain1.invoke({"topic": "Python"})
    print(f"结果: {result}")
    print("💡 推荐使用管道操作符 |，最简洁易读")
    print()


# ============================================================
# 第二部分：核心 Runnable 组件
# ============================================================

def demo_runnable_lambda():
    """2.1 RunnableLambda - 包装自定义函数"""
    print("=" * 60)
    print("2.1 RunnableLambda - 包装自定义函数")
    print("=" * 60)
    
    def add_emoji(text: str) -> str:
        return f"🎉 {text} 🎉"
    
    def word_count(text: str) -> dict:
        return {"text": text, "word_count": len(text)}
    
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    
    # 在链中使用自定义函数
    chain = prompt | llm | StrOutputParser() | RunnableLambda(add_emoji)
    result = chain.invoke({"topic": "LangChain"})
    print(f"带 emoji: {result}")
    
    # 链式处理
    chain2 = prompt | llm | StrOutputParser() | RunnableLambda(word_count)
    result2 = chain2.invoke({"topic": "Python"})
    print(f"带字数统计: {result2}")
    print()


def demo_chain_decorator():
    """2.2 @chain 装饰器 - 更优雅的自定义链"""
    print("=" * 60)
    print("2.2 @chain 装饰器 - 更优雅的自定义链")
    print("=" * 60)
    
    @chain
    def analyze_topic(input_dict: dict) -> str:
        """自定义链：分析主题并返回格式化结果"""
        topic = input_dict["topic"]
        
        prompt1 = ChatPromptTemplate.from_template("用一句话解释{topic}")
        explanation = (prompt1 | llm | StrOutputParser()).invoke({"topic": topic})
        
        prompt2 = ChatPromptTemplate.from_template("列举{topic}的3个应用场景，每个用一句话")
        applications = (prompt2 | llm | StrOutputParser()).invoke({"topic": topic})
        
        return f"📚 {topic}\n\n定义：{explanation}\n\n应用：\n{applications}"
    
    result = analyze_topic.invoke({"topic": "深度学习"})
    print(result)
    print()


def demo_runnable_passthrough():
    """2.3 RunnablePassthrough - 数据透传与增强"""
    print("=" * 60)
    print("2.3 RunnablePassthrough - 数据透传与增强")
    print("=" * 60)
    
    # 基础透传
    print("--- 基础透传 ---")
    passthrough = RunnablePassthrough()
    print(f"透传结果: {passthrough.invoke({'name': 'Alice'})}")
    
    # assign() - 添加新字段
    print("\n--- assign() 添加新字段 ---")
    enhanced = RunnablePassthrough.assign(
        text_length=lambda x: len(x.get("text", "")),
        uppercase=lambda x: x.get("text", "").upper()
    )
    result = enhanced.invoke({"text": "hello world", "id": 1})
    print(f"增强后: {result}")
    
    # 实际应用：RAG 场景
    print("\n--- 实际应用：模拟 RAG ---")
    def fake_retriever(query):
        return f"[检索到的相关文档：关于 {query} 的信息...]"
    
    prompt = ChatPromptTemplate.from_template(
        "根据以下上下文回答问题：\n上下文：{context}\n问题：{question}"
    )
    
    rag_chain = (
        {"context": lambda x: fake_retriever(x["question"]), "question": itemgetter("question")}
        | prompt | llm | StrOutputParser()
    )
    result = rag_chain.invoke({"question": "什么是向量数据库？"})
    print(f"RAG 结果: {result}")
    print()


def demo_runnable_parallel():
    """2.4 RunnableParallel - 并行执行"""
    print("=" * 60)
    print("2.4 RunnableParallel - 并行执行多个任务")
    print("=" * 60)
    
    joke_prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话，一句话")
    poem_prompt = ChatPromptTemplate.from_template("写一句关于{topic}的诗")
    fact_prompt = ChatPromptTemplate.from_template("说一个关于{topic}的有趣事实，一句话")
    
    parallel = RunnableParallel(
        joke=joke_prompt | llm | StrOutputParser(),
        poem=poem_prompt | llm | StrOutputParser(),
        fact=fact_prompt | llm | StrOutputParser()
    )
    
    results = parallel.invoke({"topic": "程序员"})
    print(f"笑话: {results['joke']}")
    print(f"诗句: {results['poem']}")
    print(f"事实: {results['fact']}")
    print()


def demo_runnable_branch():
    """2.5 RunnableBranch - 条件路由"""
    print("=" * 60)
    print("2.5 RunnableBranch - 条件路由分支")
    print("=" * 60)
    
    tech_prompt = ChatPromptTemplate.from_template("你是技术专家。用专业术语回答：{question}")
    casual_prompt = ChatPromptTemplate.from_template("你是友好的助手。用轻松的语气回答：{question}")
    default_prompt = ChatPromptTemplate.from_template("请回答：{question}")
    
    tech_chain = tech_prompt | llm | StrOutputParser()
    casual_chain = casual_prompt | llm | StrOutputParser()
    default_chain = default_prompt | llm | StrOutputParser()
    
    def is_tech_question(x):
        keywords = ["代码", "编程", "算法", "API", "数据库", "Python", "Java"]
        return any(kw in x.get("question", "") for kw in keywords)
    
    def is_casual_question(x):
        keywords = ["笑话", "故事", "有趣", "好玩", "推荐"]
        return any(kw in x.get("question", "") for kw in keywords)
    
    branch = RunnableBranch(
        (is_tech_question, tech_chain),
        (is_casual_question, casual_chain),
        default_chain
    )
    
    questions = [
        {"question": "Python的装饰器是什么？"},
        {"question": "讲个有趣的笑话"},
        {"question": "今天天气怎么样？"},
    ]
    
    for q in questions:
        result = branch.invoke(q)
        print(f"问题: {q['question']}")
        print(f"回答: {result}\n")


# ============================================================
# 第三部分：顺序链与数据流
# ============================================================

def demo_sequential_chain():
    """3.1 顺序链 - 多步骤串联"""
    print("=" * 60)
    print("3.1 顺序链 - 多步骤串联执行")
    print("=" * 60)
    
    translate_prompt = ChatPromptTemplate.from_template(
        "将以下中文翻译成英文，只输出翻译结果：{text}"
    )
    translate_chain = translate_prompt | llm | StrOutputParser()
    
    summary_prompt = ChatPromptTemplate.from_template("Summarize in one sentence: {translated}")
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    full_chain = (
        {"translated": translate_chain, "original": itemgetter("text")}
        | RunnablePassthrough.assign(summary=summary_chain)
    )
    
    result = full_chain.invoke({
        "text": "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车，AI无处不在。"
    })
    print(f"原文: {result['original']}")
    print(f"翻译: {result['translated']}")
    print(f"总结: {result['summary']}")
    print()


def demo_itemgetter():
    """3.2 itemgetter - 提取字典字段"""
    print("=" * 60)
    print("3.2 itemgetter - 提取和重组数据")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("用户 {name} 问：{question}\n请友好地回答。")
    
    chain = (
        {"name": itemgetter("user_info") | RunnableLambda(lambda x: x["name"]), "question": itemgetter("query")}
        | prompt | llm | StrOutputParser()
    )
    
    result = chain.invoke({"user_info": {"name": "小明", "age": 25}, "query": "Python怎么学？"})
    print(f"结果: {result}")
    print()


# ============================================================
# 第四部分：调用方式
# ============================================================

def demo_invoke_methods():
    """4.1 同步调用方式"""
    print("=" * 60)
    print("4.1 同步调用方式: invoke / stream / batch")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    chain = prompt | llm | StrOutputParser()
    
    print("--- invoke() 单个调用 ---")
    result = chain.invoke({"topic": "Python"})
    print(f"结果: {result}")
    
    print("\n--- stream() 流式输出 ---")
    print("结果: ", end="")
    for chunk in chain.stream({"topic": "机器学习的应用场景"}):
        print(chunk, end="", flush=True)
    print()
    
    print("\n--- batch() 批量处理 ---")
    topics = [{"topic": "Java"}, {"topic": "Go"}, {"topic": "Rust"}]
    results = chain.batch(topics)
    for topic, result in zip(topics, results):
        print(f"{topic['topic']}: {result}")
    print()


async def demo_async_methods():
    """4.2 异步调用方式"""
    print("=" * 60)
    print("4.2 异步调用方式: ainvoke / astream / abatch")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    chain = prompt | llm | StrOutputParser()
    
    print("--- ainvoke() 异步调用 ---")
    result = await chain.ainvoke({"topic": "Docker"})
    print(f"结果: {result}")
    
    print("\n--- astream() 异步流式 ---")
    print("结果: ", end="")
    async for chunk in chain.astream({"topic": "Kubernetes"}):
        print(chunk, end="", flush=True)
    print()
    
    print("\n--- abatch() 异步批量 ---")
    topics = [{"topic": "Redis"}, {"topic": "MongoDB"}]
    results = await chain.abatch(topics)
    for topic, result in zip(topics, results):
        print(f"{topic['topic']}: {result}")
    
    print("\n--- asyncio.gather() 并发调用 ---")
    tasks = [chain.ainvoke({"topic": "MySQL"}), chain.ainvoke({"topic": "PostgreSQL"})]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"结果: {result}")
    print()


# ============================================================
# 第五部分：错误处理
# ============================================================

def demo_retry():
    """5.1 with_retry - 自动重试"""
    print("=" * 60)
    print("5.1 with_retry - 自动重试机制")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    chain = prompt | llm | StrOutputParser()
    
    chain_with_retry = chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    result = chain_with_retry.invoke({"topic": "错误处理"})
    print(f"结果: {result}")
    print("💡 with_retry 适合处理网络波动、API 限流等临时错误")
    print()


def demo_fallback():
    """5.2 with_fallbacks - 降级备选"""
    print("=" * 60)
    print("5.2 with_fallbacks - 降级备选方案")
    print("=" * 60)
    
    main_prompt = ChatPromptTemplate.from_template("详细解释{topic}的原理")
    main_chain = main_prompt | llm | StrOutputParser()
    
    fallback_prompt = ChatPromptTemplate.from_template("简单介绍{topic}")
    fallback_chain = fallback_prompt | llm | StrOutputParser()
    
    chain_with_fallback = main_chain.with_fallbacks([fallback_chain])
    result = chain_with_fallback.invoke({"topic": "量子计算"})
    print(f"结果: {result}")
    print("💡 with_fallbacks 适合主模型不可用时切换到备用模型")
    print()


# ============================================================
# 第六部分：配置与调试
# ============================================================

def demo_bind():
    """6.1 bind - 绑定参数"""
    print("=" * 60)
    print("6.1 bind - 绑定固定参数")
    print("=" * 60)
    
    llm_with_stop = llm.bind(stop=["\n"])
    prompt = ChatPromptTemplate.from_template("列举3个{topic}：\n1.")
    chain = prompt | llm_with_stop | StrOutputParser()
    
    result = chain.invoke({"topic": "编程语言"})
    print(f"结果（遇到换行停止）: 1.{result}")
    print()


def demo_config():
    """6.2 with_config - 添加配置"""
    print("=" * 60)
    print("6.2 with_config - 添加 tags 和 metadata")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("介绍{topic}")
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke(
        {"topic": "LangSmith"},
        config={"run_name": "demo_config_run", "tags": ["runtime_tag"], "metadata": {"user_id": "test"}}
    )
    print(f"结果: {result}")
    print("💡 tags 和 metadata 在 LangSmith 中可用于过滤和分析")
    print()


def demo_debug():
    """6.3 调试技巧"""
    print("=" * 60)
    print("6.3 调试技巧 - 查看链的结构")
    print("=" * 60)
    
    prompt = ChatPromptTemplate.from_template("介绍{topic}")
    chain = prompt | llm | StrOutputParser()
    
    print("输入 Schema:")
    print(chain.input_schema.model_json_schema())
    print("\n输出 Schema:")
    print(chain.output_schema.model_json_schema())
    print(f"\n链的第一步: {chain.first}")
    print(f"链的最后一步: {chain.last}")
    print()


# ============================================================
# 主函数
# ============================================================

async def run_async_demos():
    await demo_async_methods()


def main():
    print("\n🔗 第四课：链 (Chains) - LCEL 完全指南\n")
    
    print("\n" + "=" * 60)
    print("📚 第一部分：LCEL 基础")
    print("=" * 60)
    demo_simple_chain()
    demo_chain_equivalents()
    
    print("\n" + "=" * 60)
    print("📚 第二部分：核心 Runnable 组件")
    print("=" * 60)
    demo_runnable_lambda()
    demo_chain_decorator()
    demo_runnable_passthrough()
    demo_runnable_parallel()
    demo_runnable_branch()
    
    print("\n" + "=" * 60)
    print("📚 第三部分：顺序链与数据流")
    print("=" * 60)
    demo_sequential_chain()
    demo_itemgetter()
    
    print("\n" + "=" * 60)
    print("📚 第四部分：调用方式")
    print("=" * 60)
    demo_invoke_methods()
    asyncio.run(run_async_demos())
    
    print("\n" + "=" * 60)
    print("📚 第五部分：错误处理")
    print("=" * 60)
    demo_retry()
    demo_fallback()
    
    print("\n" + "=" * 60)
    print("📚 第六部分：配置与调试")
    print("=" * 60)
    demo_bind()
    demo_config()
    demo_debug()
    
    print("\n" + "=" * 60)
    print("📌 第四课总结")
    print("=" * 60)
    print("""
    核心组件                用途
    ─────────────────────────────────────────────
    RunnableLambda         包装自定义 Python 函数
    RunnablePassthrough    透传数据 / assign() 增强
    RunnableParallel       并行执行多个链
    RunnableBranch         条件路由分支
    @chain 装饰器          优雅定义自定义链
    
    调用方式                适用场景
    ─────────────────────────────────────────────
    invoke()               单个同步调用
    stream()               流式输出（用户交互）
    batch()                批量处理
    ainvoke/astream/abatch 异步版本（Web服务）
    
    错误处理                用途
    ─────────────────────────────────────────────
    with_retry()           自动重试（网络波动）
    with_fallbacks()       降级备选（模型切换）
    
    配置调试                用途
    ─────────────────────────────────────────────
    bind()                 绑定固定参数
    with_config()          添加 tags/metadata
    input_schema           查看输入类型
    """)


if __name__ == "__main__":
    main()

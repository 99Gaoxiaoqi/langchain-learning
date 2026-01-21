"""
第四课：链 (Chains)
学习目标：
1. 理解 LCEL (LangChain Expression Language)
2. 使用管道操作符 | 组合组件
3. 创建顺序链和并行链
4. 实现条件分支
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

def simple_chain():
    """简单链示例"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # 使用 LCEL 创建链
    prompt = ChatPromptTemplate.from_template("用一句话解释什么是{concept}")
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"concept": "机器学习"})
    print("结果:", result)

def sequential_chain():
    """顺序链示例 - 先翻译再总结"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # 第一步：翻译
    translate_prompt = ChatPromptTemplate.from_template(
        "将以下中文翻译成英文：{text}"
    )
    translate_chain = translate_prompt | llm | StrOutputParser()
    
    # 第二步：总结
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {translated}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # 组合链
    full_chain = (
        {"translated": translate_chain}
        | summary_chain
    )
    
    result = full_chain.invoke({"text": "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车。"})
    print("最终结果:", result)

def parallel_chain():
    """并行链示例"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # 创建多个并行任务
    joke_prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
    poem_prompt = ChatPromptTemplate.from_template("写一首关于{topic}的俳句")
    fact_prompt = ChatPromptTemplate.from_template("说一个关于{topic}的有趣事实")
    
    # 并行执行
    parallel = RunnableParallel(
        joke=joke_prompt | llm | StrOutputParser(),
        poem=poem_prompt | llm | StrOutputParser(),
        fact=fact_prompt | llm | StrOutputParser()
    )
    
    results = parallel.invoke({"topic": "猫"})
    print("笑话:", results["joke"])
    print("\n俳句:", results["poem"])
    print("\n事实:", results["fact"])

if __name__ == "__main__":
    print("=== 简单链 ===")
    simple_chain()
    print("\n=== 顺序链 ===")
    sequential_chain()
    print("\n=== 并行链 ===")
    parallel_chain()

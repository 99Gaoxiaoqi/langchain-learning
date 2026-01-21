"""
第二课：Prompt 模板
学习目标：
1. 掌握 3 种模板格式：f-string / jinja2 / mustache
2. 理解 PromptTemplate 和 ChatPromptTemplate
3. 学会 Few-shot 提示和 MessagesPlaceholder
4. 使用企业级 PromptManager 管理模板
"""
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from llm_factory import get_llm
from prompt_manager import PromptManager, create_default_manager

llm = get_llm()


# ============================================================
# 第一部分：3 种模板格式
# ============================================================

def demo_template_formats():
    """演示 3 种模板格式"""
    print("=" * 50)
    print("1️⃣ 三种模板格式对比")
    print("=" * 50)
    
    # 方式1：f-string 格式（默认）
    print("\n📌 f-string 格式（默认，推荐简单场景）")
    prompt1 = PromptTemplate.from_template(
        "用一句话介绍{topic}"
    )
    chain1 = prompt1 | llm
    response1 = chain1.invoke({"topic": "Python"})
    print(f"   结果: {response1.content}")
    
    # 方式2：jinja2 格式（企业级推荐）
    print("\n📌 jinja2 格式（企业级推荐，支持条件/循环）")
    prompt2 = PromptTemplate.from_template(
        "{% if vip %}尊敬的VIP用户{% else %}亲爱的用户{% endif %}，用一句话回答：{{ question }}",
        template_format="jinja2"
    )
    chain2 = prompt2 | llm
    response2 = chain2.invoke({"vip": True, "question": "什么是LangChain？"})
    print(f"   结果: {response2.content}")
    
    # 方式3：mustache 格式
    print("\n📌 mustache 格式（前端常用）")
    prompt3 = PromptTemplate.from_template(
        "用一句话介绍{{topic}}",
        template_format="mustache"
    )
    chain3 = prompt3 | llm
    response3 = chain3.invoke({"topic": "Java"})
    print(f"   结果: {response3.content}")
    print()


def demo_jinja2_advanced():
    """jinja2 高级用法 - 条件和循环"""
    print("=" * 50)
    print("2️⃣ jinja2 高级用法（条件/循环）")
    print("=" * 50)
    
    # 带条件判断的模板
    template_with_condition = """
{% if role == "expert" %}你是资深专家，用专业术语简洁回答。
{% elif role == "teacher" %}你是老师，用通俗语言简洁回答。
{% else %}你是助手，简洁回答。{% endif %}
问题：{{ question }}"""
    
    prompt = PromptTemplate.from_template(
        template_with_condition.strip(),
        template_format="jinja2"
    )
    
    chain = prompt | llm
    
    # 测试不同角色
    for role in ["expert", "teacher", "normal"]:
        response = chain.invoke({
            "role": role,
            "question": "什么是装饰器？"
        })
        content = response.content[:80] + "..." if len(response.content) > 80 else response.content
        print(f"   [{role}] {content}")
    
    # 带循环的模板
    print("\n" + "-" * 30)
    print("📌 jinja2 循环示例")
    
    template_with_loop = """根据要点回答：
{% for point in points %}{{ loop.index }}. {{ point }} {% endfor %}
问题：{{ question }}（用一句话回答）"""
    
    prompt2 = PromptTemplate.from_template(
        template_with_loop.strip(),
        template_format="jinja2"
    )
    chain2 = prompt2 | llm
    response = chain2.invoke({
        "points": ["简洁", "举例"],
        "question": "列表推导式怎么用？"
    })
    print(f"   回答: {response.content}")
    print()


# ============================================================
# 第二部分：ChatPromptTemplate
# ============================================================

def demo_chat_prompt_template():
    """ChatPromptTemplate - 多角色对话模板"""
    print("=" * 50)
    print("3️⃣ ChatPromptTemplate - 多角色模板")
    print("=" * 50)
    
    # 基本用法
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请用{style}的方式回答问题。"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "role": "Python专家",
        "style": "简洁专业",
        "question": "什么是生成器？"
    })
    print(f"响应: {response.content}\n")


# ============================================================
# 第三部分：Few-shot 提示
# ============================================================

def demo_few_shot():
    """Few-shot 提示学习"""
    print("=" * 50)
    print("4️⃣ Few-shot 提示（示例学习）")
    print("=" * 50)
    
    # 定义示例
    examples = [
        {"input": "开心", "output": "😊"},
        {"input": "难过", "output": "😢"},
        {"input": "生气", "output": "😠"},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个表情翻译器，将情绪词翻译成对应的emoji。"),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    chain = final_prompt | llm
    
    for word in ["惊讶", "困惑", "期待"]:
        response = chain.invoke({"input": word})
        print(f"   {word} → {response.content}")
    print()


# ============================================================
# 第四部分：MessagesPlaceholder
# ============================================================

def demo_messages_placeholder():
    """MessagesPlaceholder - 历史消息占位符"""
    print("=" * 50)
    print("5️⃣ MessagesPlaceholder - 历史消息")
    print("=" * 50)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    # 模拟历史对话
    history = [
        HumanMessage(content="我叫小明，我在学Python"),
        AIMessage(content="你好小明！Python是个很好的选择，有什么问题可以问我。"),
    ]
    
    chain = prompt | llm
    response = chain.invoke({
        "history": history,
        "question": "你还记得我在学什么吗？"
    })
    print(f"响应: {response.content}\n")


# ============================================================
# 第五部分：企业级 PromptManager
# ============================================================

def demo_prompt_manager():
    """企业级 PromptManager 使用"""
    print("=" * 50)
    print("6️⃣ 企业级 PromptManager")
    print("=" * 50)
    
    # 使用预置管理器
    manager = create_default_manager()
    
    print(f"已注册意图: {manager.list_intents()}\n")
    
    # 模拟意图识别后的路由
    test_cases = [
        ("presale", "你们产品多少钱？"),
        ("aftersale", "我要退货"),
        ("technical", "代码报错了怎么办？"),
    ]
    
    for intent, question in test_cases:
        chain = manager.route(intent, llm, default_intent="general")
        response = chain.invoke({"question": question})
        content = response.content[:80] + "..." if len(response.content) > 80 else response.content
        print(f"[{intent}] {question}")
        print(f"   → {content}\n")
    
    # 自定义注册
    print("-" * 30)
    print("📌 动态注册新模板")
    
    manager.register(
        intent="code_review",
        name="代码审查",
        system_prompt="你是一个代码审查专家，请审查用户提供的代码，指出问题并给出改进建议。",
    )
    
    chain = manager.get_chain("code_review", llm)
    response = chain.invoke({"question": "def add(a,b): return a+b"})
    print(f"代码审查结果: {response.content[:100]}...")
    print()


def main():
    print("\n🚀 Prompt 模板教程\n")
    
    demo_template_formats()
    demo_jinja2_advanced()
    demo_chat_prompt_template()
    demo_few_shot()
    demo_messages_placeholder()
    demo_prompt_manager()
    
    print("=" * 50)
    print("📌 总结")
    print("=" * 50)
    print("""
    模板格式                    适用场景
    ─────────────────────────────────────────────
    f-string {var}              简单场景（默认）
    jinja2 {{ var }}            企业级（支持条件/循环）
    mustache {{var}}            前端兼容
    
    模板类型                    适用场景
    ─────────────────────────────────────────────
    PromptTemplate              简单字符串模板
    ChatPromptTemplate          多角色对话（推荐）
    FewShotChatMessagePromptTemplate  示例学习
    MessagesPlaceholder         历史消息/动态内容
    
    企业级方案
    ─────────────────────────────────────────────
    PromptManager               模板注册/路由/A-B测试/持久化
    jinja2 + 数据库存储         运营可配置
    """)


if __name__ == "__main__":
    main()

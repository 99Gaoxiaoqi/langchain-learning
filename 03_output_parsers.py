"""
第三课：输出解析器
学习目标：
1. 理解输出解析器的作用
2. 掌握常用解析器：Str/Json/Pydantic/List
3. 学会 with_structured_output（企业级推荐）
4. 处理解析错误
"""
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain_classic.output_parsers import OutputFixingParser
from llm_factory import get_llm

llm = get_llm(provider="openai")


# ============================================================
# 第一部分：StrOutputParser（字符串解析）
# ============================================================

def demo_str_parser():
    """StrOutputParser - 最基础的解析器"""
    print("=" * 50)
    print("1️⃣ StrOutputParser - 字符串解析")
    print("=" * 50)
    
    prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
    
    # 不使用解析器：返回 AIMessage 对象
    chain_without_parser = prompt | llm
    result1 = chain_without_parser.invoke({"topic": "Python"})
    print(f"无解析器: {type(result1).__name__} -> {result1.content[:50]}...")
    
    # 使用 StrOutputParser：直接返回字符串
    chain_with_parser = prompt | llm | StrOutputParser()
    result2 = chain_with_parser.invoke({"topic": "Python"})
    print(f"有解析器: {type(result2).__name__} -> {result2[:50]}...")
    print()


# ============================================================
# 第二部分：CommaSeparatedListOutputParser（列表解析）
# ============================================================

def demo_list_parser():
    """CommaSeparatedListOutputParser - 逗号分隔列表"""
    print("=" * 50)
    print("2️⃣ CommaSeparatedListOutputParser - 列表解析")
    print("=" * 50)
    
    parser = CommaSeparatedListOutputParser()
    
    prompt = ChatPromptTemplate.from_template(
        "列出5种{category}，用逗号分隔，只输出名称"
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({"category": "编程语言"})
    
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print()


# ============================================================
# 第三部分：JsonOutputParser（JSON 解析）
# ============================================================

def demo_json_parser():
    """JsonOutputParser - JSON 格式解析"""
    print("=" * 50)
    print("3️⃣ JsonOutputParser - JSON 解析")
    print("=" * 50)
    
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_template(
        """分析这个水果的特点，返回JSON格式：
水果：{fruit}
格式：{{"name": "名称", "color": "颜色", "taste": "口味"}}
只返回JSON，不要其他内容。"""
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({"fruit": "草莓"})
    
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print(f"访问字段: name={result.get('name')}, color={result.get('color')}")
    print()


# ============================================================
# 第四部分：PydanticOutputParser（结构化解析）
# ============================================================

class BookInfo(BaseModel):
    """书籍信息"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    year: int = Field(description="出版年份")
    summary: str = Field(description="一句话简介")


def demo_pydantic_parser():
    """PydanticOutputParser - Pydantic 模型解析"""
    print("=" * 50)
    print("4️⃣ PydanticOutputParser - 结构化解析")
    print("=" * 50)
    
    parser = PydanticOutputParser(pydantic_object=BookInfo)
    
    # 获取格式说明（会告诉 LLM 如何输出）
    format_instructions = parser.get_format_instructions()
    print(f"格式说明:\n{format_instructions[:200]}...\n")
    
    prompt = ChatPromptTemplate.from_template(
        """推荐一本关于{topic}的经典书籍。
{format_instructions}"""
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({
        "topic": "人工智能",
        "format_instructions": format_instructions
    })
    
    print(f"类型: {type(result)}")
    print(f"书名: {result.title}")
    print(f"作者: {result.author}")
    print(f"年份: {result.year}")
    print(f"简介: {result.summary}")
    print()


# ============================================================
# 第五部分：with_structured_output（企业级推荐）
# ============================================================

class MovieInfo(BaseModel):
    """电影信息"""
    title: str = Field(description="电影名称")
    year: Optional[int] = Field(default=None, description="上映年份")
    director: Optional[str] = Field(default=None, description="导演")
    rating: Optional[float] = Field(default=None, description="评分(1-10)")
    genre: Optional[str] = Field(default=None, description="类型")


def demo_structured_output():
    """with_structured_output - 企业级推荐方式"""
    print("=" * 50)
    print("5️⃣ with_structured_output - 企业级推荐")
    print("=" * 50)
    
    # 直接绑定结构，无需手动写格式说明
    structured_llm = llm.with_structured_output(MovieInfo)
    
    result = structured_llm.invoke("介绍电影《盗梦空间》")
    
    print(f"类型: {type(result)}")
    print(f"电影: {result.title}")
    print(f"导演: {result.director}")
    print(f"年份: {result.year}")
    print(f"评分: {result.rating}")
    print()


# ============================================================
# 第六部分：with_structured_output + include_raw
# ============================================================

def demo_structured_output_raw():
    """获取原始响应和解析结果"""
    print("=" * 50)
    print("6️⃣ with_structured_output + include_raw")
    print("=" * 50)
    
    structured_llm = llm.with_structured_output(MovieInfo, include_raw=True)
    
    result = structured_llm.invoke("介绍电影《阿甘正传》")
    
    print(f"解析结果: {result['parsed']}")
    print(f"解析错误: {result['parsing_error']}")
    print(f"原始响应类型: {type(result['raw']).__name__}")
    print()


# ============================================================
# 第七部分：使用 JSON Schema（更灵活）
# ============================================================

def demo_json_schema():
    """使用 JSON Schema 定义结构"""
    print("=" * 50)
    print("7️⃣ JSON Schema 方式")
    print("=" * 50)
    
    json_schema = {
        "title": "Person",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "姓名"},
            "age": {"type": "integer", "description": "年龄"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "技能列表"
            }
        },
        "required": ["name", "age", "skills"]
    }
    
    structured_llm = llm.with_structured_output(json_schema)
    
    result = structured_llm.invoke("描述一个Python程序员的信息")
    
    print(f"类型: {type(result)}")
    print(f"结果: {result}")
    print()


# ============================================================
# 第八部分：链式组合
# ============================================================

class AnalysisResult(BaseModel):
    """分析结果"""
    sentiment: str = Field(description="情感：positive/negative/neutral")
    confidence: float = Field(description="置信度：0-1")
    keywords: list[str] = Field(description="关键词列表")


def demo_chain_with_parser():
    """完整的链式组合示例"""
    print("=" * 50)
    print("8️⃣ 链式组合示例（情感分析）")
    print("=" * 50)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个文本分析专家，分析用户输入的情感和关键词。"),
        ("human", "{text}"),
    ])
    
    structured_llm = llm.with_structured_output(AnalysisResult)
    
    chain = prompt | structured_llm
    
    texts = [
        "这个产品太棒了，我非常喜欢！",
        "服务态度很差，不会再来了。",
    ]
    
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"文本: {text}")
        print(f"  情感: {result.sentiment}, 置信度: {result.confidence}")
        print(f"  关键词: {result.keywords}")
    print()


# ============================================================
# 第九部分：模型对比（OpenAI vs 通义千问）
# ============================================================

def demo_model_comparison():
    """对比不同模型的 with_structured_output 支持"""
    print("=" * 50)
    print("9️⃣ 模型对比：OpenAI vs 通义千问")
    print("=" * 50)
    
    llm_openai = get_llm(provider="openai")
    llm_qwen = get_llm(provider="qwen")
    
    # OpenAI - 直接使用 with_structured_output
    print("\n📌 OpenAI (gpt-4o-mini):")
    try:
        structured_openai = llm_openai.with_structured_output(MovieInfo)
        result = structured_openai.invoke("介绍电影《星际穿越》")
        print(f"   ✅ 成功: {result.title} ({result.year}) - {result.director}")
    except Exception as e:
        print(f"   ❌ 失败: {str(e)[:80]}")
    
    # 通义千问 - 直接使用 with_structured_output（会失败）
    print("\n📌 通义千问 - with_structured_output（无json关键词）:")
    try:
        structured_qwen = llm_qwen.with_structured_output(MovieInfo)
        result = structured_qwen.invoke("介绍电影《星际穿越》")
        print(f"   ✅ 成功: {result.title} ({result.year}) - {result.director}")
    except Exception as e:
        print(f"   ❌ 失败: prompt需包含'json'关键词")
    
    # 通义千问 - 使用 PydanticOutputParser（推荐方案）
    print("\n📌 通义千问 - PydanticOutputParser（推荐）:")
    try:
        parser = PydanticOutputParser(pydantic_object=MovieInfo)
        prompt = ChatPromptTemplate.from_template(
            "介绍电影《{movie}》\n{format_instructions}"
        )
        chain = prompt | llm_qwen | parser
        result = chain.invoke({
            "movie": "星际穿越",
            "format_instructions": parser.get_format_instructions()
        })
        print(f"   ✅ 成功: {result.title} ({result.year}) - {result.director}")
    except Exception as e:
        print(f"   ❌ 失败: {str(e)[:80]}")
    
    print()


# ============================================================
# 第十部分：OutputFixingParser（自动修复）
# ============================================================

def demo_output_fixing_parser():
    """OutputFixingParser - 自动修复解析错误"""
    print("=" * 50)
    print("🔟 OutputFixingParser - 自动修复")
    print("=" * 50)
    
    # 基础解析器
    base_parser = PydanticOutputParser(pydantic_object=BookInfo)
    
    # 包装成自动修复解析器（解析失败时会调用 LLM 修复）
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    
    # 模拟一个格式错误的输出（单引号、缺引号等）
    bad_output = """{'title': '深度学习', 'author': 'Ian Goodfellow', year: 2016, "summary": "深度学习入门经典"}"""
    
    print(f"错误格式输入: {bad_output}")
    print()
    
    # 普通解析器会失败
    print("📌 普通 PydanticOutputParser:")
    try:
        result = base_parser.parse(bad_output)
        print(f"   ✅ 成功: {result}")
    except Exception as e:
        print(f"   ❌ 失败: {str(e)[:60]}...")
    
    # OutputFixingParser 会自动修复
    print("\n📌 OutputFixingParser（自动修复）:")
    try:
        result = fixing_parser.parse(bad_output)
        print(f"   ✅ 成功: {result.title} - {result.author} ({result.year})")
    except Exception as e:
        print(f"   ❌ 失败: {str(e)[:60]}...")
    
    print()


def main():
    print("\n🚀 输出解析器教程\n")
    
    demo_str_parser()
    demo_list_parser()
    demo_json_parser()
    demo_pydantic_parser()
    demo_structured_output()
    demo_structured_output_raw()
    demo_json_schema()
    demo_chain_with_parser()
    demo_model_comparison()
    demo_output_fixing_parser()
    
    print("=" * 50)
    print("📌 总结")
    print("=" * 50)
    print("""
    解析器类型                  适用场景
    ─────────────────────────────────────────────
    StrOutputParser             简单文本输出
    CommaSeparatedListOutputParser  逗号分隔列表
    JsonOutputParser            JSON 格式
    PydanticOutputParser        结构化数据（兼容性最好）
    OutputFixingParser          自动修复格式错误
    
    with_structured_output 支持情况
    ─────────────────────────────────────────────
    OpenAI/Anthropic            ✅ 完整支持
    通义千问/DeepSeek           ⚠️ 需prompt含'json'，推荐用PydanticOutputParser
    """)


if __name__ == "__main__":
    main()

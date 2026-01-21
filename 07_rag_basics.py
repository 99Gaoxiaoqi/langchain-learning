"""
第七课：RAG 基础 (检索增强生成)
学习目标：
1. 理解 RAG 的概念和流程
2. 文档加载和分割
3. 向量存储基础
4. 简单的问答系统
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 模拟文档数据
DOCUMENTS = [
    {
        "content": "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件，使开发者能够轻松构建复杂的 AI 应用。",
        "metadata": {"source": "langchain_intro.txt"}
    },
    {
        "content": "RAG（检索增强生成）是一种结合检索和生成的技术。它首先从知识库中检索相关文档，然后将这些文档作为上下文提供给语言模型。",
        "metadata": {"source": "rag_intro.txt"}
    },
    {
        "content": "向量数据库用于存储和检索向量嵌入。常见的向量数据库包括 Pinecone、Milvus、Chroma 等。",
        "metadata": {"source": "vector_db.txt"}
    },
    {
        "content": "Prompt 工程是设计和优化提示词的过程，好的提示词可以显著提高模型输出的质量。",
        "metadata": {"source": "prompt_engineering.txt"}
    }
]

def simple_retriever(query: str, top_k: int = 2) -> list:
    """
    简单的关键词检索器（实际应用中应使用向量检索）
    """
    results = []
    query_lower = query.lower()
    
    for doc in DOCUMENTS:
        # 简单的关键词匹配
        score = sum(1 for word in query_lower.split() if word in doc["content"].lower())
        if score > 0:
            results.append((score, doc))
    
    # 按分数排序
    results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in results[:top_k]]

def format_docs(docs: list) -> str:
    """格式化文档"""
    return "\n\n".join([doc["content"] for doc in docs])

def simple_rag():
    """简单的 RAG 示例"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # RAG 提示模板
    prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答：""")
    
    # 创建 RAG 链
    def rag_chain(question: str) -> str:
        # 1. 检索相关文档
        docs = simple_retriever(question)
        context = format_docs(docs)
        
        # 2. 生成回答
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
    
    # 测试问题
    questions = [
        "什么是 LangChain？",
        "RAG 是什么技术？",
        "有哪些向量数据库？",
        "什么是量子计算？"  # 知识库中没有的问题
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        answer = rag_chain(q)
        print(f"回答: {answer}")

def document_splitting_demo():
    """文档分割示例"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 长文本
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需明确编程。
    深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。
    
    自然语言处理（NLP）是AI的另一个重要分支，它使计算机能够理解、解释和生成人类语言。
    大语言模型（LLM）是NLP领域的最新突破，如GPT、Claude等。
    """
    
    # 创建分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    
    # 分割文档
    chunks = splitter.split_text(long_text)
    
    print("=== 文档分割结果 ===")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1} ({len(chunk)} 字符):")
        print(chunk.strip())

if __name__ == "__main__":
    print("=== 简单 RAG 示例 ===")
    simple_rag()
    print("\n" + "="*50)
    document_splitting_demo()

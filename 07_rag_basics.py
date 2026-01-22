"""
第七课：RAG 基础 (检索增强生成) - 让 LLM 拥有外部知识

学习目标：
1. 理解 RAG 的核心概念和完整流程
2. 掌握文档加载器（Document Loaders）的使用
3. 学会文本分割（Text Splitters）的各种策略
4. 理解向量存储（Vector Stores）和嵌入（Embeddings）
5. 掌握检索器（Retrievers）的多种类型
6. 构建完整的 RAG 应用

核心概念：
─────────────────────────────────────────────────────────────────────────────
RAG (Retrieval-Augmented Generation) 检索增强生成：
- LLM 的知识有截止日期，且无法访问私有数据
- RAG 通过检索外部知识库，将相关信息注入到 Prompt 中
- 让 LLM 能够基于最新/私有数据生成准确回答

RAG 完整流程：
┌─────────────────────────────────────────────────────────────────────────────┐
│ 索引阶段 (Indexing) - 离线处理                                              │
│ ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│ │ 文档加载 │ → │ 文本分割 │ → │ 向量嵌入 │ → │ 存入向量库│                  │
│ │ Loaders  │   │ Splitters│   │Embeddings│   │VectorStore│                  │
│ └──────────┘   └──────────┘   └──────────┘   └──────────┘                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 检索阶段 (Retrieval) - 在线查询                                             │
│ ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│ │ 用户问题 │ → │ 向量检索 │ → │ 构建Prompt│ → │ LLM生成  │                  │
│ │  Query   │   │ Retriever│   │ + Context │   │  Answer  │                  │
│ └──────────┘   └──────────┘   └──────────┘   └──────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘

本课涵盖的 RAG 构建方式：
┌────────────────────┬─────────────────────┬─────────────────────────────────┐
│ 方式               │ 适用场景            │ 特点                            │
├────────────────────┼─────────────────────┼─────────────────────────────────┤
│ 手动 LCEL 链       │ 学习理解/简单场景   │ 完全控制，理解原理              │
│ create_retrieval_  │ 标准 RAG 场景       │ 官方推荐，自动处理文档格式      │
│ chain              │                     │                                 │
│ 高级检索器         │ 复杂检索需求        │ 多查询/自查询/上下文压缩        │
└────────────────────┴─────────────────────┴─────────────────────────────────┘
─────────────────────────────────────────────────────────────────────────────
"""
import os
from dotenv import load_dotenv
from llm_factory import get_llm

load_dotenv()
llm = get_llm()


# ============================================================
# 第一部分：文档加载器 (Document Loaders)
# ============================================================
"""
文档加载器负责从各种来源加载数据，转换为 LangChain 的 Document 对象。

Document 对象结构：
┌─────────────────────────────────────────────────────────────┐
│ Document                                                    │
│ ├── page_content: str  # 文档的文本内容                     │
│ └── metadata: dict     # 元数据（来源、页码、作者等）       │
└─────────────────────────────────────────────────────────────┘

常用文档加载器：
┌────────────────────────┬────────────────────────────────────┐
│ 加载器                 │ 用途                               │
├────────────────────────┼────────────────────────────────────┤
│ TextLoader             │ 纯文本文件 (.txt)                  │
│ PyPDFLoader            │ PDF 文件                           │
│ CSVLoader              │ CSV 文件                           │
│ JSONLoader             │ JSON 文件                          │
│ UnstructuredLoader     │ 多种格式（PDF/Word/HTML等）        │
│ WebBaseLoader          │ 网页内容                           │
│ DirectoryLoader        │ 整个目录的文件                     │
└────────────────────────┴────────────────────────────────────┘
"""

def demo_document_loaders():
    """1.1 文档加载器演示"""
    print("=" * 60)
    print("1.1 文档加载器 (Document Loaders)")
    print("=" * 60)
    
    from langchain_core.documents import Document
    
    # 方式1：手动创建 Document（用于演示和测试）
    print("\n--- 手动创建 Document ---")
    docs = [
        Document(
            page_content="LangChain 是一个用于开发 LLM 应用的框架。它提供了模块化组件。",
            metadata={"source": "langchain_intro.txt", "page": 1}
        ),
        Document(
            page_content="RAG 是检索增强生成技术，结合检索和生成来提高回答质量。",
            metadata={"source": "rag_intro.txt", "page": 1}
        ),
        Document(
            page_content="向量数据库用于存储和检索向量嵌入，支持语义相似度搜索。",
            metadata={"source": "vector_db.txt", "page": 1}
        ),
    ]
    
    for doc in docs:
        print(f"内容: {doc.page_content[:30]}...")
        print(f"元数据: {doc.metadata}")
        print()
    
    return docs


def demo_text_loader():
    """1.2 TextLoader 示例"""
    print("=" * 60)
    print("1.2 TextLoader - 加载文本文件")
    print("=" * 60)
    
    # 先创建一个示例文本文件
    sample_text = """LangChain 框架介绍

LangChain 是一个强大的框架，用于开发由大语言模型（LLM）驱动的应用程序。

主要特点：
1. 模块化设计：提供可组合的组件
2. 链式调用：支持复杂的工作流
3. 丰富的集成：支持多种 LLM 和工具

RAG（检索增强生成）是 LangChain 的核心应用场景之一。
"""
    
    # 写入临时文件
    with open("_temp_sample.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    try:
        from langchain_community.document_loaders import TextLoader
        
        loader = TextLoader("_temp_sample.txt", encoding="utf-8")
        documents = loader.load()
        
        print(f"加载了 {len(documents)} 个文档")
        print(f"内容预览: {documents[0].page_content[:100]}...")
        print(f"元数据: {documents[0].metadata}")
        
    except ImportError:
        print("需要安装: pip install langchain-community")
    finally:
        # 清理临时文件
        if os.path.exists("_temp_sample.txt"):
            os.remove("_temp_sample.txt")
    print()


def demo_web_loader():
    """1.3 WebBaseLoader 示例"""
    print("=" * 60)
    print("1.3 WebBaseLoader - 加载网页内容")
    print("=" * 60)
    
    try:
        from langchain_community.document_loaders import WebBaseLoader
        
        # 加载网页（需要网络连接）
        # loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
        # documents = loader.load()
        
        print("WebBaseLoader 用法示例：")
        print("""
from langchain_community.document_loaders import WebBaseLoader

# 加载单个网页
loader = WebBaseLoader("https://example.com/page")
docs = loader.load()

# 加载多个网页
loader = WebBaseLoader([
    "https://example.com/page1",
    "https://example.com/page2"
])
docs = loader.load()

# 💡 提示：需要安装 beautifulsoup4
# pip install beautifulsoup4
""")
    except ImportError:
        print("需要安装: pip install langchain-community beautifulsoup4")
    print()


def demo_directory_loader():
    """1.4 DirectoryLoader 示例"""
    print("=" * 60)
    print("1.4 DirectoryLoader - 加载整个目录")
    print("=" * 60)
    
    print("""
DirectoryLoader 用法示例：

from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 加载目录下所有 .txt 文件
loader = DirectoryLoader(
    path="./documents",
    glob="**/*.txt",           # 匹配模式
    loader_cls=TextLoader,     # 使用的加载器类
    show_progress=True,        # 显示进度条
    use_multithreading=True,   # 多线程加载
)
docs = loader.load()

# 加载 PDF 文件
from langchain_community.document_loaders import PyPDFLoader
loader = DirectoryLoader(
    path="./pdfs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)

# 💡 提示：
# - glob 支持递归匹配 (**)
# - 可以指定不同的 loader_cls 处理不同格式
""")
    print()


# ============================================================
# 第二部分：文本分割器 (Text Splitters)
# ============================================================
"""
为什么需要文本分割？
1. LLM 有上下文长度限制（如 4K、8K、128K tokens）
2. 太长的文本会导致检索不精确
3. 适当大小的块能提高检索质量

分割策略对比：
┌────────────────────────────┬────────────────────────────────────────────┐
│ 分割器                     │ 特点                                       │
├────────────────────────────┼────────────────────────────────────────────┤
│ CharacterTextSplitter      │ 按字符数分割，简单但可能切断句子           │
│ RecursiveCharacterText     │ 递归分割，优先保持段落/句子完整（推荐）    │
│ Splitter                   │                                            │
│ TokenTextSplitter          │ 按 token 数分割，更精确控制                │
│ SentenceTransformers       │ 按语义分割，保持语义完整性                 │
│ TextSplitter               │                                            │
└────────────────────────────┴────────────────────────────────────────────┘

关键参数：
- chunk_size: 每个块的最大大小
- chunk_overlap: 块之间的重叠大小（保持上下文连贯）
- separators: 分割符优先级列表
"""

def demo_text_splitters():
    """2.1 文本分割器演示"""
    print("=" * 60)
    print("2.1 文本分割器 (Text Splitters)")
    print("=" * 60)
    
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    
    # 示例长文本
    long_text = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需明确编程。深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。

自然语言处理（NLP）是AI的另一个重要分支，它使计算机能够理解、解释和生成人类语言。大语言模型（LLM）是NLP领域的最新突破，如GPT、Claude、Qwen等。

LangChain 是一个用于开发 LLM 应用的框架，它提供了丰富的组件来构建复杂的 AI 应用。RAG（检索增强生成）是其核心应用场景之一。
"""
    
    print("\n--- CharacterTextSplitter（简单字符分割）---")
    char_splitter = CharacterTextSplitter(
        separator="\n\n",      # 分割符
        chunk_size=100,        # 块大小
        chunk_overlap=20,      # 重叠大小
    )
    char_chunks = char_splitter.split_text(long_text)
    print(f"分割成 {len(char_chunks)} 个块")
    for i, chunk in enumerate(char_chunks[:2]):
        print(f"块 {i+1}: {chunk[:50]}...")

    print("\n--- RecursiveCharacterTextSplitter（推荐）---")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        # 分割符优先级：先尝试段落，再句子，再逗号，最后字符
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    recursive_chunks = recursive_splitter.split_text(long_text)
    print(f"分割成 {len(recursive_chunks)} 个块")
    for i, chunk in enumerate(recursive_chunks[:3]):
        print(f"块 {i+1} ({len(chunk)} 字符): {chunk.strip()[:50]}...")
    
    print("\n💡 RecursiveCharacterTextSplitter 是最常用的分割器")
    print("   它会递归尝试不同的分割符，尽量保持文本的语义完整性")
    print()


def demo_split_documents():
    """2.2 分割 Document 对象"""
    print("=" * 60)
    print("2.2 分割 Document 对象（保留元数据）")
    print("=" * 60)
    
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 创建文档
    doc = Document(
        page_content="""
LangChain 是一个强大的框架，用于开发由大语言模型驱动的应用程序。

它的主要特点包括：
1. 模块化设计：提供可组合的组件，如 Prompts、LLMs、Chains 等
2. 链式调用：支持复杂的工作流编排
3. 丰富的集成：支持多种 LLM 提供商和外部工具

RAG（检索增强生成）是 LangChain 的核心应用场景之一，它通过检索外部知识来增强 LLM 的回答能力。
""",
        metadata={"source": "langchain_guide.txt", "author": "AI教程"}
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    
    # split_documents 会保留并传递元数据
    chunks = splitter.split_documents([doc])
    
    print(f"原文档分割成 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1}:")
        print(f"  内容: {chunk.page_content[:40]}...")
        print(f"  元数据: {chunk.metadata}")
    print()


# ============================================================
# 第三部分：向量嵌入 (Embeddings)
# ============================================================
"""
嵌入（Embedding）是将文本转换为数值向量的过程。
相似的文本会有相似的向量表示，这是语义搜索的基础。

常用嵌入模型：
┌────────────────────────────┬────────────────────────────────────────────┐
│ 模型                       │ 特点                                       │
├────────────────────────────┼────────────────────────────────────────────┤
│ OpenAI text-embedding-3    │ 高质量，需要 API Key，收费                 │
│ HuggingFace 模型           │ 开源免费，可本地运行                       │
│ 阿里云 DashScope           │ 国内访问快，支持中文                       │
│ 智谱 embedding             │ 国内服务，中文效果好                       │
└────────────────────────────┴────────────────────────────────────────────┘

💡 本教程使用 DashScope 的嵌入模型（与 qwen 同一平台）
"""

# 导入嵌入模型工厂
from embedding_factory import get_embeddings


def demo_embeddings():
    """3.1 嵌入模型演示"""
    print("=" * 60)
    print("3.1 嵌入模型 (Embeddings)")
    print("=" * 60)
    
    try:
        embeddings = get_embeddings()
        
        # 嵌入单个文本
        text = "LangChain 是一个 LLM 应用开发框架"
        vector = embeddings.embed_query(text)
        
        print(f"文本: {text}")
        print(f"向量维度: {len(vector)}")
        print(f"向量前5个值: {vector[:5]}")
        
        # 嵌入多个文本
        texts = [
            "LangChain 是一个框架",
            "RAG 是检索增强生成",
            "今天天气很好"
        ]
        vectors = embeddings.embed_documents(texts)
        
        print(f"\n批量嵌入 {len(texts)} 个文本")
        print(f"每个向量维度: {len(vectors[0])}")
        
        # 计算相似度（余弦相似度）
        import numpy as np
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        print("\n--- 语义相似度演示 ---")
        query = "什么是 LangChain？"
        query_vec = embeddings.embed_query(query)
        
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            sim = cosine_similarity(query_vec, vec)
            print(f"'{query}' vs '{text}': {sim:.4f}")
        
    except Exception as e:
        print(f"嵌入模型初始化失败: {e}")
        print("请确保设置了 DASHSCOPE_API_KEY 环境变量")
    print()


# ============================================================
# 第四部分：向量存储 (Vector Stores)
# ============================================================
"""
向量存储用于存储嵌入向量并支持相似度搜索。

常用向量存储：
┌────────────────────────────┬────────────────────────────────────────────┐
│ 向量存储                   │ 特点                                       │
├────────────────────────────┼────────────────────────────────────────────┤
│ Chroma                     │ 轻量级，支持持久化，适合开发和小规模       │
│ FAISS                      │ Facebook 开源，高性能，适合大规模          │
│ Pinecone                   │ 云服务，全托管，适合生产环境               │
│ Milvus                     │ 开源分布式，适合企业级应用                 │
│ Qdrant                     │ Rust 实现，高性能，支持过滤                │
└────────────────────────────┴────────────────────────────────────────────┘

💡 本教程使用 Chroma（轻量级，无需额外服务）
   安装: pip install langchain-chroma chromadb
"""

def demo_vector_store():
    """4.1 向量存储基础"""
    print("=" * 60)
    print("4.1 向量存储 (Vector Store) - Chroma")
    print("=" * 60)
    
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        
        embeddings = get_embeddings()
        
        # 准备文档
        docs = [
            Document(page_content="LangChain 是一个用于开发 LLM 应用的框架", 
                     metadata={"source": "intro", "topic": "langchain"}),
            Document(page_content="RAG 是检索增强生成，结合检索和生成技术", 
                     metadata={"source": "rag", "topic": "rag"}),
            Document(page_content="向量数据库用于存储和检索向量嵌入", 
                     metadata={"source": "vector", "topic": "database"}),
            Document(page_content="Prompt 工程是设计和优化提示词的过程", 
                     metadata={"source": "prompt", "topic": "prompt"}),
            Document(page_content="Agent 是能够自主决策和使用工具的 AI 系统", 
                     metadata={"source": "agent", "topic": "agent"}),
        ]
        
        # 创建向量存储（内存模式）
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="demo_collection"
        )
        
        print(f"已存储 {len(docs)} 个文档到向量库")
        
        # 相似度搜索
        print("\n--- 相似度搜索 ---")
        query = "什么是 RAG？"
        results = vectorstore.similarity_search(query, k=2)
        
        print(f"查询: {query}")
        for i, doc in enumerate(results):
            print(f"结果 {i+1}: {doc.page_content}")
            print(f"        元数据: {doc.metadata}")
        
        # 带分数的搜索
        print("\n--- 带相似度分数的搜索 ---")
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        
        for doc, score in results_with_scores:
            print(f"分数: {score:.4f} | {doc.page_content[:30]}...")
        
        return vectorstore
        
    except ImportError as e:
        print(f"需要安装依赖: {e}")
        print("pip install langchain-chroma chromadb")
        return None
    print()


def demo_vector_store_persistence():
    """4.2 向量存储持久化"""
    print("=" * 60)
    print("4.2 向量存储持久化")
    print("=" * 60)
    
    print("""
Chroma 持久化示例：

from langchain_chroma import Chroma

# 创建持久化向量存储
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化目录
    collection_name="my_collection"
)

# 后续加载已有的向量存储
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)

# 添加新文档
vectorstore.add_documents(new_docs)

# 删除文档（通过 ID）
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])

💡 提示：
- persist_directory 指定存储目录
- 重启程序后可以直接加载，无需重新嵌入
- 适合需要持久化的生产场景
""")
    print()


def demo_faiss_vector_store():
    """4.3 FAISS 向量存储"""
    print("=" * 60)
    print("4.3 FAISS 向量存储（高性能）")
    print("=" * 60)
    
    print("""
FAISS 是 Facebook 开源的高性能向量搜索库：

# 安装
pip install faiss-cpu  # CPU 版本
# pip install faiss-gpu  # GPU 版本（需要 CUDA）

from langchain_community.vectorstores import FAISS

# 从文档创建
vectorstore = FAISS.from_documents(docs, embeddings)

# 从文本创建
vectorstore = FAISS.from_texts(
    texts=["文本1", "文本2"],
    embedding=embeddings,
    metadatas=[{"source": "a"}, {"source": "b"}]
)

# 保存到本地
vectorstore.save_local("./faiss_index")

# 加载
vectorstore = FAISS.load_local(
    "./faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True  # 信任本地文件
)

# 合并两个向量存储
vectorstore1.merge_from(vectorstore2)

💡 FAISS vs Chroma：
- FAISS：更高性能，适合大规模数据
- Chroma：更易用，内置持久化，适合开发
""")
    print()


# ============================================================
# 第五部分：检索器 (Retrievers)
# ============================================================
"""
检索器是 RAG 的核心组件，负责根据查询找到相关文档。

检索器类型：
┌────────────────────────────┬────────────────────────────────────────────┐
│ 检索器                     │ 特点                                       │
├────────────────────────────┼────────────────────────────────────────────┤
│ VectorStoreRetriever       │ 基础向量检索，最常用                       │
│ MultiQueryRetriever        │ 生成多个查询变体，提高召回率               │
│ SelfQueryRetriever         │ 自动从问题中提取过滤条件                   │
│ ContextualCompression      │ 压缩检索结果，只保留相关部分               │
│ Retriever                  │                                            │
│ ParentDocumentRetriever    │ 检索小块，返回父文档                       │
│ EnsembleRetriever          │ 组合多个检索器                             │
└────────────────────────────┴────────────────────────────────────────────┘
"""

def demo_basic_retriever():
    """5.1 基础检索器"""
    print("=" * 60)
    print("5.1 基础检索器 (VectorStoreRetriever)")
    print("=" * 60)
    
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        
        embeddings = get_embeddings()
        
        docs = [
            Document(page_content="Python 是一种流行的编程语言，简单易学", 
                     metadata={"topic": "python"}),
            Document(page_content="JavaScript 是网页开发的核心语言", 
                     metadata={"topic": "javascript"}),
            Document(page_content="LangChain 是用 Python 开发的 LLM 框架", 
                     metadata={"topic": "langchain"}),
            Document(page_content="React 是一个 JavaScript 前端框架", 
                     metadata={"topic": "react"}),
        ]
        
        vectorstore = Chroma.from_documents(docs, embeddings)
        
        # 方式1：as_retriever() 转换为检索器
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # 搜索类型
            search_kwargs={"k": 2}     # 返回 top-k 结果
        )
        
        print("--- 基础相似度检索 ---")
        results = retriever.invoke("Python 编程")
        for doc in results:
            print(f"  - {doc.page_content}")
        
        # 方式2：MMR (最大边际相关性) - 增加结果多样性
        print("\n--- MMR 检索（增加多样性）---")
        retriever_mmr = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 2,
                "fetch_k": 4,      # 先获取更多候选
                "lambda_mult": 0.5 # 多样性参数 (0=最大多样性, 1=最大相关性)
            }
        )
        results = retriever_mmr.invoke("编程语言")
        for doc in results:
            print(f"  - {doc.page_content}")
        
        # 方式3：带过滤条件的检索
        print("\n--- 带元数据过滤的检索 ---")
        retriever_filtered = vectorstore.as_retriever(
            search_kwargs={
                "k": 2,
                "filter": {"topic": "python"}  # 只检索 topic=python 的文档
            }
        )
        results = retriever_filtered.invoke("框架")
        for doc in results:
            print(f"  - {doc.page_content} (topic: {doc.metadata['topic']})")
            
    except Exception as e:
        print(f"错误: {e}")
    print()


def demo_multi_query_retriever():
    """5.2 多查询检索器"""
    print("=" * 60)
    print("5.2 MultiQueryRetriever（提高召回率）")
    print("=" * 60)
    
    print("""
MultiQueryRetriever 工作原理：
1. 使用 LLM 将用户问题改写成多个不同角度的查询
2. 对每个查询分别检索
3. 合并去重所有结果

适用场景：
- 用户问题表述不清晰
- 需要从多个角度检索信息
- 提高召回率

示例代码：
─────────────────────────────────────────────────────────────
from langchain.retrievers.multi_query import MultiQueryRetriever

# 创建多查询检索器
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 使用
results = multi_retriever.invoke("LangChain 有什么用？")

# 查看生成的查询变体（调试用）
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)
─────────────────────────────────────────────────────────────

💡 优点：提高召回率，覆盖更多相关文档
💡 缺点：需要额外的 LLM 调用，增加延迟和成本
""")
    print()


def demo_contextual_compression():
    """5.3 上下文压缩检索器"""
    print("=" * 60)
    print("5.3 ContextualCompressionRetriever（精准提取）")
    print("=" * 60)
    
    print("""
ContextualCompressionRetriever 工作原理：
1. 先用基础检索器获取文档
2. 使用压缩器（LLM 或其他）提取与问题相关的部分
3. 返回压缩后的结果

适用场景：
- 检索到的文档太长
- 只需要文档中的特定部分
- 提高答案精确度

示例代码：
─────────────────────────────────────────────────────────────
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 创建压缩器（使用 LLM 提取相关内容）
compressor = LLMChainExtractor.from_llm(llm)

# 创建压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# 使用
results = compression_retriever.invoke("什么是 RAG？")
# 返回的文档只包含与问题相关的部分
─────────────────────────────────────────────────────────────

其他压缩器选项：
- LLMChainFilter: 使用 LLM 过滤不相关文档
- EmbeddingsFilter: 使用嵌入相似度过滤
- DocumentCompressorPipeline: 组合多个压缩器

💡 优点：提高精确度，减少无关信息
💡 缺点：需要额外的 LLM 调用
""")
    print()


def demo_self_query_retriever():
    """5.4 自查询检索器"""
    print("=" * 60)
    print("5.4 SelfQueryRetriever（自动提取过滤条件）")
    print("=" * 60)
    
    print("""
SelfQueryRetriever 工作原理：
1. 使用 LLM 分析用户问题
2. 自动提取语义查询和元数据过滤条件
3. 结合语义搜索和元数据过滤

适用场景：
- 文档有丰富的元数据（日期、类别、作者等）
- 用户问题包含过滤条件（如"2024年的文章"）

示例代码：
─────────────────────────────────────────────────────────────
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

# 定义元数据字段
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="文档发布年份",
        type="integer",
    ),
    AttributeInfo(
        name="category",
        description="文档类别，如 'tutorial', 'news', 'api'",
        type="string",
    ),
]

# 创建自查询检索器
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="技术文档",
    metadata_field_info=metadata_field_info,
)

# 使用 - LLM 会自动提取过滤条件
results = self_query_retriever.invoke("2024年的教程文档")
# 自动转换为: 语义搜索"教程" + 过滤 year=2024, category="tutorial"
─────────────────────────────────────────────────────────────

💡 优点：自动理解用户意图，智能过滤
💡 缺点：需要预定义元数据结构
""")
    print()


# ============================================================
# 第六部分：构建完整的 RAG 链
# ============================================================
"""
RAG 链的两种构建方式：
1. 手动 LCEL 链：完全控制，适合学习和自定义
2. create_retrieval_chain：官方推荐，自动处理文档格式
"""

def demo_rag_chain_manual():
    """6.1 手动构建 RAG 链（LCEL 方式）"""
    print("=" * 60)
    print("6.1 手动构建 RAG 链（LCEL 方式）")
    print("=" * 60)
    
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        embeddings = get_embeddings()
        
        # 准备知识库
        docs = [
            Document(page_content="LangChain 是一个用于开发 LLM 应用的开源框架，由 Harrison Chase 创建于 2022 年。"),
            Document(page_content="LangChain 的核心概念包括：Prompts、LLMs、Chains、Agents、Memory 和 Retrieval。"),
            Document(page_content="RAG（检索增强生成）通过检索外部知识来增强 LLM 的回答能力，减少幻觉。"),
            Document(page_content="LCEL（LangChain Expression Language）是 LangChain 的声明式编排语法，使用 | 管道符组合组件。"),
            Document(page_content="LangGraph 是 LangChain 团队开发的图状态机框架，用于构建复杂的 Agent 工作流。"),
        ]
        
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # RAG 提示模板
        prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题。如果上下文中没有相关信息，请说"根据提供的信息，我无法回答这个问题"。

上下文：
{context}

问题：{question}

回答：""")
        
        # 格式化检索到的文档
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 构建 RAG 链
        rag_chain = (
            {
                "context": retriever | format_docs,  # 检索 -> 格式化
                "question": RunnablePassthrough()     # 直接传递问题
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 测试
        questions = [
            "什么是 LangChain？",
            "RAG 有什么作用？",
            "LCEL 是什么？",
            "量子计算是什么？"  # 知识库中没有的问题
        ]
        
        for q in questions:
            print(f"\n问题: {q}")
            answer = rag_chain.invoke(q)
            print(f"回答: {answer}")
            
    except Exception as e:
        print(f"错误: {e}")
    print()


def demo_rag_chain_official():
    """6.2 使用 create_retrieval_chain（官方推荐）"""
    print("=" * 60)
    print("6.2 create_retrieval_chain（官方推荐方式）")
    print("=" * 60)
    
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        
        embeddings = get_embeddings()
        
        # 准备知识库
        docs = [
            Document(page_content="LangChain 是一个用于开发 LLM 应用的开源框架。"),
            Document(page_content="RAG 通过检索外部知识来增强 LLM 的回答能力。"),
            Document(page_content="向量数据库用于存储和检索向量嵌入。"),
        ]
        
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        
        # 系统提示词
        system_prompt = """你是一个问答助手。使用以下检索到的上下文来回答问题。
如果你不知道答案，就说不知道。保持回答简洁。

上下文：
{context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 创建文档处理链（stuff = 将所有文档塞入 prompt）
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # 创建检索链
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # 调用
        response = rag_chain.invoke({"input": "什么是 RAG？"})
        
        print(f"问题: 什么是 RAG？")
        print(f"回答: {response['answer']}")
        print(f"\n检索到的文档数: {len(response['context'])}")
        for i, doc in enumerate(response['context']):
            print(f"  文档 {i+1}: {doc.page_content[:50]}...")
            
    except ImportError as e:
        print(f"需要安装: {e}")
    except Exception as e:
        print(f"错误: {e}")
    print()


def demo_rag_with_history():
    """6.3 带对话历史的 RAG"""
    print("=" * 60)
    print("6.3 带对话历史的 RAG")
    print("=" * 60)
    
    try:
        from langchain.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage
        
        embeddings = get_embeddings()
        
        docs = [
            Document(page_content="LangChain 是一个 LLM 应用开发框架，支持 Python 和 JavaScript。"),
            Document(page_content="LangChain 的主要组件包括：Models、Prompts、Chains、Agents、Memory。"),
            Document(page_content="LangGraph 是 LangChain 团队开发的状态机框架，用于复杂 Agent。"),
        ]
        
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        
        # 1. 创建历史感知检索器（将对话历史融入检索）
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据对话历史，将用户的问题改写为独立的问题。"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # 2. 创建问答链
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "基于以下上下文回答问题：\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # 3. 组合成完整的 RAG 链
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # 模拟多轮对话
        chat_history = []
        
        # 第一轮
        q1 = "LangChain 是什么？"
        r1 = rag_chain.invoke({"input": q1, "chat_history": chat_history})
        print(f"用户: {q1}")
        print(f"AI: {r1['answer']}")
        
        chat_history.extend([
            HumanMessage(content=q1),
            AIMessage(content=r1['answer'])
        ])
        
        # 第二轮（使用代词"它"，需要理解上下文）
        q2 = "它有哪些主要组件？"
        r2 = rag_chain.invoke({"input": q2, "chat_history": chat_history})
        print(f"\n用户: {q2}")
        print(f"AI: {r2['answer']}")
        
    except Exception as e:
        print(f"错误: {e}")
    print()


def demo_rag_streaming():
    """6.4 流式输出的 RAG"""
    print("=" * 60)
    print("6.4 流式输出的 RAG")
    print("=" * 60)
    
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        
        embeddings = get_embeddings()
        
        docs = [
            Document(page_content="LangChain 是一个强大的 LLM 应用开发框架。"),
            Document(page_content="它提供了丰富的组件来构建复杂的 AI 应用。"),
        ]
        
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        
        prompt = ChatPromptTemplate.from_template("""
基于上下文回答问题：

上下文：{context}

问题：{question}

回答：""")
        
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("问题: 介绍一下 LangChain")
        print("流式回答: ", end="")
        
        # 流式输出
        for chunk in rag_chain.stream("介绍一下 LangChain"):
            print(chunk, end="", flush=True)
        print()
        
    except Exception as e:
        print(f"错误: {e}")
    print()


# ============================================================
# 第七部分：RAG 最佳实践
# ============================================================

def print_best_practices():
    """打印 RAG 最佳实践"""
    print("=" * 60)
    print("📊 RAG 最佳实践")
    print("=" * 60)
    print("""
    文档处理最佳实践：
    ─────────────────────────────────────────────────────────────
    1. 选择合适的 chunk_size：
       - 太小：丢失上下文
       - 太大：检索不精确
       - 推荐：500-1000 字符，根据文档类型调整
    
    2. 设置适当的 chunk_overlap：
       - 推荐：chunk_size 的 10-20%
       - 保持上下文连贯性
    
    3. 保留元数据：
       - 来源、页码、日期等
       - 便于过滤和溯源
    
    检索优化：
    ─────────────────────────────────────────────────────────────
    1. 选择合适的 k 值：
       - k 太小：可能遗漏相关信息
       - k 太大：引入噪音，增加成本
       - 推荐：3-5 个文档
    
    2. 使用 MMR 增加多样性：
       - 避免检索到重复内容
       - lambda_mult 控制多样性程度
    
    3. 结合元数据过滤：
       - 先过滤再检索，提高效率
       - 适合有明确过滤条件的场景
    
    Prompt 设计：
    ─────────────────────────────────────────────────────────────
    1. 明确指示基于上下文回答
    2. 处理"不知道"的情况
    3. 要求引用来源（如果需要）
    
    生产环境建议：
    ─────────────────────────────────────────────────────────────
    ┌─────────────────────┬─────────────────────────────────────┐
    │ 场景                │ 推荐方案                            │
    ├─────────────────────┼─────────────────────────────────────┤
    │ 开发/测试           │ Chroma（内存或本地持久化）          │
    │ 小规模生产          │ Chroma/FAISS + 本地持久化           │
    │ 大规模生产          │ Pinecone/Milvus/Qdrant（云服务）    │
    │ 企业级              │ 自建 Milvus/Elasticsearch           │
    └─────────────────────┴─────────────────────────────────────┘
    
    常见问题排查：
    ─────────────────────────────────────────────────────────────
    1. 检索结果不相关：
       - 检查嵌入模型是否适合你的语言/领域
       - 调整 chunk_size
       - 尝试不同的检索策略
    
    2. 回答不准确：
       - 检查检索到的文档是否正确
       - 优化 Prompt
       - 增加 k 值
    
    3. 性能问题：
       - 使用更高效的向量存储
       - 添加缓存
       - 异步处理
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n📚 第七课：RAG 基础 (检索增强生成)\n")
    
    print("\n" + "=" * 60)
    print("📚 第一部分：文档加载器")
    print("=" * 60)
    demo_document_loaders()
    demo_text_loader()
    demo_web_loader()
    demo_directory_loader()
    
    print("\n" + "=" * 60)
    print("📚 第二部分：文本分割器")
    print("=" * 60)
    demo_text_splitters()
    demo_split_documents()
    
    print("\n" + "=" * 60)
    print("📚 第三部分：向量嵌入")
    print("=" * 60)
    demo_embeddings()
    
    print("\n" + "=" * 60)
    print("📚 第四部分：向量存储")
    print("=" * 60)
    demo_vector_store()
    demo_vector_store_persistence()
    demo_faiss_vector_store()
    
    print("\n" + "=" * 60)
    print("📚 第五部分：检索器")
    print("=" * 60)
    demo_basic_retriever()
    demo_multi_query_retriever()
    demo_contextual_compression()
    demo_self_query_retriever()
    
    print("\n" + "=" * 60)
    print("📚 第六部分：构建 RAG 链")
    print("=" * 60)
    demo_rag_chain_manual()
    demo_rag_chain_official()
    demo_rag_with_history()
    demo_rag_streaming()
    
    print_best_practices()
    
    print("\n" + "=" * 60)
    print("📌 第七课总结")
    print("=" * 60)
    print("""
    RAG 完整流程
    ─────────────────────────────────────────────────────────────
    索引阶段：文档加载 → 文本分割 → 向量嵌入 → 存入向量库
    检索阶段：用户问题 → 向量检索 → 构建 Prompt → LLM 生成
    
    核心组件
    ─────────────────────────────────────────────────────────────
    Document Loaders  : TextLoader, WebBaseLoader, DirectoryLoader
    Text Splitters    : RecursiveCharacterTextSplitter（推荐）
    Embeddings        : DashScope, OpenAI, HuggingFace
    Vector Stores     : Chroma（开发）, FAISS/Pinecone（生产）
    Retrievers        : VectorStoreRetriever, MultiQuery, SelfQuery
    
    RAG 链构建方式
    ─────────────────────────────────────────────────────────────
    手动 LCEL 链          : 完全控制，适合学习和自定义
    create_retrieval_chain: 官方推荐，自动处理文档格式
    带历史的 RAG          : 支持多轮对话，理解上下文
    
    下一课预告：第八课 Agents（智能代理）
    ─────────────────────────────────────────────────────────────
    学习如何构建能够自主决策、使用工具的 AI Agent
    """)


if __name__ == "__main__":
    main()

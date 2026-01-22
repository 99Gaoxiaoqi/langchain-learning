"""
Embedding 工厂模块 - 统一的嵌入模型兼容层
支持通过环境变量配置切换不同的嵌入模型提供商
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 预置的嵌入模型配置
EMBEDDING_CONFIGS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["text-embedding-v3", "text-embedding-v2", "text-embedding-v1"],
        "default_model": "text-embedding-v3",
        "env_key": "DASHSCOPE_API_KEY",
        "dimensions": 1024,  # text-embedding-v3 默认维度
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        "default_model": "text-embedding-3-small",
        "env_key": "OPENAI_API_KEY",
        "dimensions": 1536,
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["embedding-3", "embedding-2"],
        "default_model": "embedding-3",
        "env_key": "ZHIPU_API_KEY",
        "dimensions": 2048,
    },
}


def get_embeddings(
    provider: str = None,
    model: str = None,
    **kwargs
):
    """
    获取 Embedding 模型实例的统一入口
    
    Args:
        provider: 模型提供商 (dashscope/openai/zhipu)
                  默认从 EMBEDDING_PROVIDER 环境变量读取，未设置则用 dashscope
        model: 模型名称，默认使用该提供商的默认模型
        **kwargs: 其他参数（如 dimensions）
    
    Returns:
        Embeddings 实例
    
    Example:
        # 使用默认配置（环境变量）
        embeddings = get_embeddings()
        
        # 指定提供商
        embeddings = get_embeddings(provider="openai")
        
        # 指定提供商和模型
        embeddings = get_embeddings(provider="dashscope", model="text-embedding-v2")
    """
    # 确定提供商
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "dashscope")
    provider = provider.lower()
    
    if provider not in EMBEDDING_CONFIGS:
        raise ValueError(f"不支持的提供商: {provider}，可选: {list(EMBEDDING_CONFIGS.keys())}")
    
    config = EMBEDDING_CONFIGS[provider]
    
    # 获取 API Key
    api_key = os.getenv(config["env_key"])
    if not api_key:
        raise ValueError(f"请设置环境变量 {config['env_key']}")
    
    # 确定模型
    model = model or config["default_model"]
    
    # DashScope 使用专门的 Embeddings 类（OpenAI 兼容接口对 embedding 支持不完整）
    if provider == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
            **kwargs
        )
    
    # 其他提供商使用 OpenAI 兼容接口
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=model,
        base_url=config["base_url"],
        api_key=api_key,
        **kwargs
    )


def list_providers():
    """列出所有支持的嵌入模型提供商"""
    print("支持的嵌入模型提供商：")
    print("-" * 50)
    for name, config in EMBEDDING_CONFIGS.items():
        print(f"\n📦 {name}")
        print(f"   环境变量: {config['env_key']}")
        print(f"   默认模型: {config['default_model']}")
        print(f"   向量维度: {config['dimensions']}")
        print(f"   可用模型: {', '.join(config['models'])}")


if __name__ == "__main__":
    list_providers()

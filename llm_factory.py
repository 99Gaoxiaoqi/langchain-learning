"""
LLM 工厂模块 - 统一的模型兼容层
支持通过环境变量配置切换不同的大模型提供商
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 预置的模型配置
MODEL_CONFIGS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen2.5-32b-instruct"],
        "default_model": "qwen2.5-32b-instruct",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "default_model": "moonshot-v1-8k",
        "env_key": "MOONSHOT_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4", "glm-4-flash", "glm-4-plus"],
        "default_model": "glm-4-flash",
        "env_key": "ZHIPU_API_KEY",
    },
    "openai": {
        "base_url": "https://api.zhizengzeng.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}


def get_llm(
    provider: str = None,
    model: str = None,
    temperature: float = 0.7,
    **kwargs
) -> ChatOpenAI:
    """
    获取 LLM 实例的统一入口
    
    Args:
        provider: 模型提供商 (qwen/deepseek/moonshot/zhipu/openai)
                  默认从 LLM_PROVIDER 环境变量读取，未设置则用 qwen
        model: 模型名称，默认使用该提供商的默认模型
        temperature: 温度参数
        **kwargs: 其他 ChatOpenAI 支持的参数
    
    Returns:
        ChatOpenAI 实例
    
    Example:
        # 使用默认配置（环境变量）
        llm = get_llm()
        
        # 指定提供商
        llm = get_llm(provider="deepseek")
        
        # 指定提供商和模型
        llm = get_llm(provider="qwen", model="qwen-max")
    """
    # 确定提供商
    provider = provider or os.getenv("LLM_PROVIDER", "qwen")
    provider = provider.lower()
    
    if provider not in MODEL_CONFIGS:
        raise ValueError(f"不支持的提供商: {provider}，可选: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[provider]
    
    # 获取 API Key
    api_key = os.getenv(config["env_key"])
    if not api_key:
        raise ValueError(f"请设置环境变量 {config['env_key']}")
    
    # 确定模型
    model = model or config["default_model"]
    
    return ChatOpenAI(
        model=model,
        base_url=config["base_url"],
        api_key=api_key,
        temperature=temperature,
        **kwargs
    )


def list_providers():
    """列出所有支持的提供商及其模型"""
    print("支持的模型提供商：")
    print("-" * 50)
    for name, config in MODEL_CONFIGS.items():
        print(f"\n📦 {name}")
        print(f"   环境变量: {config['env_key']}")
        print(f"   默认模型: {config['default_model']}")
        print(f"   可用模型: {', '.join(config['models'])}")


if __name__ == "__main__":
    list_providers()

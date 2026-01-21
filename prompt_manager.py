"""
PromptManager - 企业级提示词管理模块

功能：
1. 模板注册与管理
2. 意图路由
3. 版本管理与 A/B 测试
4. 支持多种存储后端（内存/文件/数据库）
"""
import json
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@dataclass
class PromptConfig:
    """提示词配置"""
    intent: str                          # 意图标识
    name: str                            # 显示名称
    system_prompt: str                   # 系统提示词
    version: str = "1.0"                 # 版本号
    weight: int = 100                    # A/B 测试权重 (0-100)
    enabled: bool = True                 # 是否启用
    include_history: bool = False        # 是否包含历史消息
    metadata: dict = field(default_factory=dict)  # 扩展元数据


class PromptManager:
    """
    企业级提示词管理器
    
    使用示例：
        manager = PromptManager()
        manager.register("presale", "售前顾问", "你是售前顾问...")
        
        prompt = manager.get_prompt("presale")
        chain = prompt | llm
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self._prompts: dict[str, list[PromptConfig]] = {}  # intent -> [configs]
        self._storage_path = Path(storage_path) if storage_path else None
        
        if self._storage_path and self._storage_path.exists():
            self._load_from_file()
    
    # ============================================================
    # 注册与管理
    # ============================================================
    
    def register(
        self,
        intent: str,
        name: str,
        system_prompt: str,
        version: str = "1.0",
        weight: int = 100,
        include_history: bool = False,
        **metadata
    ) -> "PromptManager":
        """注册提示词模板"""
        config = PromptConfig(
            intent=intent,
            name=name,
            system_prompt=system_prompt,
            version=version,
            weight=weight,
            include_history=include_history,
            metadata=metadata,
        )
        
        if intent not in self._prompts:
            self._prompts[intent] = []
        
        # 检查是否已存在相同版本
        for i, existing in enumerate(self._prompts[intent]):
            if existing.version == version:
                self._prompts[intent][i] = config
                return self
        
        self._prompts[intent].append(config)
        return self
    
    def unregister(self, intent: str, version: Optional[str] = None) -> bool:
        """注销提示词模板"""
        if intent not in self._prompts:
            return False
        
        if version:
            self._prompts[intent] = [
                p for p in self._prompts[intent] if p.version != version
            ]
        else:
            del self._prompts[intent]
        return True
    
    def list_intents(self) -> list[str]:
        """列出所有意图"""
        return list(self._prompts.keys())
    
    def get_config(self, intent: str, version: Optional[str] = None) -> Optional[PromptConfig]:
        """获取配置（支持 A/B 测试权重选择）"""
        if intent not in self._prompts:
            return None
        
        configs = [c for c in self._prompts[intent] if c.enabled]
        if not configs:
            return None
        
        # 指定版本
        if version:
            for c in configs:
                if c.version == version:
                    return c
            return None
        
        # A/B 测试：按权重随机选择
        if len(configs) == 1:
            return configs[0]
        
        total_weight = sum(c.weight for c in configs)
        r = random.randint(1, total_weight)
        current = 0
        for c in configs:
            current += c.weight
            if r <= current:
                return c
        return configs[0]
    
    # ============================================================
    # 获取 Prompt 模板
    # ============================================================
    
    def get_prompt(
        self,
        intent: str,
        version: Optional[str] = None
    ) -> Optional[ChatPromptTemplate]:
        """获取 ChatPromptTemplate"""
        config = self.get_config(intent, version)
        if not config:
            return None
        
        messages = [("system", config.system_prompt)]
        
        if config.include_history:
            messages.append(MessagesPlaceholder(variable_name="history"))
        
        messages.append(("human", "{question}"))
        
        return ChatPromptTemplate.from_messages(messages)
    
    def get_chain(self, intent: str, llm, version: Optional[str] = None):
        """获取完整的 Chain"""
        prompt = self.get_prompt(intent, version)
        if not prompt:
            raise ValueError(f"未找到意图: {intent}")
        return prompt | llm
    
    # ============================================================
    # 路由
    # ============================================================
    
    def route(self, intent: str, llm, default_intent: Optional[str] = None):
        """
        根据意图路由到对应的 Chain
        
        使用示例：
            chain = manager.route(detected_intent, llm, default_intent="general")
            response = chain.invoke({"question": "..."})
        """
        prompt = self.get_prompt(intent)
        if not prompt and default_intent:
            prompt = self.get_prompt(default_intent)
        if not prompt:
            raise ValueError(f"未找到意图: {intent}，且无默认意图")
        return prompt | llm
    
    # ============================================================
    # 持久化
    # ============================================================
    
    def save(self, path: Optional[str] = None):
        """保存到文件"""
        save_path = Path(path) if path else self._storage_path
        if not save_path:
            raise ValueError("未指定存储路径")
        
        data = {}
        for intent, configs in self._prompts.items():
            data[intent] = [
                {
                    "intent": c.intent,
                    "name": c.name,
                    "system_prompt": c.system_prompt,
                    "version": c.version,
                    "weight": c.weight,
                    "enabled": c.enabled,
                    "include_history": c.include_history,
                    "metadata": c.metadata,
                }
                for c in configs
            ]
        
        save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    
    def _load_from_file(self):
        """从文件加载"""
        if not self._storage_path or not self._storage_path.exists():
            return
        
        data = json.loads(self._storage_path.read_text())
        for intent, configs in data.items():
            for c in configs:
                self.register(**c)
    
    # ============================================================
    # 便捷方法
    # ============================================================
    
    def __contains__(self, intent: str) -> bool:
        return intent in self._prompts
    
    def __len__(self) -> int:
        return len(self._prompts)


# ============================================================
# 预置模板（可选）
# ============================================================

def create_default_manager() -> PromptManager:
    """创建带有预置模板的管理器"""
    manager = PromptManager()
    
    # 通用助手
    manager.register(
        intent="general",
        name="通用助手",
        system_prompt="你是一个友好的AI助手，请用简洁专业的方式回答用户问题。",
    )
    
    # 售前顾问
    manager.register(
        intent="presale",
        name="售前顾问",
        system_prompt="""你是一位专业的售前顾问，负责：
1. 介绍产品功能和优势
2. 解答价格和套餐问题
3. 提供产品对比和推荐
4. 引导客户完成购买决策

请保持热情专业的态度，突出产品价值。""",
    )
    
    # 售后客服
    manager.register(
        intent="aftersale",
        name="售后客服",
        system_prompt="""你是一位耐心的售后客服，负责：
1. 处理退换货请求
2. 解答使用问题
3. 收集用户反馈
4. 处理投诉和建议

请保持耐心和同理心，优先解决用户问题。""",
    )
    
    # 技术支持
    manager.register(
        intent="technical",
        name="技术支持",
        system_prompt="""你是一位专业的技术支持工程师，负责：
1. 解答技术问题
2. 提供故障排查指导
3. 给出代码示例和解决方案
4. 解释技术概念

请用清晰准确的技术语言回答，必要时提供代码示例。""",
    )
    
    return manager


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    from llm_factory import get_llm
    
    print("\n🚀 PromptManager 企业级提示词管理演示\n")
    
    llm = get_llm()
    manager = create_default_manager()
    
    print(f"已注册意图: {manager.list_intents()}\n")
    
    # 测试不同意图
    test_cases = [
        ("presale", "你们的产品有什么优势？"),
        ("aftersale", "我想退货，怎么操作？"),
        ("technical", "Python装饰器怎么用？"),
        ("general", "今天天气怎么样？"),
    ]
    
    for intent, question in test_cases:
        print(f"{'=' * 50}")
        print(f"意图: {intent}")
        print(f"问题: {question}")
        print("-" * 50)
        
        chain = manager.route(intent, llm, default_intent="general")
        response = chain.invoke({"question": question})
        print(f"回答: {response.content}\n")
    
    # A/B 测试演示
    print("=" * 50)
    print("📊 A/B 测试演示")
    print("=" * 50)
    
    # 注册两个版本
    manager.register(
        intent="greeting",
        name="问候语-正式版",
        system_prompt="你是一个正式的AI助手，用专业的语气回答。",
        version="formal",
        weight=50,
    )
    manager.register(
        intent="greeting",
        name="问候语-轻松版",
        system_prompt="你是一个轻松的AI助手，用活泼的语气回答，可以加emoji。",
        version="casual",
        weight=50,
    )
    
    print("运行5次，观察 A/B 测试效果：")
    for i in range(5):
        config = manager.get_config("greeting")
        print(f"  第{i+1}次选中: {config.name} (版本: {config.version})")

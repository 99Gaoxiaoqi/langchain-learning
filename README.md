# LangChain 学习项目

支持多模型提供商（通义千问/DeepSeek/Moonshot/智谱/OpenAI）的 LangChain 学习项目。

## 环境配置

1. 复制 `.env.example` 为 `.env`
2. 配置你要使用的模型提供商 API Key
3. 设置 `LLM_PROVIDER` 选择默认提供商

```bash
# .env 示例
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=sk-xxx      # 通义千问
OPENAI_API_KEY=sk-xxx         # OpenAI
# DEEPSEEK_API_KEY=sk-xxx     # DeepSeek
# MOONSHOT_API_KEY=sk-xxx     # Moonshot
# ZHIPU_API_KEY=xxx           # 智谱
```

## 运行示例

```bash
# 运行示例
uv run python 01_basic_chat.py
uv run python 02_prompt_templates.py
uv run python 03_output_parsers.py
uv run python 04_chains.py
```

## 项目结构

```
├── llm_factory.py          # 统一 LLM 工厂（多模型兼容层）
├── prompt_manager.py       # 企业级提示词管理器
├── 01_basic_chat.py        # 第1课：基础对话
├── 02_prompt_templates.py  # 第2课：Prompt 模板
├── 03_output_parsers.py    # 第3课：输出解析器
├── 04_chains.py            # 第4课：链
├── 05_memory.py            # 第5课：记忆
├── 06_tools.py             # 第6课：工具
├── 07_rag_basics.py        # 第7课：RAG
└── 08_agents.py            # 第8课：Agent
```

## 学习内容

### 第1课：基础对话 (`01_basic_chat.py`)
- 4 种消息格式：字符串/元组/字典/Message对象
- 6 种调用方式：invoke/stream/batch + 异步版本
- 统一 LLM 工厂：一行代码切换模型

### 第2课：Prompt 模板 (`02_prompt_templates.py`)
- 3 种模板格式：f-string/jinja2/mustache
- ChatPromptTemplate 多角色模板
- Few-shot 示例学习
- MessagesPlaceholder 历史消息
- 企业级 PromptManager（模板注册/路由/A-B测试）

### 第3课：输出解析器 (`03_output_parsers.py`)
- StrOutputParser / JsonOutputParser
- PydanticOutputParser（推荐）
- with_structured_output（OpenAI 完整支持，通义千问需 prompt 含 json）
- OutputFixingParser 自动修复
- 模型对比：OpenAI vs 通义千问

### 第4课：链 (`04_chains.py`)
- LCEL 基础：管道操作符 `|`、pipe() 方法、RunnableSequence
- 核心组件：
  - `RunnableLambda`：把普通函数包装成 Runnable
  - `RunnablePassthrough`：数据透传，assign() 追加字段（常用于 RAG）
  - `RunnableParallel`：并行执行多个链
  - `RunnableBranch`：条件路由分支
  - `@chain` 装饰器：用函数方式定义复杂链
- 调用方式：invoke / stream / batch + 异步版本（ainvoke / astream / abatch）
- 错误处理：with_retry（自动重试）/ with_fallbacks（降级备选）
- 配置调试：bind（绑定参数）/ with_config（tags/metadata）/ input_schema
- 工具函数：`itemgetter` 从字典提取字段

### 第5-8课
待完成...

## 企业级组件

### llm_factory.py
统一的模型兼容层，支持一行代码切换不同提供商：
```python
from llm_factory import get_llm

llm = get_llm()                              # 使用环境变量配置
llm = get_llm(provider="openai")             # 指定提供商
llm = get_llm(provider="qwen", model="qwen-max")  # 指定模型
```

### prompt_manager.py
企业级提示词管理：
```python
from prompt_manager import create_default_manager

manager = create_default_manager()
chain = manager.route(intent, llm, default_intent="general")
```

## 注意事项

- 通义千问使用 `with_structured_output` 需要 prompt 中包含 "json" 关键词
- 推荐国产模型使用 `PydanticOutputParser`，兼容性最好
- `OutputFixingParser` 在 `langchain_classic` 包中

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [阿里百炼文档](https://help.aliyun.com/zh/dashscope/)

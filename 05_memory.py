"""
第五课：对话记忆
学习目标：
1. 理解对话记忆的重要性
2. 使用 ConversationBufferMemory
3. 使用 ConversationSummaryMemory
4. 实现多轮对话
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# 存储会话历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def chat_with_memory():
    """带记忆的对话示例"""
    llm = ChatTongyi(model="qwen-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，记住用户告诉你的信息。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    # 添加消息历史
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # 模拟多轮对话
    config = {"configurable": {"session_id": "user_001"}}
    
    print("用户: 我叫小明，今年25岁")
    response1 = with_history.invoke(
        {"input": "我叫小明，今年25岁"},
        config=config
    )
    print("AI:", response1.content)
    
    print("\n用户: 我喜欢编程和打篮球")
    response2 = with_history.invoke(
        {"input": "我喜欢编程和打篮球"},
        config=config
    )
    print("AI:", response2.content)
    
    print("\n用户: 你还记得我的名字和爱好吗？")
    response3 = with_history.invoke(
        {"input": "你还记得我的名字和爱好吗？"},
        config=config
    )
    print("AI:", response3.content)

def manual_memory_demo():
    """手动管理记忆示例"""
    llm = ChatTongyi(model="qwen-turbo")
    
    # 手动维护消息历史
    messages = [
        ("system", "你是一个数学老师。")
    ]
    
    # 第一轮
    messages.append(("human", "1+1等于多少？"))
    prompt = ChatPromptTemplate.from_messages(messages)
    response = (prompt | llm).invoke({})
    messages.append(("ai", response.content))
    print("Q: 1+1等于多少？")
    print("A:", response.content)
    
    # 第二轮
    messages.append(("human", "那再乘以3呢？"))
    prompt = ChatPromptTemplate.from_messages(messages)
    response = (prompt | llm).invoke({})
    print("\nQ: 那再乘以3呢？")
    print("A:", response.content)

if __name__ == "__main__":
    print("=== 带记忆的对话 ===")
    chat_with_memory()
    print("\n=== 手动管理记忆 ===")
    manual_memory_demo()

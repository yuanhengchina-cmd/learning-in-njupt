三种消息管理策略
当对话变长超出 context window 时，有三种处理方案：
策略一：裁剪消息（Trim）— 保留最新 N 条
**原理**：每次调用模型**之前**，只保留最近几条消息，旧消息不写入 DB 删除，只是本次不传给模型
```
from langchain.messages import RemoveMessage  
from langgraph.graph.message import REMOVE_ALL_MESSAGES  
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents import create_agent, AgentState  
from langchain.agents.middleware import before_model  
from langgraph.runtime import Runtime  
from langchain_core.runnables import RunnableConfig  
from typing import Any  
  
#- `@before_model`：装饰器，把这个函数注册为"模型调用前的钩子
- 返回 `None` = 什么都不做；返回字典 = 用字典里的值更新状态
@before_model  # 在模型调用前触发  
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  
    """保留首条消息 + 最近 3~4 条，防止超出上下文窗口。"""  
    messages = state["messages"]  
  #消息数量 ≤ 3 条时，不需要裁剪，直接放行
    if len(messages) <= 3:  
        return None  # 消息数量少，无需处理  
  #保留**第一条消息**，因为它通常是用户的第一句话（含关键信息，如"我叫小明"）。
    first_msg = messages[0]  
    # 根据奇偶保持消息对的完整性（human/ai 成对）  
    recent = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]  
  
    return {  
        "messages": [  
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 先清空  
            first_msg,                               # 保留首条（通常是 system/首次问候）  
            *recent,                                 # 保留最近几条  
        ]  
    }  
  
  
agent = create_agent(  
    "deepseek-chat",  
    tools=[],  
    middleware=[trim_messages],  
    checkpointer=InMemorySaver(),  
  
)  
  
config: RunnableConfig = {"configurable": {"thread_id": "1"}}  
agent.invoke({"messages": "你好，我叫小明"}, config)  
agent.invoke({"messages": "写一首关于猫的短诗"}, config)  
agent.invoke({"messages": "换成写狗的"}, config)  
final = agent.invoke({"messages": "我叫什么名字？"}, config)  
final["messages"][-1].pretty_print()  
# → 你叫小明，你在最开始的时候告诉我的。
```
**在每次调用模型之前，自动裁剪消息列表，防止超出 context window**。我逐层拆解。
```
用户 invoke
    ↓
@before_model 触发 → trim_messages() 执行（裁剪消息）
    ↓
DeepSeek-Chat 收到裁剪后的消息
    ↓
AI 生成回复
    ↓
checkpointer 把完整消息存入内存
    ↓
返回结果
```
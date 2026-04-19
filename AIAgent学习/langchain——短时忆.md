**短时记忆（Short-term Memory）** 让 Agent 在**单次会话（thread）** 内记住对话历史。每个 thread 通过唯一的 `thread_id` 隔离，状态通过 **checkpointer** 持久化到存储介质中 。[docs.langchain](https://docs.langchain.com/oss/python/concepts/memory)

|概念|说明|
|---|---|
|**Thread**|一次对话会话，包含多轮交互|
|**Checkpointer**|负责保存/读取 Agent 状态的组件|
|**AgentState**|Agent 的状态对象，默认含 `messages` 字段|
|**Middleware**|在模型调用前后拦截并处理状态的钩子|
MemorySaver
**短时记忆的挑战**：对话越长，消息列表越大，可能超出 LLM 的 context window，导致性能下降、响应变慢、成本升高
uv add langgraph-checkpoint-postgres
```
from langchain.chat_models import init_chat_model  
model=init_chat_model(model="deepseek-chat")
```
基础用法：启用短时记忆
开发环境（内存存储）
create_agent中使用新的参数checkpoint填引入的inmemorysaver，表示使用的是内存型checkpoint；
agent.invoke里可以包含多条消息，但是在消息之后我们加入config的参数，都是加入messages的消息数组里（字典里套数组）

```
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "deepseek-chat",                    # ← 替换为 deepseek-chat
    tools=[],
    checkpointer=InMemorySaver(),       # 内存 checkpointer，进程结束后丢失
)

# 通过 thread_id 区分不同会话
config = {"configurable": {"thread_id": "session_001"}}
#阻塞式调用时传入消息和参数
agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫小明。"}]},
    config,
)
```
生产环境（PostgreSQL 持久化）
让 Agent 记住同一个会话中的所有对话历史，并把这些记录永久保存到 PostgreSQL 数据库，即使程序重启也不会丢失 。
- `create_agent`：一键创建 Agent，内部自动构建 LangGraph 状态图（节点、边、循环逻辑）
- `PostgresSaver`：负责把 Agent 状态序列化后写入/读取 PostgreSQL
连接字符串
`DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable" #                      ↑用户名  ↑密码   ↑主机    ↑端口  ↑数据库名`
告诉 PostgresSaver **去哪里存数据**，格式是标准 PostgreSQL 连接 URL 。
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
`with` 确保数据库连接用完后**自动关闭**，避免连接泄漏。`checkpointer` 是一个持久化存储对象，专门负责保存和恢复 Agent 状态。
checkpointer.setup()**只需第一次运行时执行一次**。在 PostgreSQL 中自动创建 LangGraph 所需的数据表
`agent = create_agent(         "deepseek-chat",    # 使用的 LLM 模型        tools=[],           # 工具列表（空代表纯对话）        checkpointer=checkpointer,  # 注入持久化存储    )`
`checkpointer` 注入后，Agent 每完成一步（收到消息、调用工具、生成回复）都会自动把状态写入 PostgreSQL 。
```
from langchain.agents import create_agent  
from langgraph.checkpoint.postgres import PostgresSaver  
  
  
DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"  
  
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:  
    checkpointer.setup()  
    agent = create_agent(  
        "deepseek-chat",  
        tools=[],  
        checkpointer=checkpointer,  
  
    )
```
💡 其他可选 checkpointer：SQLite（轻量本地）、Azure Cosmos DB（云端）。
（可能出现问题：，我这里的解决方案是使用内存版本）
![[Pasted image 20260415182725.png]]
![[Pasted image 20260415184340.png]]
```

```
扩展 Agent 状态
默认 `AgentState` 只有 `messages`，可以通过继承添加自定义字段：

小练习：
```
from langchain.agents import create_agent  
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.messages import HumanMessage,AIMessage  
agent=create_agent(  
    model="deepseek-chat",  
    tools=[],  
    checkpointer=InMemorySaver(),  
)  
config1 = {"configurable": {"thread_id": "session_001"}}  
config2 = {"configurable": {"thread_id": "session_002"}}  
response1=agent.invoke(  
    {"messages": [HumanMessage("i am jim"),  
                  AIMessage("hello"),  
                  HumanMessage("Who am i")  
                  ]},config1  
)
print(response1)
```

```
response2=agent.invoke(  
    {"messages":[{"role":"user","content":"who am i"}]  
},config2  
)  
print(response2)
```

---
扩展 Agent 状态
默认 `AgentState` 只有 `messages`，可以通过继承添加自定义字段：
```
from langchain.agents import create_agent, AgentState  
from langgraph.checkpoint.memory import InMemorySaver  
  
class CustomAgentState(AgentState):  
    user_id: str  
    preferences: dict   # 存储用户偏好  
  
agent = create_agent(  
    "deepseek-chat",  
    tools=[],  
    state_schema=CustomAgentState,    # ← 注入自定义状态  
    checkpointer=InMemorySaver(),  
  
)  
  
result = agent.stream(  
    {  
        "messages": [{"role": "user", "content": "你好"}],  
        "user_id": "user_42",  
        "preferences": {"language": "zh", "theme": "dark"},  
    },  
    {"configurable": {"thread_id": "1"}},stream_mode="messages"  
)
for chunk,metadata in result:  
    print(chunk.content,end="",flush=True)
```
实例使用
```
from langchain.tools import tool, ToolRuntime  
from langchain.agents import create_agent, AgentState  
  
class CustomAgentState(AgentState):  
    user_id: str  
    preferences: dict  
  
@tool  
def get_user_profile(runtime: ToolRuntime) -> str:  
    """查询用户个人信息。"""  
    user_id = runtime.state["user_id"]           # ← 从状态里读取  
    prefs = runtime.state["preferences"]         # ← 从状态里读取  
    theme = prefs.get("theme", "light")  
    return f"用户ID: {user_id}，偏好主题: {theme}"  
  
agent = create_agent(  
    "deepseek-chat",  
    tools=[get_user_profile],  
    state_schema=CustomAgentState,  
)  
  
result = agent.invoke({  
    "messages": [{"role": "user", "content": "查询我的个人信息"}],  
    "user_id": "user_42",  
    "preferences": {"theme": "dark"},  
})  
print(result["messages"][-1].content)  
# → 用户ID: user_42，偏好主题: dark
```

```
from langchain.tools import tool, ToolRuntime  
from langchain.agents import create_agent, AgentState  
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_openai import ChatOpenAI  
  
  
  
class CustomAgentState(AgentState):  
    user_id: str  
    preferences: dict  
  
@tool  
def get_user_profile(runtime: ToolRuntime) -> str:  
    """查询用户个人信息。"""  
    user_id = runtime.state["user_id"]  
    prefs = runtime.state["preferences"]  
    theme = prefs.get("theme", "light")  
    return f"用户ID: {user_id}，偏好主题: {theme}"  
  
agent = create_agent(  
    model="deepseek-chat",  
    tools=[get_user_profile],  
    state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),       # ← 加了 checkpointer 才能用 config)  
  
config = {"configurable": {"thread_id": "1"}}  # ← config 定义  
  
result = agent.invoke(  
    {  
        "messages": [{"role": "user", "content": "查询我的个人信息"}],  
        "user_id": "user_42",  
        "preferences": {"theme": "dark"},  
    },  
    config,   # ← config 放第二个参数  
)  
print(result["messages"][-1].content)
```
![[Pasted image 20260415200910.png]]

`CustomAgentState` 里的字段是**每次调用时注入的临时上下文**（比如当前登录用户的信息），而 `thread_id` 对应的消息历史才是真正跨轮次持久保存的内容
![[Pasted image 20260415202314.png]]
解决：## 换 thread_id（推荐，最符合实际场景），不同的"登录会话"用不同的 thread_id
```
from langchain.tools import tool, ToolRuntime  
from langchain.agents import create_agent, AgentState  
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.messages import  HumanMessage  
from langchain_openai import ChatOpenAI  
  
  
  
class CustomAgentState(AgentState):  
    user_id: str  
    preferences: dict  
  
@tool  
def get_user_profile(runtime: ToolRuntime) -> str:  
    """查询用户个人信息。"""  
    user_id = runtime.state["user_id"]  
    prefs = runtime.state["preferences"]  
    theme = prefs.get("theme", "light")  
    return f"用户ID: {user_id}，偏好主题: {theme}"  
  
agent = create_agent(  
    model="deepseek-chat",  
    tools=[get_user_profile],  
    state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),       # ← 加了 checkpointer 才能用 config)  
  
config1 = {"configurable": {"thread_id": "1"}}  # ← config 定义  
config2 = {"configurable": {"thread_id": "2"}}  
config3 = {"configurable": {"thread_id": "1_3"}}  
  
result1 = agent.invoke(  
    {  
        "messages": [{"role": "user", "content": "查询我的个人信息"},HumanMessage("我的主题是不是dark")],  
        "user_id": "user_01",  
        "preferences": {"theme": "dark"},  
    },  
    config1,   # ← config 放第二个参数  
)  
print(result1["messages"][-1].content)  
  
result2 = agent.invoke(  
    {  
        "messages": [{"role": "user", "content": "查询我的个人信息"}],  
        "user_id": "user_02",  
        "preferences": {"theme": "bright"},  
    },  
    config2,   # ← config 放第二个参数  
)  
print(result2["messages"][-1].content)  
  
  
result3 = agent.invoke(  
    {  
        "messages": [{"role": "user", "content": "查询我的个人信息"},HumanMessage("我的主题是不是dark")],  
        "user_id": "user_01",  
        "preferences": {"theme": "bright"},  
    },  
    config3,   # ← config 放第二个参数  
)  
print(result3["messages"][-1].content)
```
![[Pasted image 20260415202632.png]]
---



```
一般的消息长这样，消息数组
response=model.invoke([
{"role":"system",
 "content":"你将扮演一个学生"},
 {"role":"user",
 "content":"你是谁？"},
 {"role":"assistant",
 "content":"我是..."},
 {"role":"user",
 "content":"你上课记笔记没有“
 }
])
```
数组，里面是字典，形式统一，role表示角色，content表示内容
role中system表示系统消息，或者是系统的背景、说明
user表示用户消息，指的是用户像模型提出的问题、消息
asistant表示模型返回给我们的回答，响应
这些角色是我们划分消息类型的依据，尽管这些角色不同，但是格式是统一。
但是这样过于低效，langchain对他们进行封装成BaseMessage，并准备了多个BaseMessage的子类对应不同的角色类型的消息
SystemMessage对应role:system的消息
HumanMessage对应role:user的消息
AIMessage对应role:assistant的消息
ToolMessage对应role:tool的消息（工具调用时的消息，工具调用时产生的结果）
这些类可以接受消息内容的参数以及其他参数

```
from langchain.agents import create_agent  
from langchain.tools import tool  
from langchain.messages import SystemMessage,AIMessage,HumanMessage  
#define tool  
def get_weather(location:str)->str:  
    """  
    Get the weather in a given location.    Args:     location:city_name or coordinates    """    return f"current weather is {location} is sunny"  
#创建Agent,同时把工具传给Agent  
#带有工具的智能体  
agent=create_agent(model="deepseek-chat",tools=[get_weather])  
  
#调用Agent，发送消息  
response= agent.invoke(  
    {"messages":[  
        # {"role":"system","content":"你是一名大学生"},  
        # {"role":"user","content":"你来自哪个学校"},  
        # {"role":"assistant","content":"尊敬的评委，您好，我是来自南京邮电大学的考生"},  
        # {"role":"user","content":"南京的天气怎么样"},  
        SystemMessage(content="请使用工具来获取天气信息"),  
        HumanMessage("我是用户"),  
        AIMessage("我是查询系统"),  
        HumanMessage("南京天气怎么样"),  
  
  
    ]  
    }  
)  
print(response)
```
![[Pasted image 20260413204717.png]]
为了更直观展示我们使用pretty_print
```
for message in response['messages']:  
    message.pretty_print()
```
![[Pasted image 20260413204815.png]]
这样先打印出他的角色或者类型，系统背景、用户提问（第一轮对话）
![[Pasted image 20260413204848.png]]
不仅是返回“我来帮您查询南京的天气情况。”还返回“Tool Calls”（以及“南京”等参数），Agent帮我们调用get_weather得到的结果（包括传入的参数）被封装成toolmessage，把toolmessage工具执行的结果再返回给模型。模型根据这个再次回答我们。“根据查询结果，南京当前的天气是晴天。”
封装的好处：使用时更清晰、简洁；定义公用的方法，可以便于我们调试、查看（使用pretty_look查看)
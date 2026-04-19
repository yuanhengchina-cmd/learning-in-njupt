langchain提供了一个create_agent函数用来创建智能体，当然create_agent时可以指定模型或者工具
-1、使用初始化好的模型对象
-2、使用模型名称，让langchain自动初始化模型

1、使用初始化好的模型对象
```
from langchain.chat_models import init_chat_model  
model = init_chat_model(  
    model="deepseek-chat"  
)
from langchain.agents import create_agent  
agent = create_agent(model=model)
```

2、使用模型名称（直接指定模型的名称），让langchain自动初始化模型

```
agent=create_agent(model="deepseek-chat")
```
和init_chat_model类似，同样是根据模型名称推断出模型，并帮我们完成初始化，只不过相对简化，不用先初始化model，直接在智能体来制定名称，直接帮我们调init_chat_model帮我们完成这个任务，只有能够根据模型名称推断的才可以。（int_chat_model那种）
![[Pasted image 20260413165228.png]]

调用智能体
-1、invoke阻塞式调用
-2、stream流式访问
智能体调用时需要传入一个dict，其中必须包含一个message字段，也就是消息列表
不同的是他不在穿字符串了，他传的是对象，对象的属性是message消息数组
```
response=agent.invoke({"messages":[{"role":"user",  
                                  "content":"你是谁？" }]})  
print(response)
```
![[Pasted image 20260413165936.png]]返回的是一个对象，是一个消息数组messages，包含humanmessage和AImessage，不仅有我们发给ai的信息，还有ai返回给我们的信息，这也是智能体区别于正常ai调用的差异，相当于我们传入的对象message数组里包含user消息，他将ai的消息给我们拼接进去，作为结果返回（相当于也是用了我们发给他的信息——问题，把问题和结果一起返回）
流式调用

```
message=agent.stream(  
    {"messages":[{"role":"user","content":"你是谁？"}]},  
     stream_mode="messages"  
)  
print(type(message))
```
stream_mode是流的一种模式，例如想让agent一个个输出文字采用messages
<class 'generator'>种类也是生成器
```
for token,metadata in messages:  
    if token.content:  
        print(token.content,end="",flush=True)
```
token还是片段，metadata是片段的元信息，是元数据（这个与阻塞式不同），同时加入了一个判断，没有数据不打印
![[Pasted image 20260413192856.png]]
据我观察，似乎只有stream流式才有这种只打印content的功能，阻塞式片段不含content属性
在直接使用模型调用时传的是字符串或者是消息数组，智能体调用时先传一个对象，里面有messages属性，然后才是消息数组。
流式调用除了传messages数组以外还要传一个stream_mode模式

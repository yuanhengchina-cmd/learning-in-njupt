我们正常是调用openai的SDK
langchain调用模型只需要使用model中的invoke方法，传入消息数组，和之前的OPENAI的SDK中的Message没什么区别，langchain对他还做了一定的简化，因为很多情况下是不用到stream(控制，阻塞式调用、流式调用）；还可以直接使用response=model.invoke("你是谁")，langchain默认将传入的字符串作为user的消息内容，会帮我们自动组成消息数组
(正常是
```
response=model.invoke([
{"role":"system",
"content":"你将扮演火箭队的武藏"
},
{"role":"user",
"content":"你是谁？"
}
])
```
)
langchain中是通过方法名称控制流式或者是非流式的，例如invoke调用就是一种阻塞式调用。response=model.invoke("你是谁")；stream调用是流式调用response=model.stream("你是谁")
使用：
先初始化模型
```
from dotenv import load_dotenv  
load_dotenv()  
from langchain_community.chat_models.tongyi import ChatTongyi  
model = ChatTongyi(  
    model="qwen-max"  
)
```
再通过invoke访问模型，需要阻塞等待模型生成结果，等待时间长
传一个简单字符串，作为user发送的问题拼接到message数组（原openai的SDK)
```
response=model.invoke("你是谁？")
```
![[Pasted image 20260413152445.png]]![[Pasted image 20260413152523.png]]
与模型相关的内容是与我们初始化模型的操作相关
![[Pasted image 20260413152843.png]]
```
from langchain.chat_models import init_chat_model  
model = init_chat_model(  
    model="deepseek-chat"  
)  
response=model.invoke("你是谁?")  
print(response)
```
这就是通用框架的优势
也可以复杂一点，设置系统背景，系统提示词，在问题的基础上

```
from dotenv import load_dotenv  
load_dotenv()  
from langchain_community.chat_models.tongyi import ChatTongyi  
model = ChatTongyi(  
    model="qwen-max",  
)  
response=model.invoke([  
    {  
        "role":"system",  
        "content":"你是一名学生，请以学生的口吻回答用户的问题"  
    },  
    {  
        "role":"user",  
        "content":"你是谁？"  
    }  
]  
)  
print(response.content)#只打印回答的内容
```
![[Pasted image 20260413153436.png]]invoke阻塞调用等待时间长是因为他需要等待模型生成全部内容才会返回结果
stream调用是每当模型返回一个token就会直接返回给我们，这样响应的速度会更快，用户的体验会更好。
![[Pasted image 20260413153948.png]]generator不是一次性的把结果给你了，而是不断生成新的结果作为数组的元素给我们，元素从模型的调用的流里来，模型返回一个，他就给我们返回一个，我们想访问他的元素当然选取的方法是遍历循环访问
```
stream=model.stream("你是谁？")
print(type(stream))
```
到现在这步我们其实还没有向模型发请求，只有在我们便利的时候才会发请求
for循环，使用chunk（片段）
```
for chunk in stream:  
    print(chunk)
```
![[Pasted image 20260413154652.png]]content='我是' additional_kwargs={} response_metadata={} id='lc_run--019d85cd-e279-72a0-897e-9a176ea380b7' tool_calls=[] invalid_tool_calls=[] tool_call_chunks=[]
content='Qwen，' additional_kwargs={} response_metadata={} id='lc_run--019d85cd-e279-72a0-897e-9a176ea380b7' tool_calls=[] invalid_tool_calls=[] tool_call_chunks=[]
content='由阿里云' additional_kwargs={} response_metadata={} id='lc_run--019d85cd-e279-72a0-897e-9a176ea380b7' tool_calls=[] invalid_tool_calls=[] tool_call_chunks=[]
每一个都是一个content，但是每个内容都只有一个片段，是一个个token返回的
但是形式丑，我们只需要content（chunk.content)，同时我们不希望打印一个换一行，使用end=“”，flush=True。（同时我们需要重新获取一次stream重新遍历）
```
stream=model.stream("你是谁？")  
print(type(stream))  
for chunk in stream:  
    print(chunk.content,end="",flush=True)
```
![[Pasted image 20260413155308.png]]我们现在使用模型都是直接拿模型去调用，但是langchain核心是智能体开发。我们正常初始化好模型后，其实大多不会直接去调用模型，把他放在智能体里。
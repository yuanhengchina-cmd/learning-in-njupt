正常是发文本消息，模型基本上都支持；
但是langchain也支持模型发送多模态消息，比如图片、音频、视频、文本等，但前提你使用的模型是多模态模型。
支持多模态的模型例如qwen3.5-plus。
![[Pasted image 20260413210048.png]]
可以看官网里的信息，例如使用阿里云百炼的模型套壳支持多模态的openai（伪装一下）
进行图片的多模态
首先发送在线图片给模型
```
message = { "role": "user", "content": [ {"type": "text", "text": "Describe the content of this image."}, {"type": "image", "url": "https://example.com/path/to/image.jpg"}, ] }
```
多模态消息agent里message里的content对应的不是字符串，而是数组；这里是用图片链接的形式，还有图片base64的形式等
第一个text的文本参数指的是你向模型提的问题，描述图片的内容，告诉ai干什么事
第二个是image，告诉图片信息，告诉agent接下来是图片，地址是什么
```
from dotenv import load_dotenv  
from langchain_classic.agents.xml.prompt import agent_instructions  
from langchain_core.messages import HumanMessage  
  
load_dotenv()#本地.env一定会要UV add python-dotenv  
#再初始化后进行  
import os  
from langchain.chat_models import  init_chat_model  
base_url=os.environ["IMAGE_BASE_URL"]#加载环境变量里的这两个值  
api_key=os.environ["API_KEY"]  
  
model=init_chat_model(  
    model="qwen3.5-plus",#模型名称选择阿里云百炼平台支持的  
    model_provider="openai",#借用openai这个通道，适用于langchain不支持的模型，需要指定模型的提供者  
    api_key=api_key,  
    base_url=base_url,  
  
)
#创建 Agentfrom langchain.agents import create_agent  
agent=create_agent(model=model)
#准备1多模态消息  
# message = {  
#     "role": "user",  
#     "content": [  
#         {"type": "text", "text": "Describe the content of this image."},  
#         {"type": "image", "url": "https://www.bing.com/th/id/OIP.OFMlQdJWRGUjT2PNEWN00AHaEK?w=312&h=180&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3"},  
#     ]  
# }  
#现在又变成字典了,放入数组，因为humanmessage本来就代表content  
message=HumanMessage([  
        {"type": "text", "text": "Describe the content of this image."},  
        {"type": "image", "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},  
    ])

#调用，流式  
stream=agent.stream(  
    {"messages": [message]},  
    stream_mode="messages"  
)  
for chunk,metadata in stream:  
    if chunk.content:  
        print(chunk.content,end="",flush=True)
```
![[Pasted image 20260414145538.png]]需要注意的是图片的url最好是有效可访问的公开链接或者是本地的
对于本地图片
先安装一个图片上传的工具，模拟图片上传
uv add ipywidgets
```
from ipywidgets import FileUpload  
from IPython.display import display  
uploader=FileUpload(accept='*',multiple=False)  
display(uploader)
```
运行后会出现弹窗
![[Pasted image 20260414150113.png]]![[Pasted image 20260414150224.png]]
本地图片需要转成base64的形式
![[Pasted image 20260414150344.png]]同时我们的upload变量已经拿到了我们上传的图片的数据
```
print(uploader.value)
```
![[Pasted image 20260414150534.png]]
```
#读取图片，转为base64字符串  
import base64  
#获取第一个也是唯一一个上传的文件  
uploaded_file=uploader.value[0]#从上传组件里取图片  
#获取其内存视图,拿到具体内容  
content_mv=uploaded_file["content"]  
#转换内存视图->字节,将内容转成字节  
img_bytes=bytes(content_mv)  
#将自己进行base64编码  
img_b64=base64.b64encode(img_bytes).decode("utf-8")
```

```
#组织多模态消息  
# message = {  
#     "role": "user",  
#     "content": [  
#         {"type": "text", "text": "Describe the content of this image."},  
#         {  
#             "type": "image",  
#             "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",  
#             "mime_type": "image/jpeg",  
#         },  
#     ]  
# }  
multimodel_question=HumanMessage(content=[  
    {"type":"image",  
     "base64": img_b64,  
     "mime_type":"image/jpeg"  
     },  
    {"type":"text","text":"Describe the content of this image."},  
  
])  
#两步合成一步  
for chunk,metadata in agent.stream(  
        {"messages": [multimodel_question]},  
    stream_mode="messages"  
):  
    print(chunk.content,end="",flush=True)
```
![[Pasted image 20260414152204.png]]
```
#组织多模态消息  
message = {  
    "role": "user",  
    "content": [  
        {"type": "text", "text": "Describe the content of this image."},  
        {  
            "type": "image",  
            "base64": img_b64,  
            "mime_type": "image/jpeg",  
        },  
    ]  
}  
# multimodel_question=HumanMessage(content=[  
#     {"type":"image",  
#      "base64": img_b64,  
#      "mime_type":"image/jpeg"  
#      },  
#     {"type":"text","text":"Describe the content of this image."},  
#  
# ])  
#两步合成一步  
for chunk,metadata in agent.stream(  
        {"messages": [message]},  
    stream_mode="messages"  
):  
    print(chunk.content,end="",flush=True)
```
![[Pasted image 20260414152550.png]]
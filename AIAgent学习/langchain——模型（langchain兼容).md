-init_chat_model
-Model类
-调用模型
-Agent中的model

langchain基本上支持市面上常见的大语言模型，并提供了各个模型的对应的依赖库
还支持一些通用服务本地平台ollama之类
百炼平台没有对应的依赖。
langchain为每个模型的提供商都封装了一个类，ep.ChatOpenAI——langchain-openai，ChatDeepSeek——langchain-deepseek，ChatOllama——langchain-ollama
使用模型，只需引入对应的类，创建合适的对象，并选择合适的模型
```
#早期是这样
from langchain_deepseek import ChatDeepSeek
model=ChatDeepSeek(model="deepseek-v3")
#langchain1.0后引入一个统一的方式,这样不需要导入任何模型提供商的类了，这样更统一，帮你去全自动化完成创建对应的对象；当然老版方式也是兼容的。
from langchain.chat_models import init_chat_model
model=init_chat_model(model="deepseek-chat")
```
使用步骤
1、引入依赖uv add langchain
uv add langchain-deepseek
2、配置环境（可能需要查一下具体是什么）
DEEPSEEK_API_KEY=sk-xxx
（在.env文件里LANGCHAIN_API_KEY=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也需要指定）
3、初始化模型
from langchain.chat_models import init_chat_model
model=init_chat_model(model="deepseek-chat")
都是由langchain根据模型名称推断出来的
官方推荐init_chat_model，最方便，不需要自己去查模型对应的类和指定apikey、baseurl（但是依赖都是要安装的，langchain的依赖，模型的依赖，模型名字与apikey名字不能乱写，要查官方文档）
![[Pasted image 20260411204531.png]]
![[Pasted image 20260411203837.png]]

![[Pasted image 20260411204058.png]]
![[Pasted image 20260411204119.png]]![[Pasted image 20260411204456.png]]
```
#导入langchain初始化模型的函数  
from langchain.chat_models import init_chat_model  
#调用init_chat_model函数初始化模型  
#参数model用来指定模型名称，langchain会根据模型名称自动设base_url，并从环境变量中获取api_key  
model=init_chat_model(  
    model="deepseek-chat"  
)
```
注意必须在计算机系统变量里指定（langchain会根据模型名称自动去推断提供商，然后去创建对应的模型对象，还是类创建出的对象）
DEEPSEEK_API_KEY=sk-696f3e25346b4cd6924b4c8b738c9f38  
LANGCHAIN_API_KEY=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
```
#init_chat_model会根据模型名称自动确定类  
print(type(model))
```
![[Pasted image 20260411205725.png]]
这种方法仅限于langchain支持的模型，很有限但十分方便
![[Pasted image 20260411210106.png]]
函数内部指明支持类型的模型以及提供商，除此之外的模型不支持用名称自动推断模型提供商的这种方式。而是需要我们手动指明
例如，阿里云百炼是支持openai，把他伪造成openai就行
```
api_key="sk-fe45c35707f34927a5c1d63ae121b3a6"  
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
"""  
import os  
base_url=os.environ["BASE_URL"]#加载环境变量里的这两个值  
api_key=os.environ["API_KEY"]  
"""  
model=init_chat_model(  
    model="qwen-max",#模型名称选择阿里云百炼平台支持的  
    model_provider="openai",#借用openai这个通道，适用于langchain不支持的模型，需要指定模型的提供者  
    api_key=api_key,  
    base_url=base_url,  
)
print(type(model))
```
![[Pasted image 20260411211453.png]]
使用chatopen类进行的初始化，底层还是阿里云百炼

os加载f方法需要配.env
![[Pasted image 20260411212245.png]]
```
from dotenv import load_dotenv  
load_dotenv()#本地.env一定会要UV add python-dotenv  
#再初始化后进行  
import os  
base_url=os.environ["BASE_URL"]#加载环境变量里的这两个值  
api_key=os.environ["API_KEY"]  
  
model=init_chat_model(  
    model="qwen-max",#模型名称选择阿里云百炼平台支持的  
    model_provider="openai",#借用openai这个通道，适用于langchain不支持的模型，需要指定模型的提供者  
    api_key=api_key,  
    base_url=base_url,  
)
```
![[Pasted image 20260411212443.png]]还有configurable_fields可配置的字段
ep.temperature控制生成文本的随机性，值越小越确定
max_token控制文本生成的最大长度
top_p控制生成文本的多样性，值越小越多样
timeout控制生成文本的超时时间
max_retries控制生成文本的最大重试次数
![[Pasted image 20260411212537.png]]
![[Pasted image 20260411212956.png]]



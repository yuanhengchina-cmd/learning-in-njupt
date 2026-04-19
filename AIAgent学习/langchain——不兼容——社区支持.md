当然遇到openai不兼容，使用社区方法访问也可以（最特殊的情况）
使用社区的model类来访问阿里云百炼的qwen模型（而不是init_chat_model)
先安装langchain社区依赖
uv add langchain-community
再安装阿里云百炼的依赖
uv add dashcope
这样就可以使用langchain社区的api，而不报错
from langchain.community.chat_models.tongyi import ChatTongyi
ChatTongyi是比较新的类

```
from dotenv import load_dotenv  
load_dotenv()
from langchain_community.chat_models.tongyi import ChatTongyi  
model = ChatTongyi(  
    model="qwen-max"  
)
```
这里不用指定apikey，他会自动·从环境变量里加载apikey。名字是DASHSCOPEAPIKEY，baseurl由类指定默认是知道的。
当然temperature等参数还是可以配置
![[Pasted image 20260413145759.png]]正常我们还是直接用init_chat_model（自动初始化模型名称和模型的连接，自动告诉你模型提供商）,只需要交代model这一个参数就行；但是遇到qwen这种特殊的，我们还需要在环境变量.env里指定apikey、baseurl、模型提供商（这里我们借助openai的模型提供商，实在不放心就自己指定model_provider）/或者是安装langchain-community从社区支持里找对应的类，这样也是只用交代model，他的baseurl已经帮你自动配好，我们只需要在.env里配好DASHSCOPE就行，他会自动加载DASHSCOPE，这里也是会自动推断模型提供商。
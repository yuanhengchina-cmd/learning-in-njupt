利用langchain提供的api快速构建一个agent
输入即用户提问
模型部分存在模型分析（1、是否需要调用工具2、调用哪个工具（包括提取关键信息传参给工具）3、工具执行结果是否足以回答用户问题）
工具拿到关键信息后有反馈给模型，模型又要进行分析和思考了observation（感知）即执行3，不满足可能会接着调其他更合适的用工具/使用相同工具再次去查询
模型分析发现满足后生成结果（把查询到的结果进行组装返回用户友好的结果）
![[Pasted image 20260411143332.png|245]]
用户输入后，agent不会立刻去调用大模型，先将我们定义的工具的信息组织成模型所需要的参数的形式（工具的名字、描述信息、参数）把这个封装成tools，把用户问题封装成message，将tools和meesage组装在一起作为请求参数一起发给模型。模型此时知道我们的工具和用户的需求，大模型根据用户的提问的问题来分析我要不要调工具，调那个工具，这是agent返回一个结果（结果中包含工具信息），名字叫什么参数叫什么，langchain会自动分析，发现要调的工具，帮你去调，得到一个结果，再把结果发送给模型。模型后拿到结果后分析并组织答案并返回给用户。
小知识点：模型使用的api接口（人家服务器上的），tool工具是我们自定义的，模型是如何感知/调用我们定义的工具集的呢，官方文档是将model、message、stream（可选）定义在一起
![[Pasted image 20260411144845.png]]
![[Pasted image 20260411144952.png]]
![[Pasted image 20260411145032.png]]
仅支持函数，我们定义的工具就是函数（仅使用@tool的装饰器装饰了一下）
（
#创建Agent  
from langchain.agents import create_agent  
agent = create_agent(  
    "deepseek-chat",#第一个参数指定模型的名字  
 tools =[getweather]#参数是工具集（是数组，将前面定义的工具/方法的函数名传进来）  
)
**langchain可以帮我们把message消息和工具toolsde所有信息全部组装起来，作为一个完整的请求参数发给模型**
模型是如何调用工具？（模型在对方服务器，工具在我们本地）
模型返回的结果是chat；形式是右侧json
![[Pasted image 20260411150203.png]]
模型返回的结果除了内容以外，还可以返回我要调哪个工具（模型本身不能调工具，但他可以返回文字我想要调哪个工具），我们分析返回的结果，找到他要调用的工具和参数信息，我们帮他去调用。（langchain帮我们自动分析模型返回的结果，从里面找到有没有要调用的工具，如果有，工具叫什么名字，参数是什么，langchain帮我们去调用）
![[Pasted image 20260411150744.png|380]]
![[Pasted image 20260411150438.png]]
ep.需求：开发一个智能体，可以帮助用户查询天气信息，当用户来询问时可以调用api查询并返回给用户。
先进行环境的准备
uv add notebook
uv add langchain
uv add langchain-deepseek
uv add langchain-openai
![[Pasted image 20260411125831.png|387]]
1、加载环境变量
2、定义工具
3、定义agent
4、调用agent

1、加载环境变量
先配置.env
OPENAL_API_KEY=sk-fe45c35707f34927a5c1d63ae121b3a6  
LANGCHAIN_API_KEY=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
![[Pasted image 20260411131406.png]]

```
#加载环境变量
from dotenv import load_dotenv  
load_dotenv()
```
不需要os去读取apikey，langchain会自动做

2、定义工具

```
#定义工具  
#工具1：查特定地区天气的工具/函数  
from langchain.tools import tool  
@tool  
def getweather(location:str)->str:  
    """get weather in a given location  
    Args:        location (str): location,city,state    """    
    return f"current weather in {location} is sunny"
```
用英文返回，大模型识别英文效果更好
![[Pasted image 20260411145235.png]]
参数列表：
工具名-函数名
工具的描述（告诉模型你是干什么的）-注释（get weather in a given location  
参数是location， Args:  location (str):  ；
代表是 location,city,state 或者某一个坐标

参数的描述：location:str)->str（可以重复描述，什么1类型怎么传）

3、创建Agent
create_agent第一个参数指定模型的名字
不用再去指定apikey、url的参数，因为langchain自动帮我们从环境变量里加载api_key,同时他也是直到模型的baseURL
第二个参数是工具集（是数组，将前面定义的工具/方法的函数名传进来），更多工具都可以添加
```
#创建Agent  
from langchain.agents import create_agent  
agent = create_agent(  
    "deepseek-chat",#第一个参数指定模型的名字  
 tools =[getweather]#参数是工具集（是数组，将前面定义的工具/方法的函数名传进来）  
)
```
![[Pasted image 20260411132444.png]]
换api_key后需要重新加载环境
（#加载环境  
from dotenv import load_dotenv  
load_dotenv()
）

4.发起调用
agent.invoke调用
传参数，只传了一个消息（他是一个数组形式，用户、内容；没有系统提示词），模型等已经指定
```
#调用Agent  
print("正在调用大模型")  
response= agent.invoke(  
    {  
        "messages":[  
            {"role":"user","content":"扬州今天天气如何"}  
        ]  
    }  
  
)
```
```
print(response)
```
![[Pasted image 20260411142314.png]]因为我们是langchain去调用的，返回的也是langchain的格式
有一个是message的属性，message是数组，分为很多部分，1、用户问的问题”扬州今天天气如何“2、ai的回答”我来帮您查询扬州今天的天气情况“3、查询结果没有直接回答4、在最后有回答”'根据查询结果，扬州今天的天气是**晴天**。天气不错，适合外出活动！'“
模型并非胡说，而是给予我们的工具得到一个结果。
如果我们的函数/工具改成了真正调用api查询，那么这个天气也能正确的查询
{'messages': [HumanMessage(content='扬州今天天气如何', additional_kwargs={}, response_metadata={}, id='0bf205a2-7d7d-48ab-a63c-6cb0e47cdd83'), AIMessage(content='我来帮您查询扬州今天的天气情况。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 314, 'total_tokens': 365, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 314}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache_20260410', 'id': 'b3ce1744-7025-4271-9799-c320c32129e2', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019d7b34-6d44-74d3-b65f-1873a8e9dd45-0', tool_calls=[{'name': 'getweather', 'args': {'location': '扬州'}, 'id': 'call_00_HHviMiGDXTl9SzyqQTsZkogT', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 314, 'output_tokens': 51, 'total_tokens': 365, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}), ToolMessage(content='current weather in 扬州 is sunny', name='getweather', id='8c693fe3-64e8-43e7-b340-effb68f7e7f9', tool_call_id='call_00_HHviMiGDXTl9SzyqQTsZkogT'), AIMessage(content='根据查询结果，扬州今天的天气是**晴天**。天气不错，适合外出活动！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 389, 'total_tokens': 408, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 320}, 'prompt_cache_hit_tokens': 320, 'prompt_cache_miss_tokens': 69}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache_20260410', 'id': '5dec4c29-4b80-4536-adb5-e1b29a3d2067', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019d7b34-7d57-7423-9f4f-074aa44bc9e2-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 389, 'output_tokens': 19, 'total_tokens': 408, 'input_token_details': {'cache_read': 320}, 'output_token_details': {}})]}
```
for message in response["messages"]:  
    print(message.model_dump_json(indent=2))
```
model_dump_json可以指定缩进是多少，他可以吧这里的每一条消息都打印出来
![[Pasted image 20260411151650.png]]
使用了toolcalls。agent分析他，得到结果，并把结果返回给大模型。
![[Pasted image 20260411151804.png]]
大模型性就会分析并输出符合用户要求的结果
![[Pasted image 20260411151922.png]]
{
  "content": "扬州今天天气如何",
  "additional_kwargs": {},
  "response_metadata": {},
  "type": "human",
  "name": null,
  "id": "0bf205a2-7d7d-48ab-a63c-6cb0e47cdd83"
}
{
  "content": "我来帮您查询扬州今天的天气情况。",
  "additional_kwargs": {
    "refusal": null
  },
  "response_metadata": {
    "token_usage": {
      "completion_tokens": 51,
      "prompt_tokens": 314,
      "total_tokens": 365,
      "completion_tokens_details": null,
      "prompt_tokens_details": {
        "audio_tokens": null,
        "cached_tokens": 0
      },
      "prompt_cache_hit_tokens": 0,
      "prompt_cache_miss_tokens": 314
    },
    "model_provider": "deepseek",
    "model_name": "deepseek-chat",
    "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache_20260410",
    "id": "b3ce1744-7025-4271-9799-c320c32129e2",
    "finish_reason": "tool_calls",
    "logprobs": null
  },
  "type": "ai",
  "name": null,
  "id": "lc_run--019d7b34-6d44-74d3-b65f-1873a8e9dd45-0",
  "tool_calls": [
    {
      "name": "getweather",
      "args": {
        "location": "扬州"
      },
      "id": "call_00_HHviMiGDXTl9SzyqQTsZkogT",
      "type": "tool_call"
    }
  ],
  "invalid_tool_calls": [],
  "usage_metadata": {
    "input_tokens": 314,
    "output_tokens": 51,
    "total_tokens": 365,
    "input_token_details": {
      "cache_read": 0
    },
    "output_token_details": {}
  }
}
{
  "content": "current weather in 扬州 is sunny",
  "additional_kwargs": {},
  "response_metadata": {},
  "type": "tool",
  "name": "getweather",
  "id": "8c693fe3-64e8-43e7-b340-effb68f7e7f9",
  "tool_call_id": "call_00_HHviMiGDXTl9SzyqQTsZkogT",
  "artifact": null,
  "status": "success"
}
{
  "content": "根据查询结果，扬州今天的天气是**晴天**。天气不错，适合外出活动！",
  "additional_kwargs": {
    "refusal": null
  },
  "response_metadata": {
    "token_usage": {
      "completion_tokens": 19,
      "prompt_tokens": 389,
      "total_tokens": 408,
      "completion_tokens_details": null,
      "prompt_tokens_details": {
        "audio_tokens": null,
        "cached_tokens": 320
      },
      "prompt_cache_hit_tokens": 320,
      "prompt_cache_miss_tokens": 69
    },
    "model_provider": "deepseek",
    "model_name": "deepseek-chat",
    "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache_20260410",
    "id": "5dec4c29-4b80-4536-adb5-e1b29a3d2067",
    "finish_reason": "stop",
    "logprobs": null
  },
  "type": "ai",
  "name": null,
  "id": "lc_run--019d7b34-7d57-7423-9f4f-074aa44bc9e2-0",
  "tool_calls": [],
  "invalid_tool_calls": [],
  "usage_metadata": {
    "input_tokens": 389,
    "output_tokens": 19,
    "total_tokens": 408,
    "input_token_details": {
      "cache_read": 320
    },
    "output_token_details": {}
  }
}
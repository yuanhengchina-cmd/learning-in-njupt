OLLama：简化大模型在本地部署和运行的开源软件，开发者可以导入定制自己的模型，不用关心底层的实现细节，只需要本地的下载安装部署——云端平台的本地版，由自身硬件提供算力支撑模型的运行。模型通过ollama软件在电脑里运行。
模型：‘n’ b代表模型的一个参数量
在ollama里运行的都是蒸馏模型（不是完整的模型），只保留核心功能，性能肯定不如云上模型。
根据显存选模型（的参数量），ep.显卡8GB显存跑8b蒸馏模型是没问题的，在安全的范围内安装，参数量-性能只是影响模型输出文字的速度/推理速度（运行时依赖显卡）
![[Pasted image 20260410180338.png|498]]
ollama对外有接口：
-命令行访问ollama run qwen2.5:7b
![[Pasted image 20260410180154.png]]
![[Pasted image 20260410181129.png]]
实际使用也是使用api接口形式去调用ollama管理的模型
代码调用ollama的本地模型，在原本云端模型的代码进行简单改动
主要修改点
1、将base_url从阿里云地址改成http://localhost:11434/v1)基本上这个地址是固定格式（api_key直接省略，用不到），默认是http，不是https。
```
base_url="http://localhost:11434/v1",
```
2、model改成本地模型的名称
```
completion = client.chat.completions.create(  
    model="qwen2.5:7b",  # 您可以按需更换为其它深度思考模型qwen-plus，qwen3-max，deepseek-v3  
    messages=messages,  
    extra_body={"enable_thinking": True},  
    stream=True  
)
```
注意：由于是我们本地模型算力不如云端模型，等待时间较长，速度慢，由本地硬件支撑。运行的前提是必须提前启动ollama这个软件平台，我们是通过这个平台部署的本地模型，自然需要这个平台的运行作为必要的支撑条件。
可以使用ollama list查看我们通过ollama部署的本地模型
![[Pasted image 20260410193338.png]]
完整的验证代码：
```
import openai  
from openai import OpenAI  
import os  
  
client = OpenAI(  
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"  
    #api_key=os.getenv("DASHSCOPE_API_KEY"),    # api_key="sk-fe45c35707f34927a5c1d63ae121b3a6",    base_url="http://localhost:11434/v1",  
)  
  
messages = [{"role": "user", "content": "你是谁"}]#提问你是谁？这是问题部分  
completion = client.chat.completions.create(  
    model="qwen2.5:7b",  # 您可以按需更换为其它深度思考模型qwen-plus，qwen3-max，deepseek-v3  
    messages=messages,  
    extra_body={"enable_thinking": True},  
    stream=True  
)  
is_answering = False  # 是否进入回复阶段  
print("\n" + "=" * 20 + "思考过程" + "=" * 20)  
for chunk in completion:  
    delta = chunk.choices[0].delta  
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:  
        if not is_answering:  
            print(delta.reasoning_content, end="", flush=True)  
    if hasattr(delta, "content") and delta.content:  
        if not is_answering:  
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)  
            is_answering = True  
        print(delta.content, end="", flush=True)
```

```
(base) PS C:\Users\lenovo> cd D:
(base) PS D:\> mkdir MyHyperspectralAgent


    目录: D:\


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2026/3/21     21:28                MyHyperspectralAgent


(base) PS D:\> cd MyHyperspectralAgent
(base) PS D:\MyHyperspectralAgent> mkdir papers


    目录: D:\MyHyperspectralAgent


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2026/3/21     21:28                papers


(base) PS D:\MyHyperspectralAgent> conda config --set ssl_verify false
(base) PS D:\MyHyperspectralAgent> conda create -n local_agent_env python=3.10 -y
Collecting package metadata (current_repodata.json): failed

ProxyError: Conda cannot proceed due to an error in your proxy configuration.
Check for typos and other configuration errors in any '.netrc' file in your home directory,
any environment variables ending in '_PROXY', and any other system-wide proxy
configuration settings.


(base) PS D:\MyHyperspectralAgent> conda create -n local_agent_env python=3.10 -y
Collecting package metadata (current_repodata.json): done
Solving environment: /
Warning: >10 possible package resolutions (only showing differing packages):
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1, https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0, https://mirrors.ustc.edu.cn/anaconda/pkgs/main/win-64::xz-5.8.2-h53af0af_0
  ... and othedone


==> WARNING: A newer version of conda exists. <==
  current version: 4.14.0
  latest version: 26.1.1

Please update conda by running

    $ conda update -n base -c conda-forge conda



## Package Plan ##

  environment location: D:\anaconda\envs\local_agent_env

  added / updated specs:
    - python=3.10


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    bzip2-1.0.8                |       h2bbff1b_6          90 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    ca-certificates-2025.12.2  |       haa95532_0         125 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    libexpat-2.7.4             |       hd7fb8db_0         121 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    libffi-3.4.4               |       hd77b12b_1         122 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    libzlib-1.3.1              |       h02ab6af_0          64 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    openssl-3.5.5              |       hbb43b14_0         8.9 MB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    packaging-25.0             |  py310haa95532_1         165 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    pip-26.0.1                 |     pyhc872135_0         1.1 MB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    python-3.10.20             |       h1044e36_0        15.5 MB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    setuptools-80.10.2         |  py310haa95532_0         1.3 MB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    sqlite-3.51.2              |       hee5a0db_0         917 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    tk-8.6.15                  |       hf199647_0         3.5 MB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    tzdata-2026a               |       he532380_0         117 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    ucrt-10.0.22621.0          |       haa95532_0         620 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    vc-14.3                    |      h2df5915_10          19 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    vc14_runtime-14.44.35208   |      h4927774_10         825 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    vs2015_runtime-14.44.35208 |      ha6b5a95_10          19 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    wheel-0.46.3               |  py310haa95532_0          78 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    xz-5.8.2                   |       h53af0af_0         265 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    zlib-1.3.1                 |       h02ab6af_0         113 KB  https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
    ------------------------------------------------------------
                                           Total:        33.9 MB

The following NEW packages will be INSTALLED:

  bzip2              anaconda/pkgs/main/win-64::bzip2-1.0.8-h2bbff1b_6
  ca-certificates    anaconda/pkgs/main/win-64::ca-certificates-2025.12.2-haa95532_0
  libexpat           anaconda/pkgs/main/win-64::libexpat-2.7.4-hd7fb8db_0
  libffi             anaconda/pkgs/main/win-64::libffi-3.4.4-hd77b12b_1
  libzlib            anaconda/pkgs/main/win-64::libzlib-1.3.1-h02ab6af_0
  openssl            anaconda/pkgs/main/win-64::openssl-3.5.5-hbb43b14_0
  packaging          anaconda/pkgs/main/win-64::packaging-25.0-py310haa95532_1
  pip                anaconda/pkgs/main/noarch::pip-26.0.1-pyhc872135_0
  python             anaconda/pkgs/main/win-64::python-3.10.20-h1044e36_0
  setuptools         anaconda/pkgs/main/win-64::setuptools-80.10.2-py310haa95532_0
  sqlite             anaconda/pkgs/main/win-64::sqlite-3.51.2-hee5a0db_0
  tk                 anaconda/pkgs/main/win-64::tk-8.6.15-hf199647_0
  tzdata             anaconda/pkgs/main/noarch::tzdata-2026a-he532380_0
  ucrt               anaconda/pkgs/main/win-64::ucrt-10.0.22621.0-haa95532_0
  vc                 anaconda/pkgs/main/win-64::vc-14.3-h2df5915_10
  vc14_runtime       anaconda/pkgs/main/win-64::vc14_runtime-14.44.35208-h4927774_10
  vs2015_runtime     anaconda/pkgs/main/win-64::vs2015_runtime-14.44.35208-ha6b5a95_10
done
#
# To activate this environment, use
#
#     $ conda activate local_agent_env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

Retrieving notices: ...working... done
(base) PS D:\MyHyperspectralAgent> conda activate local_agent_env
(local_agent_env) PS D:\MyHyperspectralAgent> pip install streamlit llama-index llama-index-llms-ollama llama-index-embeddings-ollama
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting streamlit
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dc/e6/412c1e1f200ca8c32ecf10201839183e261ad61ced3ede34a66f6d4be3cf/streamlit-1.55.0-py3-none-any.whl (9.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.1/9.1 MB 11.1 MB/s  0:00:00
Collecting llama-index
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9d/83/58b3fa26242122e336d18a0d5c9b98d35960a2f7f4ec397226a25ebf2ce7/llama_index-0.14.18-py3-none-any.whl (7.8 kB)
Collecting llama-index-llms-ollama
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c0/87/ecb53ae8f4a79e9e8b2e2f9ef649840ab985c5a095438b2f53b72674b66f/llama_index_llms_ollama-0.10.1-py3-none-any.whl (8.8 kB)
Collecting llama-index-embeddings-ollama
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9a/36/53674403380483510a7f657c5d6f0bdac5b7f9ec5a1a8d06cdfdd6dc47f2/llama_index_embeddings_ollama-0.9.0-py3-none-any.whl (6.2 kB)
Collecting altair!=5.4.0,!=5.4.1,<7,>=4.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/db/33/ef2f2409450ef6daa61459d5de5c08128e7d3edb773fefd0a324d1310238/altair-6.0.0-py3-none-any.whl (795 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 795.4/795.4 kB 11.3 MB/s  0:00:00
Collecting blinker<2,>=1.5.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/10/cb/f2ad4230dc2eb1a74edf38f1a38b9b52277f75bef262d8908e60d957e13c/blinker-1.9.0-py3-none-any.whl (8.5 kB)
Collecting cachetools<8,>=5.5 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/06/f3/39cf3367b8107baa44f861dc802cbf16263c945b62d8265d36034fc07bea/cachetools-7.0.5-py3-none-any.whl (13 kB)
Collecting click<9,>=7.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/98/78/01c019cdb5d6498122777c1a43056ebb3ebfeef2076d9d026bfe15583b2b/click-8.3.1-py3-none-any.whl (108 kB)
Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6a/09/e21df6aef1e1ffc0c816f0522ddc3f6dcded766c3261813131c78a704470/gitpython-3.1.46-py3-none-any.whl (208 kB)
Collecting numpy<3,>=1.23 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a3/dd/4b822569d6b96c39d1215dbae0582fd99954dcbcf0c1a13c61783feaca3f/numpy-2.2.6-cp310-cp310-win_amd64.whl (12.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.9/12.9 MB 14.2 MB/s  0:00:00
Requirement already satisfied: packaging>=20 in d:\anaconda\envs\local_agent_env\lib\site-packages (from streamlit) (25.0)
Collecting pandas<3,>=1.4.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/85/72/530900610650f54a35a19476eca5104f38555afccda1aa11a92ee14cb21d/pandas-2.3.3-cp310-cp310-win_amd64.whl (11.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.3/11.3 MB 13.9 MB/s  0:00:00
Collecting pillow<13,>=7.1.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f8/f9/9f6b01c0881d7036063aa6612ef04c0e2cad96be21325a1e92d0203f8e91/pillow-12.1.1-cp310-cp310-win_amd64.whl (7.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 13.5 MB/s  0:00:00
Collecting pydeck<1,>=0.8.0b4 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ab/4c/b888e6cf58bd9db9c93f40d1c6be8283ff49d88919231afe93a6bcf61626/pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 9.9 MB/s  0:00:00
Collecting protobuf<7,>=3.20 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/76/5d/683efcd4798e0030c1bab27374fd13a89f7c2515fb1f3123efdfaa5eab57/protobuf-6.33.6-cp310-abi3-win_amd64.whl (437 kB)
Collecting pyarrow>=7.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8d/94/d359e708672878d7638a04a0448edf7c707f9e5606cee11e15aaa5c7535a/pyarrow-23.0.1-cp310-cp310-win_amd64.whl (27.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 27.5/27.5 MB 12.8 MB/s  0:00:02
Collecting requests<3,>=2.27 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1e/db/4254e3eabe8020b458f1a747140d32277ec7a271daf1d235b70dc0b4e6e3/requests-2.32.5-py3-none-any.whl (64 kB)
Collecting tenacity<10,>=8.1.0 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d7/c1/eb8f9debc45d3b7918a32ab756658a0904732f75e555402972246b0b8e71/tenacity-9.1.4-py3-none-any.whl (28 kB)
Collecting toml<2,>=0.10.1 (from streamlit)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/44/6f/7120676b6d73228c96e17f1f794d8ab046fc910d781c8d151120c3f1569e/toml-0.10.2-py2.py3-none-any.whl (16 kB)
Collecting tornado!=6.5.0,<7,>=6.0.3 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2c/23/f6c6112a04d28eed765e374435fb1a9198f73e1ec4b4024184f21faeb1ad/tornado-6.5.5-cp39-abi3-win_amd64.whl (448 kB)
Collecting typing-extensions<5,>=4.10.0 (from streamlit)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Collecting watchdog<7,>=2.1.5 (from streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/db/d9/c495884c6e548fce18a8f40568ff120bc3a4b7b99813081c8ac0c936fa64/watchdog-6.0.0-py3-none-win_amd64.whl (79 kB)
Collecting jinja2 (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl (134 kB)
Collecting jsonschema>=3.0 (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/69/90/f63fb5873511e014207a475e2bb4e8b2e570d655b00ac19a9a0ca0a385ee/jsonschema-4.26.0-py3-none-any.whl (90 kB)
Collecting narwhals>=1.27.1 (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fe/75/0b4a10da17a44cf13567d08a9c7632a285297e46253263f1ae119129d10a/narwhals-2.18.0-py3-none-any.whl (444 kB)
Collecting colorama (from click<9,>=7.0->streamlit)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/d1/d6/3965ed04c63042e047cb6a3e6ed1a63a35087b6a609aa3a15ed8ac56c221/colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a0/61/5c78b91c3143ed5c14207f463aecfc8f9dbb5092fb2869baf37c273b2705/gitdb-4.0.12-py3-none-any.whl (62 kB)
Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c1/d4/59e74daffcb57a07668852eeeb6035af9f32cbfd7a1d2511f17d2fe6a738/smmap-5.0.3-py3-none-any.whl (24 kB)
Collecting python-dateutil>=2.8.2 (from pandas<3,>=1.4.0->streamlit)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Collecting pytz>=2020.1 (from pandas<3,>=1.4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/10/99/781fe0c827be2742bcc775efefccb3b048a3a9c6ce9aec0cbf4a101677e5/pytz-2026.1.post1-py2.py3-none-any.whl (510 kB)
Collecting tzdata>=2022.7 (from pandas<3,>=1.4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c7/b0/003792df09decd6849a5e39c28b513c06e84436a54440380862b5aeff25d/tzdata-2025.3-py2.py3-none-any.whl (348 kB)
Collecting charset_normalizer<4,>=2 (from requests<3,>=2.27->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d8/8f/78f5489ffadb0db3eb7aff53d31c24531d33eb545f0c6f6567c25f49a5ff/charset_normalizer-3.4.6-cp310-cp310-win_amd64.whl (154 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.27->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0e/61/66938bbb5fc52dbdf84594873d5b51fb1f7c7794e9c0f5bd885f30bc507b/idna-3.11-py3-none-any.whl (71 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.27->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/08/aaaad47bc4e9dc8c725e68f9d04865dbcb2052843ff09c97b08904852d84/urllib3-2.6.3-py3-none-any.whl (131 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.27->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9a/3c/c17fb3ca2d9c3acff52e30b309f538586f9f5b9c9cf454f3845fc9af4881/certifi-2026.2.25-py3-none-any.whl (153 kB)
Collecting llama-index-cli<0.6,>=0.5.0 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b6/68/4d35de5871a39a26eb17cca308d47cd61354b38622ba4753b0f6c210bc6d/llama_index_cli-0.5.6-py3-none-any.whl (28 kB)
Collecting llama-index-core<0.15.0,>=0.14.18 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/27/33/64a32bc376390340a05f3bcea0594dc6384c77a8298569e851add75ff9bb/llama_index_core-0.14.18-py3-none-any.whl (11.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.9/11.9 MB 13.4 MB/s  0:00:00
Collecting llama-index-embeddings-openai<0.7,>=0.6.0 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4e/d1/4bb0b80f4057903110060f617ef519197194b3ff5dd6153d850c8f5676fa/llama_index_embeddings_openai-0.6.0-py3-none-any.whl (7.7 kB)
Collecting llama-index-indices-managed-llama-cloud>=0.4.0 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/47/02/5d04cd99c395d23f4ca7cfab4577e21af3c2edddf0e91906adf362804fe8/llama_index_indices_managed_llama_cloud-0.11.0-py3-none-any.whl (3.8 kB)
Collecting llama-index-llms-openai<0.8,>=0.7.0 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/99/1f/da6ec7e5f998fac3257def0b99a3f8a69dbcb17dc685dba65944e6fd3962/llama_index_llms_openai-0.7.3-py3-none-any.whl (28 kB)
Collecting llama-index-readers-llama-parse>=0.4.0 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/72/52/1d8e1aa23ef32d0df3a7fd533ae0c1bbcd15ae24d9c43d5c66096ef47e0e/llama_index_readers_llama_parse-0.6.0-py3-none-any.whl (3.2 kB)
Collecting nltk>3.8.1 (from llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c2/7e/9af5a710a1236e4772de8dfcc6af942a561327bb9f42b5b4a24d0cf100fd/nltk-3.9.3-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 10.1 MB/s  0:00:00
Collecting aiohttp<4,>=3.8.6 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/80/df/29cd63c7ecfdb65ccc12f7d808cac4fa2a19544660c06c61a4a48462de0c/aiohttp-3.13.3-cp310-cp310-win_amd64.whl (456 kB)
Collecting aiosqlite (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/00/b7/e3bf5133d697a08128598c8d0abc5e16377b51465a33756de24fa7dee953/aiosqlite-0.22.1-py3-none-any.whl (17 kB)
Collecting banks<3,>=2.3.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b8/5a/f38b49e8b225b0c774e97c9495e52ab9ccdf6d82bde68c513bd736820eb2/banks-2.4.1-py3-none-any.whl (35 kB)
Collecting dataclasses-json (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c3/be/d0d44e092656fe7a06b55e6103cbce807cdbdee17884a5367c68c9860853/dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
Collecting deprecated>=1.2.9.3 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/84/d0/205d54408c08b13550c733c4b85429e7ead111c7f0014309637425520a9a/deprecated-1.3.1-py2.py3-none-any.whl (11 kB)
Collecting dirtyjson<2,>=1.0.8 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/68/69/1bcf70f81de1b4a9f21b3a62ec0c83bdff991c88d6cc2267d02408457e88/dirtyjson-1.0.8-py3-none-any.whl (25 kB)
Collecting filetype<2,>=1.2.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/18/79/1b8fa1bb3568781e84c9200f951c735f3f157429f44be0495da55894d620/filetype-1.2.0-py2.py3-none-any.whl (19 kB)
Collecting fsspec>=2023.5.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e6/ab/fb21f4c939bb440104cc2b396d3be1d9b7a9fd3c6c2a53d98c45b3d7c954/fsspec-2026.2.0-py3-none-any.whl (202 kB)
Collecting httpx (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/39/e50c7c3a983047577ee07d2a9e53faf5a69493943ec3f6a384bdc792deb2/httpx-0.28.1-py3-none-any.whl (73 kB)
Collecting llama-index-workflows!=2.9.0,<3,>=2 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8c/28/20dc2db83adc2d9a11e042eac568f52788eb850e9381ffb1087d51f46672/llama_index_workflows-2.17.1-py3-none-any.whl (110 kB)
Collecting nest-asyncio<2,>=1.5.8 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/a0/c4/c2971a3ba4c6103a3d10c4b0f24f461ddc027f0f09763220cf35ca1401b3/nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
Collecting networkx>=3.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b9/54/dd730b32ea14ea797530a4479b2ed46a6fb250f682a9cfb997e968bf0261/networkx-3.4.2-py3-none-any.whl (1.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 10.4 MB/s  0:00:00
Collecting platformdirs (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/63/d7/97f7e3a6abb67d8080dd406fd4df842c2be0efaf712d1c899c32a075027c/platformdirs-4.9.4-py3-none-any.whl (21 kB)
Collecting pydantic>=2.8.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5a/87/b70ad306ebb6f9b585f114d0ac2137d792b48be34d732d60e597c2f8465a/pydantic-2.12.5-py3-none-any.whl (463 kB)
Collecting pyyaml>=6.0.1 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/fa/926c003379b19fca39dd4634818b00dec6c62d87faf628d1394e137354d4/pyyaml-6.0.3-cp310-cp310-win_amd64.whl (158 kB)
Requirement already satisfied: setuptools>=80.9.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15.0,>=0.14.18->llama-index) (80.10.2)
Collecting sqlalchemy>=1.4.49 (from sqlalchemy[asyncio]>=1.4.49->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/de/03/9752eb2a41afdd8568e41ac3c3128e32a0a73eada5ab80483083604a56d1/sqlalchemy-2.0.48-cp310-cp310-win_amd64.whl (2.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 10.9 MB/s  0:00:00
Collecting tiktoken>=0.7.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5f/d2/82e795a6a9bafa034bf26a58e68fe9a89eeaaa610d51dbeb22106ba04f0a/tiktoken-0.12.0-cp310-cp310-win_amd64.whl (879 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 879.4/879.4 kB 7.8 MB/s  0:00:00
Collecting tinytag>=2.2.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ce/34/d50e338631baaf65ec5396e70085e5de0b52b24b28db1ffbc1c6e82190dc/tinytag-2.2.1-py3-none-any.whl (32 kB)
Collecting tqdm<5,>=4.66.1 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/16/e1/3079a9ff9b8e11b846c6ac5c8b5bfb7ff225eee721825310c91b3b50304f/tqdm-4.67.3-py3-none-any.whl (78 kB)
Collecting typing-inspect>=0.8.0 (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/65/f3/107a22063bf27bdccf2024833d3445f4eea42b2e598abfbd46f6a63b6cb0/typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
Collecting wrapt (from llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/42/5a/4efc997bccadd3af5749c250b49412793bc41e13a83a486b2b54a33e240c/wrapt-2.1.2-cp310-cp310-win_amd64.whl (60 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0f/15/5bf3b99495fb160b63f95972b81750f18f7f4e02ad051373b669d17d44f2/aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Collecting aiosignal>=1.4.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fb/76/641ae371508676492379f16e2fa48f4e2c11741bd63c48be4b12a6b09cba/aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Collecting async-timeout<6.0,>=4.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fe/ba/e2081de779ca30d473f21f5b30e0e737c438205440784c7dfc81efc2b029/async_timeout-5.0.1-py3-none-any.whl (6.2 kB)
Collecting attrs>=17.3.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/64/b4/17d4b0b2a2dc85a6df63d1157e028ed19f90d4cd97c36717afef2bc2f395/attrs-26.1.0-py3-none-any.whl (67 kB)
Collecting frozenlist>=1.1.1 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/66/aa/3f04523fb189a00e147e60c5b2205126118f216b0aa908035c45336e27e4/frozenlist-1.8.0-cp310-cp310-win_amd64.whl (43 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/68/00/9b38e272a770303692fc406c36e1a4c740f401522d5787691eb38a8925a8/multidict-6.7.1-cp310-cp310-win_amd64.whl (46 kB)
Collecting propcache>=0.2.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a6/67/354aac4e0603a15f76439caf0427781bcd6797f370377f75a642133bc954/propcache-0.4.1-cp310-cp310-win_amd64.whl (41 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp<4,>=3.8.6->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a0/71/ad95c33da18897e4c636528bbc24a1dd23fe16797de8bc4ec667b8db0ba4/yarl-1.23.0-cp310-cp310-win_amd64.whl (87 kB)
Collecting griffe (from banks<3,>=2.3.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8b/94/ee21d41e7eb4f823b94603b9d40f86d3c7fde80eacc2c3c71845476dddaa/griffe-2.0.0-py3-none-any.whl (5.2 kB)
Collecting openai>=1.1.0 (from llama-index-embeddings-openai<0.7,>=0.6.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d0/b1/35b6f9c8cf9318e3dbb7146cc82dab4cf61182a8d5406fc9b50864362895/openai-2.29.0-py3-none-any.whl (1.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 9.4 MB/s  0:00:00
Collecting llama-index-instrumentation>=0.4.3 (from llama-index-workflows!=2.9.0,<3,>=2->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c3/45/6dcaccef44e541ffa138e4b45e33e0d40ab2a7d845338483954fcf77bc75/llama_index_instrumentation-0.5.0-py3-none-any.whl (16 kB)
Collecting anyio<5,>=3.5.0 (from openai>=1.1.0->llama-index-embeddings-openai<0.7,>=0.6.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/38/0e/27be9fdef66e72d64c0cdc3cc2823101b80585f8119b5c112c2e8f5f7dab/anyio-4.12.1-py3-none-any.whl (113 kB)
Collecting distro<2,>=1.7.0 (from openai>=1.1.0->llama-index-embeddings-openai<0.7,>=0.6.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/12/b3/231ffd4ab1fc9d679809f356cebee130ac7daa00d6d6f3206dd4fd137e9e/distro-1.9.0-py3-none-any.whl (20 kB)
Collecting jiter<1,>=0.10.0 (from openai>=1.1.0->llama-index-embeddings-openai<0.7,>=0.6.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a7/38/f4f3ea5788b8a5bae7510a678cdc747eda0c45ffe534f9878ff37e7cf3b3/jiter-0.13.0-cp310-cp310-win_amd64.whl (206 kB)
Collecting sniffio (from openai>=1.1.0->llama-index-embeddings-openai<0.7,>=0.6.0->llama-index)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/e9/44/75a9c9421471a6c4805dbf2356f7c181a29c1879239abab1ea2cc8f38b40/sniffio-1.3.1-py3-none-any.whl (10 kB)
Collecting exceptiongroup>=1.0.2 (from anyio<5,>=3.5.0->openai>=1.1.0->llama-index-embeddings-openai<0.7,>=0.6.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8a/0e/97c33bf5009bdbac74fd2beace167cab3f978feb69cc36f1ef79360d6c4e/exceptiongroup-1.3.1-py3-none-any.whl (16 kB)
Collecting httpcore==1.* (from httpx->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7e/f5/f66802a942d491edb555dd61e3a9961140fd64c90bce1eafd741609d334d/httpcore-1.0.9-py3-none-any.whl (78 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl (37 kB)
Collecting annotated-types>=0.6.0 (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/78/b6/6307fbef88d9b5ee7421e68d78a9f162e0da4900bc5f5793f6d3d0e34fb8/annotated_types-0.7.0-py3-none-any.whl (13 kB)
Collecting pydantic-core==2.41.5 (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/76/92/d338652464c6c367e5608e4488201702cd1cbb0f33f7b6a85a60fe5f3720/pydantic_core-2.41.5-cp310-cp310-win_amd64.whl (2.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 12.4 MB/s  0:00:00
Collecting typing-inspection>=0.4.2 (from pydantic>=2.8.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dc/9b/47798a6c91d8bdb567fe2698fe81e0c6b7cb7ef4d13da4114b41d239f65d/typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Collecting ollama>=0.5.3 (from llama-index-llms-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/47/4f/4a617ee93d8208d2bcf26b2d8b9402ceaed03e3853c754940e2290fed063/ollama-0.6.1-py3-none-any.whl (14 kB)
Collecting pytest-asyncio>=0.23.8 (from llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e5/35/f8b19922b6a25bc0880171a2f1a003eaeb93657475193ab516fd87cac9da/pytest_asyncio-1.3.0-py3-none-any.whl (15 kB)
Collecting MarkupSafe>=2.0 (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d6/25/55dc3ab959917602c96985cb1253efaa4ff42f71194bddeb61eb7278b8be/markupsafe-3.0.3-cp310-cp310-win_amd64.whl (15 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/41/45/1a4ed80516f02155c51f51e8cedb3c1902296743db0bbc66608a0db2814f/jsonschema_specifications-2025.9.1-py3-none-any.whl (18 kB)
Collecting referencing>=0.28.4 (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2c/58/ca301544e1fa93ed4f80d724bf5b194f6e4b945841c5bfd555878eea9fcb/referencing-0.37.0-py3-none-any.whl (26 kB)
Collecting rpds-py>=0.25.0 (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/96/cb/156d7a5cf4f78a7cc571465d8aec7a3c447c94f6749c5123f08438bcf7bc/rpds_py-0.30.0-cp310-cp310-win_amd64.whl (235 kB)
Collecting llama-cloud<2,>=1.6.0 (from llama-index-indices-managed-llama-cloud>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e2/e1/c40aff3fe70c6b457ee89f6cdaee8678fb63986b12734e7d6e4fc481d6dc/llama_cloud-1.6.0-py3-none-any.whl (394 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bd/70/3d4cc14f99c80491401d4ab514f3ebe3113e38c8017cd384a73dc67b3ae4/llama_parse-0.6.94-py3-none-any.whl (5.4 kB)
Collecting llama-cloud-services>=0.6.94 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/14/ab/876486e4f1c137cfeca8f876abd18eeec35a66a0fd8adb15afba7b28aa8c/llama_cloud_services-0.6.94-py3-none-any.whl (77 kB)
INFO: pip is looking at multiple versions of llama-cloud-services to determine which version is compatible with other requirements. This could take a while.
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ff/b9/5cb3cdb8cdb92b8472f10ad06521b3eee465d7487df8ecebbe298391f7ab/llama_parse-0.6.93-py3-none-any.whl (5.4 kB)
Collecting llama-cloud-services>=0.6.93 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cc/c0/7a32802b096df3e720d17d1a53111d603f954c16d3ba10fff350422a3e81/llama_cloud_services-0.6.93-py3-none-any.whl (77 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e6/e4/ffe29044b47057c3a20e255f14b2ecd60c6a0927892102899d49bf6e9c4b/llama_parse-0.6.92-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.92 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ba/05/ff56b34fb93b6659a4455fcea12e83b077a234d25afa29303a2a9e272246/llama_cloud_services-0.6.92-py3-none-any.whl (76 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/45/41/969d279edea31afe9c424b40c46f714e153ee81ad26a93fdd5fba7aeaec1/llama_parse-0.6.91-py3-none-any.whl (4.9 kB)
INFO: pip is still looking at multiple versions of llama-cloud-services to determine which version is compatible with other requirements. This could take a while.
Collecting llama-cloud-services>=0.6.91 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8c/d6/a0a19cd5727483c5c6943a735a025c6c1695a8e876f061ae9d7496b6388b/llama_cloud_services-0.6.91-py3-none-any.whl (76 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/62/3c/e386901e8af6078a12a06cf9ddd37c87b57978fb845b782b5c5a2f249c2d/llama_parse-0.6.90-py3-none-any.whl (4.9 kB)
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
Collecting llama-cloud-services>=0.6.90 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/88/64/7c4a9e869c869136f147eb7ac5e408e613cc50faf15c3f651005b8b3998d/llama_cloud_services-0.6.90-py3-none-any.whl (76 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/20/53/7d0e556f94f40b533ca9a77cb01b453dbd146cb97489c8c79e107c1fde19/llama_parse-0.6.89-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.89 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1b/1c/5a4e025668b27a12f1d8111409c966aba8a217a42f264dd8c002869fd3c6/llama_cloud_services-0.6.89-py3-none-any.whl (76 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a8/46/88a5fb5dc218b69551d381101be39bcc978cf33f8462a16800c134fc028e/llama_parse-0.6.88-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.88 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e0/bf/da0943d97e0351c668254251cb2a15d60b88fe58393dd0b0fd1ec463f116/llama_cloud_services-0.6.88-py3-none-any.whl (76 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d5/f7/01adce2b7ab369458fc8a1631af40e0e81a46447df2cf455a64592a264b5/llama_parse-0.6.87-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.87 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/88/f3/8504b262a31a08b09f79d30950eb7ab7a6b01994fc87ec3f02758706714c/llama_cloud_services-0.6.87-py3-none-any.whl (75 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/66/78/53bba757f247e2c0eea316c9938b4c238af6938120ea200822662de9ae2e/llama_parse-0.6.86-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.86 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5e/91/75996f02081b2ed3c9d0e102d813e0e9a0cbc513135eded7b4734ab1f575/llama_cloud_services-0.6.86-py3-none-any.whl (75 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8e/78/59a80c0a874addd8ee1cfedcc50d0dc66497f49a14dff41724839b4c263d/llama_parse-0.6.85-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.85 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b9/7f/8f214756eda1950272aa41108bd80d9212a5f54a1c116879a16908a273b2/llama_cloud_services-0.6.85-py3-none-any.whl (75 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/06/24/244af08d18d6d3dbd8684c581bb298e5df484b36d78def23c5d5757bb4eb/llama_parse-0.6.84-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.84 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/48/3e/c0d4109de51e197f2f5f10632f3392ddfe328baf5b3d5559c86e2733c348/llama_cloud_services-0.6.84-py3-none-any.whl (75 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/04/f2/b2262e42a47fb71421e39fdfc0c84d10061997c03405d1f9dcce1a929704/llama_parse-0.6.83-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.82 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/63/0b/645dbff7ec8da76bedd14dd06aff0968ed10634690c8e5cba7fb2b2c42ab/llama_cloud_services-0.6.83-py3-none-any.whl (75 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dd/72/683a57f538db93d00bd216c5a0a4bc5b4fe456fc92971c71b576bebd4762/llama_cloud_services-0.6.82-py3-none-any.whl (74 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b3/6a/fd33291907efd269b25f96793aa0e52018c7e8db55780d0c1515991e9336/llama_parse-0.6.82-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3f/da/416a61d85139907b6f2d0b7dbe992d33e15722c83710f0ea927ce31c5454/llama_parse-0.6.81-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.81 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d7/eb/da65f4213ead0d033f24b84a2a1cc797ec86955f379588e200fff8866b48/llama_cloud_services-0.6.81-py3-none-any.whl (74 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c5/01/da3f3880f48017fb13d02051b6de1e51435d51864a865901f1cc0800fa5c/llama_parse-0.6.80-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.80 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/be/ef/fab768484cc132bc602cefc62658a46757b571329420b9b3f4ea211a9284/llama_cloud_services-0.6.80-py3-none-any.whl (74 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/87/be/ba5c2f844fa97c916e1da539526c496d38047f9a0fe99288d6b441eaadc6/llama_parse-0.6.79-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.79 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8b/c1/a20fdb066d12ef364e1123ff9ef35e23370123dd4d8ead7cf5afa9f862a1/llama_cloud_services-0.6.79-py3-none-any.whl (68 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8e/82/bd2f6880d4141536fe27f12103d38aec4caddbca3d8509208b6926860bcc/llama_parse-0.6.78-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.78 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0f/3b/8f1366e201e00a21b463275633d79e0733ae839b41b7a18e129d8fbf8c17/llama_cloud_services-0.6.78-py3-none-any.whl (68 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/86/61/7aa2f5d22066dccdc5548cdba93765d3273170a06adc3d425bd9fc5e54a6/llama_parse-0.6.77-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.77 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a8/e7/4728c39c6ad802ca7741e33121625a7ae5bf29b531a164e81bde51bd6fcd/llama_cloud_services-0.6.77-py3-none-any.whl (68 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/eb/a3/fc18027ba825ab0754805ccad074a99b09fcd61008c4a816c1820bd6663a/llama_parse-0.6.76-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.76 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c8/c6/bd3478ccfb4b9a2f8c1b415a7e328dc1564eb74cb642f352604cc8dfd6c7/llama_cloud_services-0.6.76-py3-none-any.whl (66 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c8/da/b81ac4da9c888d900995ac9a4fea33e5b6538b32658d935b381041b98fc1/llama_parse-0.6.75-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.75 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e0/28/bd8b6276131e625b49d5763ae5d8292759cf7f0bdbf0fbbc63b28de27008/llama_cloud_services-0.6.75-py3-none-any.whl (66 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c3/3e/391cc52449582a9bac501cd4b2074ad80ce19301a541b33d74512af75c24/llama_parse-0.6.74-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.74 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/67/96/bb21f5e48f1437db06d5307345bd58b3e258d8c1e690a71043744bf92f62/llama_cloud_services-0.6.74-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e1/2a/a000837b2df175ced093c499d34de54cc10c1919c8855ae6215e5a6e1d1c/llama_parse-0.6.73-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.73 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7e/97/671d75d0835d8f951f070f6c896ba43ae7b0833238132905ae40907d9462/llama_cloud_services-0.6.73-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fb/e8/d4c40292d3603ff46cd25a21b2580ad60325ec1d959bcd07e525daa8b036/llama_parse-0.6.72-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.72 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/9f/9888a0d8c5cd1508430061b73478fdfb9a1f589f9d5874c1617a5a388281/llama_cloud_services-0.6.72-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1e/9c/e05093bf692c9091bcb122fbe90cc8c90c59ecbb113a3f03c42d7cbe5424/llama_parse-0.6.71-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.71 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1c/59/3a775ccf7980bb24c1f22315b05cc3126173735c9e445047833f41fc7c28/llama_cloud_services-0.6.71-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/94/fd/f43328e798490782908144bb2957d769537be3ea3193a9787ea8628e1f39/llama_parse-0.6.70-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.70 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5c/74/589f2ce370981ca3094e29fc21d734c79cfb8ffbff268831985ddbd665c2/llama_cloud_services-0.6.70-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/14/13/b806e7ff362980374133c2293aa8f644971609d328d73af0c915a73971e8/llama_parse-0.6.69-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.69 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1d/9e/2bb19f77877b5dbe971be84f81227d8af374b31ab32cc46351c5f9f3e940/llama_cloud_services-0.6.69-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/64/80/6764feeca73bb14b10f36611496e9b8e7dc0e064e0e70a23590a8de4b058/llama_parse-0.6.68-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.68 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/28/44/a07c0dcda413ed0c201ad01aa39e7af62f2a554fc5a1ebaebf31fbef2149/llama_cloud_services-0.6.68-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/df/e9/0c14fd9f5aaacd1b12e60c112a86174026029eda8fffa6944724dc9b067e/llama_parse-0.6.67-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.67 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0b/22/4abf2218d4e8e60260b465e3dceaeb09749b05093c1f6254cf2e71c3deb5/llama_cloud_services-0.6.67-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9e/8b/1bfe0841ad39761acccb6cf86db0fec2998ee44be374b61546bbe42ae5e8/llama_parse-0.6.66-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.66 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/30/87/fe724bc4f8f046691f37a2b1fd38969cb85b4a8691f8548ac8e5ef5798ba/llama_cloud_services-0.6.66-py3-none-any.whl (65 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1a/8d/21e02488cb7c49903d30ed4e9eaefed2651e43d0c5a5922acfd79cd0d27a/llama_parse-0.6.65-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.64 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b1/bc/0c3d412981b20fd94271c067bd223f91aca52d35c393ab361ba879eddefe/llama_cloud_services-0.6.65-py3-none-any.whl (65 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/76/ef/c5fc98d32078a866257e4ed1bbf316e075c88c052b33156b2591470ad696/llama_cloud_services-0.6.64-py3-none-any.whl (64 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3a/5d/e7d1f53f5770a03482e95f17d3d32333795a08495742dfe236db2ecb798e/llama_parse-0.6.64-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/52/95/80d22ea0b016843e42036eff8492eb9323ed0206bb50c94e8c1fda4c133d/llama_parse-0.6.63-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.63 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/13/93/b8afcb0add44e46ca1eccb6b377b97569abdebc8b490bcab3529fa388a64/llama_cloud_services-0.6.63-py3-none-any.whl (71 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/49/58/5b487cc80ffece36e5253c2993870cae1fe102e5f2b8cdd826a746df94be/llama_parse-0.6.62-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.62 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/82/ee/02a531def0c95940a02213d13cdb01f00fca787fb60edef80e925c932912/llama_cloud_services-0.6.62-py3-none-any.whl (70 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/16/aa/ae995f301938071ef458fd4a8b37ad2dad283f284fe2ea012db3a0f09ea7/llama_parse-0.6.60-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.60 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2f/f4/f5626a8e4538faff76d2b8bdee4b0d606b32ae2d411100de4c9e66e4ea09/llama_cloud_services-0.6.60-py3-none-any.whl (70 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/fc/52fd8ef1a15d76f70e84dc9ca176d070053dd225721b196f5400948c975d/llama_parse-0.6.59-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.59 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/04/6843856a237b9db761101acaa86713bdd99225b591659777489acaabd49b/llama_cloud_services-0.6.59-py3-none-any.whl (70 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/05/54998c1657cc2cc69397517498c9c73841f815bba2a88a50309e7b1facb6/llama_parse-0.6.58-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.58 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b4/24/3e114f0aff8e704410032bd02130a750e1d33f1cdd8a38e703b359b4a0ad/llama_cloud_services-0.6.58-py3-none-any.whl (70 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/35/c0/206638805e069aae9e4394be98b6a0984bbd35b83cf630260911ac674b7a/llama_parse-0.6.57-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.56 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a7/93/f7ea6978d943a0fcb316c72e7417883a6e401501ba03cb668002ca91d5d1/llama_cloud_services-0.6.57-py3-none-any.whl (67 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bc/dd/e230f7a8ac877ff141bb207cc19db72284f132fa8bf360c3f07a4b8c1c22/llama_cloud_services-0.6.56-py3-none-any.whl (66 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8e/e1/ffa70c6e96a27fdfb52a3799300efdcb33b3b39acda5d4275ec78dcb64c2/llama_parse-0.6.56-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/64/44/96f8cb4ce5e9d043c25122439e681a1d7531ceefc50267a135d691e4bdc4/llama_parse-0.6.55-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.55 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4e/e5/18e611d9832bdbf55a47f72ee267aa3db3d0475657b0c2479a4347903a29/llama_cloud_services-0.6.55-py3-none-any.whl (66 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/05/50/c5ccd2a50daa0a10c7f3f7d4e6992392454198cd8a7d99fcb96cb60d0686/llama_parse-0.6.54-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.54 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7f/48/4e295e3f791b279885a2e584f71e75cbe4ac84e93bba3c36e2668f60a8ac/llama_cloud_services-0.6.54-py3-none-any.whl (63 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/68/8d/de8f3ea41403265a02d9277112ed69539081694fdcd364be0549f9793410/llama_parse-0.6.53-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.53 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/60/e2/f30d6e5b8076385ff0116d5ad0711c8fd4e96d3dd772096be2581023ea66/llama_cloud_services-0.6.53-py3-none-any.whl (50 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9b/3f/473b40cdb33976a44760f2e0ba268129e126178f2f5b0cc5411e63552ca6/llama_parse-0.6.52-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.52 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bb/32/c911445c309bbc9099fb35ed23434ecf868a14234321bbbac40431236fbd/llama_cloud_services-0.6.52-py3-none-any.whl (48 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/48/35/72039f1ae4c1a22786d55f471cf421a0e3b585eab574ea81c79f7f77b06d/llama_parse-0.6.51-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.51 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/01/b4/b3e876ddf2854d65010a5a2dcb6dbd01e4d6fa88adcd6d141d03d36c2a18/llama_cloud_services-0.6.51-py3-none-any.whl (48 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0a/fc/703e1f6dcd40bd062fcc2117919f283acae6aca08ff140837d415932737c/llama_parse-0.6.50-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.49 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4e/a7/3563873ab3686f9a8de79294c5cbb862dd07be0c76502b73f1788ab22812/llama_cloud_services-0.6.50-py3-none-any.whl (48 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7e/c4/256a03b6e48fdd2401d731576b67c38e9394554fa99ae5c8bc37fb672e4a/llama_cloud_services-0.6.49-py3-none-any.whl (48 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fb/7c/f2460b5cc2af43638194f864093572ce16cd1814204bb67d65a4bdd4df40/llama_parse-0.6.49-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/32/ea/112f646e17331e88e864b5b526802ef909ee3e8e50f614cdca10499c5569/llama_parse-0.6.48-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.48 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/06/1c/80ecab634d5915b3bf933201be1ae163f66b02ba349ac6faa7d6444232ea/llama_cloud_services-0.6.48-py3-none-any.whl (48 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fc/b0/619721729668d9b981cd19be238c738f9e05eee0c79071282386d8683c4c/llama_parse-0.6.47-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.47 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9c/7c/0eba363d868309b3e83e2cb75e959215cea04eae48cc256d38b44807879d/llama_cloud_services-0.6.47-py3-none-any.whl (48 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/dc/f8/024695a0cc768561122b162f9c852b3e5fdca6abd9499de7c49ac2cca1f4/llama_parse-0.6.46-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.45 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/44/33/29a9ffde4dd395d92bc0dab759db139d111cbd0e5e08bb9cfcd1b00a1412/llama_cloud_services-0.6.46-py3-none-any.whl (47 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0c/47/23176020239524a4d5794c783b21cdc5cd2473d5a435da825e832025b727/llama_cloud_services-0.6.45-py3-none-any.whl (47 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/77/c7/9ef608a9ec3f26f1358129458ef94ca730dfa93e69867d8ef918373dcb91/llama_parse-0.6.45-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/34/b6/c22463bbf23c50290f1c5fd9feefaa08dcbe51749604734ee6856b2338b7/llama_parse-0.6.44-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.44 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d2/68/da660bd22edfd20b4b4e47fbb80b3157ec5423c68e4e6a93b5a47b6ec9d2/llama_cloud_services-0.6.44-py3-none-any.whl (47 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fa/24/8497595be04a8a0209536e9ce70d4132f8f8e001986f4c700414b3777758/llama_parse-0.6.43-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.43 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/40/c3/1e82bd37f16e6d99988e9b53829303d3124504341862721f24c9b6f4f614/llama_cloud_services-0.6.43-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d1/17/cff4dcb82bff817a7612dea18c3f16dff37a8c4e9cbfb5ba8eb9711d53c2/llama_parse-0.6.42-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.42 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/aa/56/51a25a9d54fdf5b22af9491c4c11118b19e54ea26eadcc9173517b8a2fd2/llama_cloud_services-0.6.42-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/41/31/5302eb8412ce309281db0725b1acb8cc8069190ff9c458b0187250a9769a/llama_parse-0.6.41-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.41 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6a/b6/6a90ed0ecffabf7067c5f8206e3a1e3f6136bf8c3c37529df5f6090795b6/llama_cloud_services-0.6.41-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bb/08/37f98d1383a68966710e87d0f04236af84b5df520dd6937b7a987d53d395/llama_parse-0.6.40-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.40 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/74/15/564fa345f051f78595dc21172a0fde51f166661c2ce15ebfe68b5ec052fb/llama_cloud_services-0.6.40-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d8/a0/cc1350e2e00b8653165320dc4bfe66437e5db8b4011604f3d143dfc419e7/llama_parse-0.6.39-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.39 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e3/e1/31aa6b4b666651e3bebf8b14f9a021397e65d2120484979937b39ab63a8d/llama_cloud_services-0.6.39-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8b/0d/07fb26b8fd872c4407a41b982d2f2f5a07b30a79d397496722b7e8ef9b49/llama_parse-0.6.38-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.37 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a1/45/b22e1c0f347ff4adaa4e2c4c32d654223b3a34bfa5e1b93e8220a578fe24/llama_cloud_services-0.6.38-py3-none-any.whl (40 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/86/7d/76f54e20d04315accccd1d1e5e741fa29784167fc5784f8c2e8f730cb9f7/llama_cloud_services-0.6.37-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cb/68/cc75c39d39d0c052eddcf4c5993250183deb6fb1077960b26d279c0257cb/llama_parse-0.6.37-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/04/a2/44eba429e93e525eeb83163aa1bada65dd7d2e4b4b43f79c9a7b8e818c0e/llama_parse-0.6.36-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.36 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/62/c6/5a78aca08901991a4fde281b870766952e350ade0026b9f9e47debfe63b6/llama_cloud_services-0.6.36-py3-none-any.whl (40 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a9/9b/24f4644bcf5652f1d5d82c52e2bbaf92f0ebd2b5632c78d540518ed3d76b/llama_parse-0.6.35-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.35 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bc/94/5f9d5c6d9b4fc11c88e0ea7cc5e4b5220bff7b9822f655d3f4686e57b902/llama_cloud_services-0.6.35-py3-none-any.whl (39 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/50/24/af0d46874ec18d240fe200ce3902044f6d1d373067433c2b9ade5dc217d4/llama_parse-0.6.34-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.32 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/df/36/088caee44cbafab71c9b329c056654dfd0a1fa9834e8aca3e658f61c17cb/llama_cloud_services-0.6.34-py3-none-any.whl (39 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/33/a7/faa71cda12dfb4ccd044e1e5d42a2cba7b190d7d4430a509369430756236/llama_cloud_services-0.6.33-py3-none-any.whl (39 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f2/81/9e543da4bfe31a26479b76df773e6e1d6b61c0ffe658e5353e75eaf9fbec/llama_cloud_services-0.6.32-py3-none-any.whl (39 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b7/2c/373035cc17509bc51fb6eb156eccc153221838ba0806516306d5ba6affad/llama_parse-0.6.33-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/17/82/6bb804103e8d6b9d6ed72c9418bcea568b263ce5a0c9045839d9ae4cda61/llama_parse-0.6.32-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5d/17/2f4a43837ac96f45e080c1db2000842c7b5ce8bda9267ca261481ad50ab0/llama_parse-0.6.31-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.31 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/86/bc/7983c76278bac65505e92302cbe14e1c529c12dfc47782a689f9568c3249/llama_cloud_services-0.6.31-py3-none-any.whl (38 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f3/d9/a375fe968789c874b2b47afd505a0ac3cf225077fb417d00ac43baf6e07d/llama_parse-0.6.30-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.30 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d0/79/815053ed66845e099f5369871d23fa925465dfd8e730a06f208e082d4239/llama_cloud_services-0.6.30-py3-none-any.whl (38 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/03/2061267a578864eea22ee1c2c86732fa14d54bb4b643800cdc460f8d953a/llama_parse-0.6.28-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.28 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6a/c2/8132c4b6eaaef6d84fec1ca1f29876b083d74cdef68f10cf7c99e74b8e07/llama_cloud_services-0.6.29-py3-none-any.whl (39 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/37/98/e21e35d8643930abad95f33c7b1a69578408e5ae34459aab5edfb03cf7c1/llama_cloud_services-0.6.28-py3-none-any.whl (38 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ec/e9/ae39adba29819b6376339b6e3cd701ae866e4d5df34e1ba680e9065ea680/llama_parse-0.6.27-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.27 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f4/e4/633e0e0bab1909fb42177be969148df25648d765a5868540574cfc1f03e0/llama_cloud_services-0.6.27-py3-none-any.whl (38 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/01/44/05ebad423feee96f04a73bf39ad81f59a45dbcbbfff0923bb28679266ad9/llama_parse-0.6.26-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.26 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8d/f0/781a8ac37e443c89079f251fd4bf5d57708adfc97e3e494eb2bfefadf207/llama_cloud_services-0.6.26-py3-none-any.whl (37 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/56/14/ff4f65a013e6b3ee88c1dece2de6ca57b1e6567838cab2e1e545e5693b1b/llama_parse-0.6.25-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.24 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e6/f1/99b8ef4a636dafd5f1ae1e1b19eb9f793f51573d782919bf01d9b9f797f4/llama_cloud_services-0.6.25-py3-none-any.whl (37 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3a/45/37d8109b33ce30d9e746c4529eda3b2c7b23ca4628a45bdfd9915bae0281/llama_cloud_services-0.6.24-py3-none-any.whl (37 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5a/70/a555f96186c00e22487c495e38b3b7afcc02011c8a10066d949e8b6b25b1/llama_parse-0.6.24-py3-none-any.whl (4.9 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cd/53/a7e06181f36eaff6c7176e33644db2f19a7761fc9110f0a5d13759ce94ac/llama_parse-0.6.23-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.23 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/12/43/d1e53b856bc9a96548dea06ee96a6b61260d30ae1a52e885e63c13e47e84/llama_cloud_services-0.6.23-py3-none-any.whl (37 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/16/ed/759a566cb7da5a8f771aabdcd19628e33da605d7133a2773eeb5cbaa1b5b/llama_parse-0.6.22-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.22 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/24/26/cb45d9126ccf45376ea074ab472a0e55fc9f8dba8f113ac61a583acacb4a/llama_cloud_services-0.6.22-py3-none-any.whl (37 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/67/72/cccdac06596fb1dcc53bb8114580d80081f0043e065bea468c935cbeec7e/llama_parse-0.6.21-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.21 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/91/12/8485afc9813574f0a1c9ca28df5dffa10e2fefa6cefddc73f4b3e36f0a5e/llama_cloud_services-0.6.21-py3-none-any.whl (37 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/88/90/7880cde59e61841a7f254bf22f1706c48cbd2453584255324bb32084abb2/llama_parse-0.6.20-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.20 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cf/c0/b5195393895e7ad5ac1a621712151d5ab5541d52b2e438aaae60fc9d1c08/llama_cloud_services-0.6.20-py3-none-any.whl (37 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/14/51/1158a72abfe8c245fae13c02304cb5b9a940c4464e151acfc71f26f1788f/llama_parse-0.6.18-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.17 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fe/60/749a2d733d582f5fbc89621f4a54f728cffd44c0ce1a4c997b6937c02c4b/llama_cloud_services-0.6.19-py3-none-any.whl (37 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/43/0c/b42f863d400e787c5314cb7ff6aec77358e3df33b0cc2b90560a068e5230/llama_cloud_services-0.6.18-py3-none-any.whl (37 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5a/03/7e2ff9570d3bbdbf71a132cd908346f790572a23bde00392cf7b25de79e3/llama_cloud_services-0.6.17-py3-none-any.whl (36 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f8/ab/46c9ee88c340be86b146ac1f11d02d5ef93b136617181ce87058e98f51c5/llama_parse-0.6.16-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.16 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e9/e4/c784e95c7deb37a6d27275f40f4b4d3d04ae65f522b3d9f9f8c7f9ddb366/llama_cloud_services-0.6.16-py3-none-any.whl (36 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ac/ca/71c9367d3e89d61da2462f535dea1a3a09d4a4085b96f2c9ef5c38864820/llama_parse-0.6.12-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.12 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f5/0d/88805be6a13b368c9e7a2b2cede60fd0298e0e3abc9a6a6923d414c1ab14/llama_cloud_services-0.6.15-py3-none-any.whl (36 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f7/3b/f91b2ed0d6b7ebc574af224191414e377dc23b0c282f662de621d8acde42/llama_cloud_services-0.6.14-py3-none-any.whl (36 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b5/8b/7bd898ff15873426ed6c74b477b72f735bbc3cedff1ac6d309bec11b970e/llama_cloud_services-0.6.12-py3-none-any.whl (36 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3d/c4/af03ad2b8cc56ed4a3625dd524d87b6b1640b3d8861bc7946df8a5d393b4/llama_parse-0.6.9-py3-none-any.whl (4.8 kB)
Collecting llama-cloud-services>=0.6.9 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a8/db/052d85feff68774552b9fa78c93d140090933313493d44622ac222c96fb0/llama_cloud_services-0.6.11-py3-none-any.whl (30 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/05/9d/e83b78fd54de63fd136fb1582c7eafd065c71a4d6a317cf83de0726f700a/llama_cloud_services-0.6.10-py3-none-any.whl (30 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f3/8f/446aba7d8a378fddfde6d1464f1cbbb015bbb19af1f765dc669f5b7aac23/llama_cloud_services-0.6.9-py3-none-any.whl (29 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/77/d5/9e360e31fb5852842c434043c2a7ba6714a363546b5adeedff537fae5e0d/llama_parse-0.6.4.post1-py3-none-any.whl (4.9 kB)
Collecting llama-cloud-services>=0.6.4 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/62/05/8db18cb6c2204170bf34ac85def1e40895667036df55e29482f26ddb6c00/llama_cloud_services-0.6.8-py3-none-any.whl (29 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/81/54/8ce78621c92c16636d8a211a5b2fd29ed48fffe27d373685abf57ce63652/llama_cloud_services-0.6.7-py3-none-any.whl (28 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/96/dd/b0525c55eba864335ad1d13498b27ecd2f2b9fec2b9608fd5a9a5cd6cc3a/llama_cloud_services-0.6.6-py3-none-any.whl (28 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bb/0c/47cbb539eadd1cf48e9588d8925807041690a9dceb5c56a394abfa622022/llama_cloud_services-0.6.5-py3-none-any.whl (28 kB)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/39/2e/8d1889bea33a1c7ec68773299be8a8b7b8d7c9c2f8b30590b817124303af/llama_cloud_services-0.6.4-py3-none-any.whl (28 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/04/6c/f763ae09c103becb9db821910c0e6f57979267688f4b8ef3710b49fb529d/llama_parse-0.6.4-py3-none-any.whl (4.8 kB)
Collecting llama-cloud-services>=0.6.3 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/93/77/1c76b4cb2c44c33dec804dac32d4b75e66c451640c1ba0486c5d9f94ebf8/llama_cloud_services-0.6.3-py3-none-any.whl (28 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d1/dc/b78717e1c1251cf1b7419e24c8d60d66a5732d5db47393e02f3f63ae5e86/llama_parse-0.6.2-py3-none-any.whl (4.8 kB)
Collecting llama-cloud-services>=0.6.2 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c1/55/2ceefe03c5d5c410c730ec2c5bda189311d6bad9fdde79f54c26aa49f653/llama_cloud_services-0.6.2-py3-none-any.whl (22 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f4/6a/1052d859c974823e4ff817c129352e09d1b1fd1fd7280a9bbf1e47bb437b/llama_parse-0.6.1-py3-none-any.whl (4.8 kB)
Collecting llama-cloud-services>=0.6.1 (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5e/a4/2ac5b002f2fc701f6aab63f269537c1912d8890f575c1af1031a7b50149f/llama_cloud_services-0.6.1-py3-none-any.whl (22 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4c/55/29d813069e00b823d5983f29a1668071bc731cea0cbc6cf70f7ff069d320/llama_parse-0.6.0-py3-none-any.whl (4.8 kB)
Collecting llama-cloud-services (from llama-parse>=0.5.0->llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8d/d6/b7354400d8e63a110512d740d2955b32525ee0eb6365bb6eb82f7d432d8d/llama_cloud_services-0.6.0-py3-none-any.whl (22 kB)
Collecting llama-parse>=0.5.0 (from llama-index-readers-llama-parse>=0.4.0->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8c/7c/203b7ffc633b9c0823f0d0701e361e002b93bf4e493f4c494d4bd5934c0b/llama_parse-0.5.20-py3-none-any.whl (16 kB)
Collecting joblib (from nltk>3.8.1->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7b/91/984aca2ec129e2757d1e4e3c81c3fcda9d0f85b74670a094cc443d9ee949/joblib-1.5.3-py3-none-any.whl (309 kB)
Collecting regex>=2021.8.3 (from nltk>3.8.1->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4d/0a/205c4c1466a36e04d90afcd01d8908bac327673050c7fe316b2416d99d3d/regex-2026.2.28-cp310-cp310-win_amd64.whl (277 kB)
Collecting backports-asyncio-runner<2,>=1.1 (from pytest-asyncio>=0.23.8->llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a0/59/76ab57e3fe74484f48a53f8e337171b4a2349e506eabe136d7e01d059086/backports_asyncio_runner-1.2.0-py3-none-any.whl (12 kB)
Collecting pytest<10,>=8.2 (from pytest-asyncio>=0.23.8->llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/ab/b3226f0bd7cdcf710fbede2b3548584366da3b19b5021e74f5bde2a8fa3f/pytest-9.0.2-py3-none-any.whl (374 kB)
Collecting iniconfig>=1.0.1 (from pytest<10,>=8.2->pytest-asyncio>=0.23.8->llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cb/b1/3846dd7f199d53cb17f49cba7e651e9ce294d8497c8c150530ed11865bb8/iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Collecting pluggy<2,>=1.5 (from pytest<10,>=8.2->pytest-asyncio>=0.23.8->llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/54/20/4d324d65cc6d9205fabedc306948156824eb9f0ee1633355a8f7ec5c66bf/pluggy-1.6.0-py3-none-any.whl (20 kB)
Collecting pygments>=2.7.2 (from pytest<10,>=8.2->pytest-asyncio>=0.23.8->llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c7/21/705964c7812476f378728bdf590ca4b771ec72385c533964653c68e86bdc/pygments-2.19.2-py3-none-any.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 8.8 MB/s  0:00:00
Collecting tomli>=1 (from pytest<10,>=8.2->pytest-asyncio>=0.23.8->llama-index-embeddings-ollama)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/23/d1/136eb2cb77520a31e1f64cbae9d33ec6df0d78bdf4160398e86eec8a8754/tomli-2.4.0-py3-none-any.whl (14 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl (11 kB)
Collecting greenlet>=1 (from sqlalchemy>=1.4.49->sqlalchemy[asyncio]>=1.4.49->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/ac/78/f93e840cbaef8becaf6adafbaf1319682a6c2d8c1c20224267a5c6c8c891/greenlet-3.3.2-cp310-cp310-win_amd64.whl (230 kB)
Collecting mypy-extensions>=0.3.0 (from typing-inspect>=0.8.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/79/7b/2c79738432f5c924bef5071f933bcc9efd0473bac3b4aa584a6f7c1c8df8/mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/be/2f/5108cb3ee4ba6501748c4908b908e55f42a5b66245b4cfe0c99326e1ef6e/marshmallow-3.26.2-py3-none-any.whl (50 kB)
Collecting griffecli==2.0.0 (from griffe->banks<3,>=2.3.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e6/ed/d93f7a447bbf7a935d8868e9617cbe1cadf9ee9ee6bd275d3040fbf93d60/griffecli-2.0.0-py3-none-any.whl (9.3 kB)
Collecting griffelib==2.0.0 (from griffe->banks<3,>=2.3.0->llama-index-core<0.15.0,>=0.14.18->llama-index)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4d/51/c936033e16d12b627ea334aaaaf42229c37620d0f15593456ab69ab48161/griffelib-2.0.0-py3-none-any.whl (142 kB)
Installing collected packages: pytz, filetype, dirtyjson, wrapt, watchdog, urllib3, tzdata, typing-extensions, tornado, tomli, toml, tinytag, tenacity, sniffio, smmap, six, rpds-py, regex, pyyaml, pygments, pyarrow, protobuf, propcache, pluggy, platformdirs, pillow, numpy, networkx, nest-asyncio, narwhals, mypy-extensions, marshmallow, MarkupSafe, joblib, jiter, iniconfig, idna, h11, griffelib, greenlet, fsspec, frozenlist, distro, colorama, charset_normalizer, certifi, cachetools, blinker, backports-asyncio-runner, attrs, async-timeout, annotated-types, aiosqlite, aiohappyeyeballs, typing-inspection, typing-inspect, tqdm, sqlalchemy, requests, referencing, python-dateutil, pydantic-core, multidict, jinja2, httpcore, griffecli, gitdb, exceptiongroup, deprecated, click, aiosignal, yarl, tiktoken, pytest, pydeck, pydantic, pandas, nltk, jsonschema-specifications, griffe, gitpython, dataclasses-json, anyio, pytest-asyncio, llama-index-instrumentation, jsonschema, httpx, banks, aiohttp, openai, ollama, llama-index-workflows, llama-cloud, altair, streamlit, llama-index-core, llama-parse, llama-index-llms-openai, llama-index-llms-ollama, llama-index-indices-managed-llama-cloud, llama-index-embeddings-openai, llama-index-embeddings-ollama, llama-index-readers-llama-parse, llama-index-cli, llama-index
Successfully installed MarkupSafe-3.0.3 aiohappyeyeballs-2.6.1 aiohttp-3.13.3 aiosignal-1.4.0 aiosqlite-0.22.1 altair-6.0.0 annotated-types-0.7.0 anyio-4.12.1 async-timeout-5.0.1 attrs-26.1.0 backports-asyncio-runner-1.2.0 banks-2.4.1 blinker-1.9.0 cachetools-7.0.5 certifi-2026.2.25 charset_normalizer-3.4.6 click-8.3.1 colorama-0.4.6 dataclasses-json-0.6.7 deprecated-1.3.1 dirtyjson-1.0.8 distro-1.9.0 exceptiongroup-1.3.1 filetype-1.2.0 frozenlist-1.8.0 fsspec-2026.2.0 gitdb-4.0.12 gitpython-3.1.46 greenlet-3.3.2 griffe-2.0.0 griffecli-2.0.0 griffelib-2.0.0 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.11 iniconfig-2.3.0 jinja2-3.1.6 jiter-0.13.0 joblib-1.5.3 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 llama-cloud-1.6.0 llama-index-0.14.18 llama-index-cli-0.5.6 llama-index-core-0.14.18 llama-index-embeddings-ollama-0.9.0 llama-index-embeddings-openai-0.6.0 llama-index-indices-managed-llama-cloud-0.11.0 llama-index-instrumentation-0.5.0 llama-index-llms-ollama-0.10.1 llama-index-llms-openai-0.7.3 llama-index-readers-llama-parse-0.6.0 llama-index-workflows-2.17.1 llama-parse-0.5.20 marshmallow-3.26.2 multidict-6.7.1 mypy-extensions-1.1.0 narwhals-2.18.0 nest-asyncio-1.6.0 networkx-3.4.2 nltk-3.9.3 numpy-2.2.6 ollama-0.6.1 openai-2.29.0 pandas-2.3.3 pillow-12.1.1 platformdirs-4.9.4 pluggy-1.6.0 propcache-0.4.1 protobuf-6.33.6 pyarrow-23.0.1 pydantic-2.12.5 pydantic-core-2.41.5 pydeck-0.9.1 pygments-2.19.2 pytest-9.0.2 pytest-asyncio-1.3.0 python-dateutil-2.9.0.post0 pytz-2026.1.post1 pyyaml-6.0.3 referencing-0.37.0 regex-2026.2.28 requests-2.32.5 rpds-py-0.30.0 six-1.17.0 smmap-5.0.3 sniffio-1.3.1 sqlalchemy-2.0.48 streamlit-1.55.0 tenacity-9.1.4 tiktoken-0.12.0 tinytag-2.2.1 toml-0.10.2 tomli-2.4.0 tornado-6.5.5 tqdm-4.67.3 typing-extensions-4.15.0 typing-inspect-0.9.0 typing-inspection-0.4.2 tzdata-2025.3 urllib3-2.6.3 watchdog-6.0.0 wrapt-2.1.2 yarl-1.23.0
```


```
(local_agent_env) PS D:\MyHyperspectralAgent> ollama run qwen2.5:7b
pulling manifest
pulling 2bada8a74506: 100% ▕██████████████████████████████████████████████████████████▏ 4.7 GB
pulling 66b9ea09bd5b: 100% ▕██████████████████████████████████████████████████████████▏   68 B
pulling eb4402837c78: 100% ▕██████████████████████████████████████████████████████████▏ 1.5 KB
pulling 832dd9e00a68: 100% ▕██████████████████████████████████████████████████████████▏  11 KB
pulling 2f15b3218f05: 100% ▕██████████████████████████████████████████████████████████▏  487 B
verifying sha256 digest
writing manifest
success
>>>
>>> ?
您好！看起来您的问题被省略了。能告诉我您具体需要帮助或了解什么吗？我会尽力提供帮助的。

>>> ollama pull nomic-embed-text
看来你可能是在尝试使用 `ollama` 和 `nomic-embed-text`，但没有给出具体的上下文或命令细节。假设你是想要拉取某个特定仓
库中的代码或者运行某种命令，请提供更多的详细信息，比如：

1. 你在哪个环境中操作（例如，是本地开发环境还是服务器）？
2. 你希望执行的具体步骤是什么？
3. 是否有相关的错误消息？

我可以根据这些信息为你提供更具体的帮助。如果有具体的操作步骤或遇到的问题描述，请告诉我！

>>> 如果我断网，你还能运行吗
当然可以！即使你在没有网络连接的情况下，我也能够继续与你对话和提供帮助。你可以随时提问，我会尽力回答你的问题。如果你
有任何需要解决的技术问题、查询信息或者其他需求，尽管告诉我吧！

例如：
- 你需要调试代码吗？
- 需要了解某个概念或技术的详细信息吗？
- 还是说你在遇到其他具体的难题？

请告诉我具体的情况，我将竭尽全力帮助你！

>>> 你的水平如何？、
我可以自信地说，在我的能力范围内，我可以提供广泛的知识和帮助。无论你是需要编程方面的指导，还是想了解科学知识、历史事
件、技术问题的解答，甚至是创意写作建议，我都尽力为你提供准确、有用的信息。

以下是一些我能胜任的任务示例：

1. **编程和技术支持**：解决代码问题、解释编程概念、推荐最佳实践等。
2. **学习和研究**：回答学术问题、提供参考资料、解释复杂的理论或公式等。
3. **创意写作**：帮助构思故事、提供写作建议、校对文本等。
4. **日常知识**：解答日常生活中的常见问题，如健康、饮食、旅行建议等。

如果你有任何具体的问题或需要帮助的地方，请告诉我，我会尽力为你提供支持。

>>> 你可以联网吗
作为一个由阿里云开发的AI助手，我并不实际依赖网络连接来运行和提供服务。这意味着即使你在没有网络的情况下，我也能够继续
与你对话并提供帮助。

不过需要注意的是，我的一些功能可能会受到最新数据限制的影响。例如：

- **实时信息查询**：如最新的新闻、天气预报等，这些可能不会实时更新。
- **在线资源链接**：我无法直接提供外部网站的链接或访问网络内容。
```

```
(base) PS C:\Users\lenovo> cd D:
(base) PS D:\>  cd MyHyperspectralAgent
(base) PS D:\MyHyperspectralAgent> conda activate local_agent_env
(local_agent_env) PS D:\MyHyperspectralAgent> ollama pull nomic-embed-text
pulling manifest
pulling 970aa74c0a90: 100% ▕█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏ 274 MB
pulling c71d239df917: 100% ▕█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  11 KB
pulling ce4a164fc046: 100% ▕█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏   17 B
pulling 31df23ea7daa: 100% ▕█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  420 B
verifying sha256 digest
writing manifest
success
```

```
(base) PS C:\Users\lenovo> cd D:
(base) PS D:\> cd MyHyperspectralAgent
(base) PS D:\MyHyperspectralAgent> conda activate local_agent_env
(local_agent_env) PS D:\MyHyperspectralAgent> pip install chromadb llama-index-vector-stores-chroma
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting chromadb
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a2/df/ce1ffcc0ad3eef8bd35b920809b990e6925ba94b2580dc5bd7ccde0fc06a/chromadb-1.5.5-cp39-abi3-win_amd64.whl (21.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.9/21.9 MB 13.4 MB/s  0:00:01
Collecting llama-index-vector-stores-chroma
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a1/3e/eced8b70b63067bf1600faa003f5bcc11c67ce39fcb6efb9aa38ebcc3b05/llama_index_vector_stores_chroma-0.5.5-py3-none-any.whl (8.1 kB)
Collecting build>=1.0.3 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c5/0d/84a4380f930db0010168e0aa7b7a8fed9ba1835a8fbb1472bc6d0201d529/build-1.4.0-py3-none-any.whl (24 kB)
Requirement already satisfied: pydantic>=2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (2.12.5)
Collecting pydantic-settings>=2.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/00/4b/ccc026168948fec4f7555b9164c724cf4125eac006e176541483d2c959be/pydantic_settings-2.13.1-py3-none-any.whl (58 kB)
Collecting pybase64>=1.4.1 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/79/28/c169a769fe90128f16d394aad87b2096dd4bf2f035ae0927108a46b617df/pybase64-1.4.3-cp310-cp310-win_amd64.whl (35 kB)
Collecting uvicorn>=0.18.3 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0a/89/f8827ccff89c1586027a105e5630ff6139a64da2515e24dafe860bd9ae4d/uvicorn-0.42.0-py3-none-any.whl (68 kB)
Requirement already satisfied: numpy>=1.22.5 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (2.2.6)
Requirement already satisfied: typing-extensions>=4.5.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (4.15.0)
Collecting onnxruntime>=1.14.1 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cd/6d/738e50c47c2fd285b1e6c8083f15dac1a5f6199213378a5f14092497296d/onnxruntime-1.23.2-cp310-cp310-win_amd64.whl (13.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.5/13.5 MB 14.1 MB/s  0:00:00
Collecting opentelemetry-api>=1.2.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5f/bf/93795954016c522008da367da292adceed71cca6ee1717e1d64c83089099/opentelemetry_api-1.40.0-py3-none-any.whl (68 kB)
Collecting opentelemetry-exporter-otlp-proto-grpc>=1.2.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/96/6f/7ee0980afcbdcd2d40362da16f7f9796bd083bf7f0b8e038abfbc0300f5d/opentelemetry_exporter_otlp_proto_grpc-1.40.0-py3-none-any.whl (20 kB)
Collecting opentelemetry-sdk>=1.2.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2c/c5/6a852903d8bfac758c6dc6e9a68b015d3c33f2f1be5e9591e0f4b69c7e0a/opentelemetry_sdk-1.40.0-py3-none-any.whl (141 kB)
Collecting tokenizers>=0.13.2 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/65/71/0670843133a43d43070abeb1949abfdef12a86d490bea9cd9e18e37c5ff7/tokenizers-0.22.2-cp39-abi3-win_amd64.whl (2.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB 11.3 MB/s  0:00:00
Collecting pypika>=0.48.9 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/57/83/c77dfeed04022e8930b08eedca2b6e5efed256ab3321396fde90066efb65/pypika-0.51.1-py2.py3-none-any.whl (60 kB)
Requirement already satisfied: tqdm>=4.65.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (4.67.3)
Collecting overrides>=7.3.1 (from chromadb)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/2c/ab/fc8290c6a4c722e5514d80f62b2dc4c4df1a68a41d1364e625c35990fcf3/overrides-7.7.0-py3-none-any.whl (17 kB)
Collecting importlib-resources (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a4/ed/1f1afb2e9e7f38a545d628f864d562a5ae64fe6f7a10e28ffb9b185b4e89/importlib_resources-6.5.2-py3-none-any.whl (37 kB)
Collecting grpcio>=1.58.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2a/32/4ce60d94e242725fd3bcc5673c04502c82a8e87b21ea411a63992dc39f8f/grpcio-1.78.0-cp310-cp310-win_amd64.whl (4.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 13.8 MB/s  0:00:00
Collecting bcrypt>=4.0.1 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9f/b9/9d9a641194a730bda138b3dfe53f584d61c58cd5230e37566e83ec2ffa0d/bcrypt-5.0.0-cp39-abi3-win_amd64.whl (150 kB)
Collecting typer>=0.9.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4a/91/48db081e7a63bb37284f9fbcefda7c44c277b18b0e13fbc36ea2335b71e6/typer-0.24.1-py3-none-any.whl (56 kB)
Collecting kubernetes>=28.1.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0c/70/05b685ea2dffcb2adbf3cdcea5d8865b7bc66f67249084cf845012a0ff13/kubernetes-35.0.0-py2.py3-none-any.whl (2.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 10.2 MB/s  0:00:00
Requirement already satisfied: tenacity>=8.2.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (9.1.4)
Requirement already satisfied: pyyaml>=6.0.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (6.0.3)
Collecting mmh3>=4.0.1 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c9/34/0b25889450f8aeffcec840aa73251e853f059c1b72ed1d1c027b956f95f5/mmh3-5.2.1-cp310-cp310-win_amd64.whl (41 kB)
Collecting orjson>=3.9.12 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/07/29/a576bf36d73d60df06904d3844a9df08e25d59eba64363aaf8ec2f9bff41/orjson-3.11.7-cp310-cp310-win_amd64.whl (125 kB)
Requirement already satisfied: httpx>=0.27.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (0.28.1)
Collecting rich>=10.11.0 (from chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/14/25/b208c5683343959b670dc001595f2f3737e051da617f66c31f7c4fa93abc/rich-14.3.3-py3-none-any.whl (310 kB)
Requirement already satisfied: jsonschema>=4.19.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (4.26.0)
Requirement already satisfied: llama-index-core<0.15,>=0.13.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-vector-stores-chroma) (0.14.18)
Requirement already satisfied: aiohttp<4,>=3.8.6 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.13.3)
Requirement already satisfied: aiosqlite in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (0.22.1)
Requirement already satisfied: banks<3,>=2.3.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.4.1)
Requirement already satisfied: dataclasses-json in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (0.6.7)
Requirement already satisfied: deprecated>=1.2.9.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.3.1)
Requirement already satisfied: dirtyjson<2,>=1.0.8 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.0.8)
Requirement already satisfied: filetype<2,>=1.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.2.0)
Requirement already satisfied: fsspec>=2023.5.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2026.2.0)
Requirement already satisfied: llama-index-workflows!=2.9.0,<3,>=2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.17.1)
Requirement already satisfied: nest-asyncio<2,>=1.5.8 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.6.0)
Requirement already satisfied: networkx>=3.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.4.2)
Requirement already satisfied: nltk>3.8.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.9.3)
Requirement already satisfied: pillow>=9.0.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (12.1.1)
Requirement already satisfied: platformdirs in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (4.9.4)
Requirement already satisfied: requests>=2.31.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.32.5)
Requirement already satisfied: setuptools>=80.9.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (80.10.2)
Requirement already satisfied: sqlalchemy>=1.4.49 in d:\anaconda\envs\local_agent_env\lib\site-packages (from sqlalchemy[asyncio]>=1.4.49->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.0.48)
Requirement already satisfied: tiktoken>=0.7.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (0.12.0)
Requirement already satisfied: tinytag>=2.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.2.1)
Requirement already satisfied: typing-inspect>=0.8.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (0.9.0)
Requirement already satisfied: wrapt in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.1.2)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.6.1)
Requirement already satisfied: aiosignal>=1.4.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.4.0)
Requirement already satisfied: async-timeout<6.0,>=4.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (5.0.1)
Requirement already satisfied: attrs>=17.3.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (26.1.0)
Requirement already satisfied: frozenlist>=1.1.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (0.4.1)
Requirement already satisfied: yarl<2.0,>=1.17.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.23.0)
Requirement already satisfied: griffe in d:\anaconda\envs\local_agent_env\lib\site-packages (from banks<3,>=2.3.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.0.0)
Requirement already satisfied: jinja2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from banks<3,>=2.3.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.1.6)
Requirement already satisfied: llama-index-instrumentation>=0.4.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from llama-index-workflows!=2.9.0,<3,>=2->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (0.5.0)
Requirement already satisfied: colorama in d:\anaconda\envs\local_agent_env\lib\site-packages (from tqdm>=4.65.0->chromadb) (0.4.6)
Requirement already satisfied: idna>=2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from yarl<2.0,>=1.17.0->aiohttp<4,>=3.8.6->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.11)
Requirement already satisfied: packaging>=24.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from build>=1.0.3->chromadb) (25.0)
Collecting pyproject_hooks (from build>=1.0.3->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bd/24/12818598c362d7f300f18e74db45963dbcb85150324092410c8b49405e42/pyproject_hooks-1.2.0-py3-none-any.whl (10 kB)
Requirement already satisfied: tomli>=1.1.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from build>=1.0.3->chromadb) (2.4.0)
Requirement already satisfied: anyio in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpx>=0.27.0->chromadb) (4.12.1)
Requirement already satisfied: certifi in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpx>=0.27.0->chromadb) (2026.2.25)
Requirement already satisfied: httpcore==1.* in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpx>=0.27.0->chromadb) (1.0.9)
Requirement already satisfied: h11>=0.16 in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpcore==1.*->httpx>=0.27.0->chromadb) (0.16.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (0.30.0)
Requirement already satisfied: six>=1.9.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (1.17.0)
Requirement already satisfied: python-dateutil>=2.5.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (2.9.0.post0)
Collecting websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0 (from kubernetes>=28.1.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/34/db/b10e48aa8fff7407e67470363eac595018441cf32d5e1001567a7aeba5d2/websocket_client-1.9.0-py3-none-any.whl (82 kB)
Collecting requests-oauthlib (from kubernetes>=28.1.0->chromadb)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/3b/5d/63d4ae3b9daea098d5d6f5da83984853c1bbacd5dc826764b249fe119d24/requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Requirement already satisfied: urllib3!=2.6.0,>=1.24.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (2.6.3)
Collecting durationpy>=0.7 (from kubernetes>=28.1.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b0/0d/9feae160378a3553fa9a339b0e9c1a048e147a4127210e286ef18b730f03/durationpy-0.10-py3-none-any.whl (3.9 kB)
Requirement already satisfied: click in d:\anaconda\envs\local_agent_env\lib\site-packages (from nltk>3.8.1->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (8.3.1)
Requirement already satisfied: joblib in d:\anaconda\envs\local_agent_env\lib\site-packages (from nltk>3.8.1->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.5.3)
Requirement already satisfied: regex>=2021.8.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from nltk>3.8.1->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2026.2.28)
Collecting coloredlogs (from onnxruntime>=1.14.1->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a7/06/3d6badcf13db419e25b07041d9c7b4a2c331d3f4e7134445ec5df57714cd/coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
Collecting flatbuffers (from onnxruntime>=1.14.1->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e8/2d/d2a548598be01649e2d46231d151a6c56d10b964d94043a335ae56ea2d92/flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
Requirement already satisfied: protobuf in d:\anaconda\envs\local_agent_env\lib\site-packages (from onnxruntime>=1.14.1->chromadb) (6.33.6)
Collecting sympy (from onnxruntime>=1.14.1->chromadb)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl (6.3 MB)
Collecting importlib-metadata<8.8.0,>=6.0 (from opentelemetry-api>=1.2.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/fa/5e/f8e9a1d23b9c20a551a8a02ea3637b4642e22c2626e3a13a9a29cdea99eb/importlib_metadata-8.7.1-py3-none-any.whl (27 kB)
Collecting zipp>=3.20 (from importlib-metadata<8.8.0,>=6.0->opentelemetry-api>=1.2.0->chromadb)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/2e/54/647ade08bf0db230bfea292f893923872fd20be6ac6f53b2b936ba839d75/zipp-3.23.0-py3-none-any.whl (10 kB)
Collecting googleapis-common-protos~=1.57 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/69/28/23eea8acd65972bbfe295ce3666b28ac510dfcb115fac089d3edb0feb00a/googleapis_common_protos-1.73.0-py3-none-any.whl (297 kB)
Collecting opentelemetry-exporter-otlp-proto-common==1.40.0 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8b/ca/8f122055c97a932311a3f640273f084e738008933503d0c2563cd5d591fc/opentelemetry_exporter_otlp_proto_common-1.40.0-py3-none-any.whl (18 kB)
Collecting opentelemetry-proto==1.40.0 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b9/b2/189b2577dde745b15625b3214302605b1353436219d42b7912e77fa8dc24/opentelemetry_proto-1.40.0-py3-none-any.whl (72 kB)
Collecting opentelemetry-semantic-conventions==0.61b0 (from opentelemetry-sdk>=1.2.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b2/37/cc6a55e448deaa9b27377d087da8615a3416d8ad523d5960b78dbeadd02a/opentelemetry_semantic_conventions-0.61b0-py3-none-any.whl (231 kB)
Requirement already satisfied: annotated-types>=0.6.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic>=2.0->chromadb) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.5 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic>=2.0->chromadb) (2.41.5)
Requirement already satisfied: typing-inspection>=0.4.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic>=2.0->chromadb) (0.4.2)
Collecting python-dotenv>=0.21.0 (from pydantic-settings>=2.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/0b/d7/1959b9648791274998a9c3526f6d0ec8fd2233e4d4acce81bbae76b44b2a/python_dotenv-1.2.2-py3-none-any.whl (22 kB)
Requirement already satisfied: charset_normalizer<4,>=2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from requests>=2.31.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.4.6)
Collecting markdown-it-py>=2.2.0 (from rich>=10.11.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/94/54/e7d793b573f298e1c9013b8c4dade17d481164aa517d1d7148619c2cedbf/markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from rich>=10.11.0->chromadb) (2.19.2)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.11.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Requirement already satisfied: greenlet>=1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from sqlalchemy>=1.4.49->sqlalchemy[asyncio]>=1.4.49->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.3.2)
Collecting huggingface-hub<2.0,>=0.16.4 (from tokenizers>=0.13.2->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/08/de/3ad061a05f74728927ded48c90b73521b9a9328c85d841bdefb30e01fb85/huggingface_hub-1.7.2-py3-none-any.whl (618 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 618.0/618.0 kB 2.5 MB/s  0:00:00
Collecting filelock>=3.10.0 (from huggingface-hub<2.0,>=0.16.4->tokenizers>=0.13.2->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a4/a5/842ae8f0c08b61d6484b52f99a03510a3a72d23141942d216ebe81fefbce/filelock-3.25.2-py3-none-any.whl (26 kB)
Collecting hf-xet<2.0.0,>=1.4.2 (from huggingface-hub<2.0,>=0.16.4->tokenizers>=0.13.2->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cd/71/193eabd7e7d4b903c4aa983a215509c6114915a5a237525ec562baddb868/hf_xet-1.4.2-cp37-abi3-win_amd64.whl (3.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.7/3.7 MB 7.8 MB/s  0:00:00
Collecting shellingham>=1.3.0 (from typer>=0.9.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Collecting annotated-doc>=0.0.2 (from typer>=0.9.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1e/d3/26bf1008eb3d2daa8ef4cacc7f3bfdc11818d111f7e2d0201bc6e3b49d45/annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
Requirement already satisfied: mypy-extensions>=0.3.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from typing-inspect>=0.8.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (1.1.0)
Collecting httptools>=0.6.3 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d5/72/ec8fc904a8fd30ba022dfa85f3bbc64c3c7cd75b669e24242c0658e22f3c/httptools-0.7.1-cp310-cp310-win_amd64.whl (86 kB)
Collecting watchfiles>=0.20 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/11/a0/a60c5a7c2ec59fa062d9a9c61d02e3b6abd94d32aac2d8344c4bdd033326/watchfiles-1.1.1-cp310-cp310-win_amd64.whl (287 kB)
Collecting websockets>=10.4 (from uvicorn[standard]>=0.18.3->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/b4/5f/7e40efe8df57db9b91c88a43690ac66f7b7aa73a11aa6a66b927e44f26fa/websockets-16.0-cp310-cp310-win_amd64.whl (178 kB)
Requirement already satisfied: exceptiongroup>=1.0.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from anyio->httpx>=0.27.0->chromadb) (1.3.1)
Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.14.1->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/f0/0f/310fb31e39e2d734ccaa2c0fb981ee41f7bd5056ce9bc29b2248bd569169/humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
Collecting pyreadline3 (from humanfriendly>=9.1->coloredlogs->onnxruntime>=1.14.1->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/5a/dc/491b7661614ab97483abf2056be1deee4dc2490ecbf7bff9ab5cdbac86e1/pyreadline3-3.5.4-py3-none-any.whl (83 kB)
Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from dataclasses-json->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.26.2)
Requirement already satisfied: griffecli==2.0.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from griffe->banks<3,>=2.3.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.0.0)
Requirement already satisfied: griffelib==2.0.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from griffe->banks<3,>=2.3.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (2.0.0)
Requirement already satisfied: MarkupSafe>=2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jinja2->banks<3,>=2.3.0->llama-index-core<0.15,>=0.13.0->llama-index-vector-stores-chroma) (3.0.3)
Collecting oauthlib>=3.0.0 (from requests-oauthlib->kubernetes>=28.1.0->chromadb)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/be/9c/92789c596b8df838baa98fa71844d84283302f7604ed565dafe5a6b5041a/oauthlib-3.3.1-py3-none-any.whl (160 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy->onnxruntime>=1.14.1->chromadb)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl (536 kB)
Installing collected packages: mpmath, flatbuffers, durationpy, zipp, websockets, websocket-client, sympy, shellingham, python-dotenv, pyreadline3, pyproject_hooks, pypika, pybase64, overrides, orjson, opentelemetry-proto, oauthlib, mmh3, mdurl, importlib-resources, httptools, hf-xet, grpcio, googleapis-common-protos, filelock, bcrypt, annotated-doc, uvicorn, requests-oauthlib, opentelemetry-exporter-otlp-proto-common, markdown-it-py, importlib-metadata, humanfriendly, build, watchfiles, rich, pydantic-settings, opentelemetry-api, kubernetes, coloredlogs, typer, opentelemetry-semantic-conventions, onnxruntime, opentelemetry-sdk, huggingface-hub, tokenizers, opentelemetry-exporter-otlp-proto-grpc, chromadb, llama-index-vector-stores-chroma
Successfully installed annotated-doc-0.0.4 bcrypt-5.0.0 build-1.4.0 chromadb-1.5.5 coloredlogs-15.0.1 durationpy-0.10 filelock-3.25.2 flatbuffers-25.12.19 googleapis-common-protos-1.73.0 grpcio-1.78.0 hf-xet-1.4.2 httptools-0.7.1 huggingface-hub-1.7.2 humanfriendly-10.0 importlib-metadata-8.7.1 importlib-resources-6.5.2 kubernetes-35.0.0 llama-index-vector-stores-chroma-0.5.5 markdown-it-py-4.0.0 mdurl-0.1.2 mmh3-5.2.1 mpmath-1.3.0 oauthlib-3.3.1 onnxruntime-1.23.2 opentelemetry-api-1.40.0 opentelemetry-exporter-otlp-proto-common-1.40.0 opentelemetry-exporter-otlp-proto-grpc-1.40.0 opentelemetry-proto-1.40.0 opentelemetry-sdk-1.40.0 opentelemetry-semantic-conventions-0.61b0 orjson-3.11.7 overrides-7.7.0 pybase64-1.4.3 pydantic-settings-2.13.1 pypika-0.51.1 pyproject_hooks-1.2.0 pyreadline3-3.5.4 python-dotenv-1.2.2 requests-oauthlib-2.0.0 rich-14.3.3 shellingham-1.5.4 sympy-1.14.0 tokenizers-0.22.2 typer-0.24.1 uvicorn-0.42.0 watchfiles-1.1.1 websocket-client-1.9.0 websockets-16.0 zipp-3.23.0
(local_agent_env) PS D:\MyHyperspectralAgent>
(local_agent_env) PS D:\MyHyperspectralAgent> pip install chromadb arxiv scipy
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Requirement already satisfied: chromadb in d:\anaconda\envs\local_agent_env\lib\site-packages (1.5.5)
Collecting arxiv
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3d/6a/297380dc42fa25dff095feda66d46f7abba77ba54579d079071a2459e8d3/arxiv-2.4.1-py3-none-any.whl (12 kB)
Collecting scipy
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d1/84/55bc4881973d3f79b479a5a2e2df61c8c9a04fcb986a213ac9c02cfb659b/scipy-1.15.3-cp310-cp310-win_amd64.whl (41.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.3/41.3 MB 13.3 MB/s  0:00:03
Requirement already satisfied: build>=1.0.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.4.0)
Requirement already satisfied: pydantic>=2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (2.12.5)
Requirement already satisfied: pydantic-settings>=2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (2.13.1)
Requirement already satisfied: pybase64>=1.4.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.4.3)
Requirement already satisfied: uvicorn>=0.18.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from uvicorn[standard]>=0.18.3->chromadb) (0.42.0)
Requirement already satisfied: numpy>=1.22.5 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (2.2.6)
Requirement already satisfied: typing-extensions>=4.5.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (4.15.0)
Requirement already satisfied: onnxruntime>=1.14.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.23.2)
Requirement already satisfied: opentelemetry-api>=1.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.40.0)
Requirement already satisfied: opentelemetry-exporter-otlp-proto-grpc>=1.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.40.0)
Requirement already satisfied: opentelemetry-sdk>=1.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.40.0)
Requirement already satisfied: tokenizers>=0.13.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (0.22.2)
Requirement already satisfied: pypika>=0.48.9 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (0.51.1)
Requirement already satisfied: tqdm>=4.65.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (4.67.3)
Requirement already satisfied: overrides>=7.3.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (7.7.0)
Requirement already satisfied: importlib-resources in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (6.5.2)
Requirement already satisfied: grpcio>=1.58.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (1.78.0)
Requirement already satisfied: bcrypt>=4.0.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (5.0.0)
Requirement already satisfied: typer>=0.9.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (0.24.1)
Requirement already satisfied: kubernetes>=28.1.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (35.0.0)
Requirement already satisfied: tenacity>=8.2.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (9.1.4)
Requirement already satisfied: pyyaml>=6.0.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (6.0.3)
Requirement already satisfied: mmh3>=4.0.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (5.2.1)
Requirement already satisfied: orjson>=3.9.12 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (3.11.7)
Requirement already satisfied: httpx>=0.27.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (0.28.1)
Requirement already satisfied: rich>=10.11.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (14.3.3)
Requirement already satisfied: jsonschema>=4.19.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from chromadb) (4.26.0)
Collecting feedparser~=6.0.10 (from arxiv)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/4e/eb/c96d64137e29ae17d83ad2552470bafe3a7a915e85434d9942077d7fd011/feedparser-6.0.12-py3-none-any.whl (81 kB)
Requirement already satisfied: requests~=2.32.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from arxiv) (2.32.5)
Collecting sgmllib3k (from feedparser~=6.0.10->arxiv)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9e/bd/3704a8c3e0942d711c1299ebf7b9091930adae6675d7c8f476a7ce48653c/sgmllib3k-1.0.0.tar.gz (5.8 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: charset_normalizer<4,>=2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from requests~=2.32.0->arxiv) (3.4.6)
Requirement already satisfied: idna<4,>=2.5 in d:\anaconda\envs\local_agent_env\lib\site-packages (from requests~=2.32.0->arxiv) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from requests~=2.32.0->arxiv) (2.6.3)
Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda\envs\local_agent_env\lib\site-packages (from requests~=2.32.0->arxiv) (2026.2.25)
Requirement already satisfied: packaging>=24.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from build>=1.0.3->chromadb) (25.0)
Requirement already satisfied: pyproject_hooks in d:\anaconda\envs\local_agent_env\lib\site-packages (from build>=1.0.3->chromadb) (1.2.0)
Requirement already satisfied: colorama in d:\anaconda\envs\local_agent_env\lib\site-packages (from build>=1.0.3->chromadb) (0.4.6)
Requirement already satisfied: tomli>=1.1.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from build>=1.0.3->chromadb) (2.4.0)
Requirement already satisfied: anyio in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpx>=0.27.0->chromadb) (4.12.1)
Requirement already satisfied: httpcore==1.* in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpx>=0.27.0->chromadb) (1.0.9)
Requirement already satisfied: h11>=0.16 in d:\anaconda\envs\local_agent_env\lib\site-packages (from httpcore==1.*->httpx>=0.27.0->chromadb) (0.16.0)
Requirement already satisfied: attrs>=22.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (26.1.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (0.37.0)
Requirement already satisfied: rpds-py>=0.25.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from jsonschema>=4.19.0->chromadb) (0.30.0)
Requirement already satisfied: six>=1.9.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (1.17.0)
Requirement already satisfied: python-dateutil>=2.5.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (2.9.0.post0)
Requirement already satisfied: websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (1.9.0)
Requirement already satisfied: requests-oauthlib in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (2.0.0)
Requirement already satisfied: durationpy>=0.7 in d:\anaconda\envs\local_agent_env\lib\site-packages (from kubernetes>=28.1.0->chromadb) (0.10)
Requirement already satisfied: coloredlogs in d:\anaconda\envs\local_agent_env\lib\site-packages (from onnxruntime>=1.14.1->chromadb) (15.0.1)
Requirement already satisfied: flatbuffers in d:\anaconda\envs\local_agent_env\lib\site-packages (from onnxruntime>=1.14.1->chromadb) (25.12.19)
Requirement already satisfied: protobuf in d:\anaconda\envs\local_agent_env\lib\site-packages (from onnxruntime>=1.14.1->chromadb) (6.33.6)
Requirement already satisfied: sympy in d:\anaconda\envs\local_agent_env\lib\site-packages (from onnxruntime>=1.14.1->chromadb) (1.14.0)
Requirement already satisfied: importlib-metadata<8.8.0,>=6.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from opentelemetry-api>=1.2.0->chromadb) (8.7.1)
Requirement already satisfied: zipp>=3.20 in d:\anaconda\envs\local_agent_env\lib\site-packages (from importlib-metadata<8.8.0,>=6.0->opentelemetry-api>=1.2.0->chromadb) (3.23.0)
Requirement already satisfied: googleapis-common-protos~=1.57 in d:\anaconda\envs\local_agent_env\lib\site-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb) (1.73.0)
Requirement already satisfied: opentelemetry-exporter-otlp-proto-common==1.40.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb) (1.40.0)
Requirement already satisfied: opentelemetry-proto==1.40.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb) (1.40.0)
Requirement already satisfied: opentelemetry-semantic-conventions==0.61b0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from opentelemetry-sdk>=1.2.0->chromadb) (0.61b0)
Requirement already satisfied: annotated-types>=0.6.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic>=2.0->chromadb) (0.7.0)
Requirement already satisfied: pydantic-core==2.41.5 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic>=2.0->chromadb) (2.41.5)
Requirement already satisfied: typing-inspection>=0.4.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic>=2.0->chromadb) (0.4.2)
Requirement already satisfied: python-dotenv>=0.21.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from pydantic-settings>=2.0->chromadb) (1.2.2)
Requirement already satisfied: markdown-it-py>=2.2.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from rich>=10.11.0->chromadb) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from rich>=10.11.0->chromadb) (2.19.2)
Requirement already satisfied: mdurl~=0.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->chromadb) (0.1.2)
Requirement already satisfied: huggingface-hub<2.0,>=0.16.4 in d:\anaconda\envs\local_agent_env\lib\site-packages (from tokenizers>=0.13.2->chromadb) (1.7.2)
Requirement already satisfied: filelock>=3.10.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from huggingface-hub<2.0,>=0.16.4->tokenizers>=0.13.2->chromadb) (3.25.2)
Requirement already satisfied: fsspec>=2023.5.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from huggingface-hub<2.0,>=0.16.4->tokenizers>=0.13.2->chromadb) (2026.2.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.4.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from huggingface-hub<2.0,>=0.16.4->tokenizers>=0.13.2->chromadb) (1.4.2)
Requirement already satisfied: click>=8.2.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from typer>=0.9.0->chromadb) (8.3.1)
Requirement already satisfied: shellingham>=1.3.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from typer>=0.9.0->chromadb) (1.5.4)
Requirement already satisfied: annotated-doc>=0.0.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from typer>=0.9.0->chromadb) (0.0.4)
Requirement already satisfied: httptools>=0.6.3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from uvicorn[standard]>=0.18.3->chromadb) (0.7.1)
Requirement already satisfied: watchfiles>=0.20 in d:\anaconda\envs\local_agent_env\lib\site-packages (from uvicorn[standard]>=0.18.3->chromadb) (1.1.1)
Requirement already satisfied: websockets>=10.4 in d:\anaconda\envs\local_agent_env\lib\site-packages (from uvicorn[standard]>=0.18.3->chromadb) (16.0)
Requirement already satisfied: exceptiongroup>=1.0.2 in d:\anaconda\envs\local_agent_env\lib\site-packages (from anyio->httpx>=0.27.0->chromadb) (1.3.1)
Requirement already satisfied: humanfriendly>=9.1 in d:\anaconda\envs\local_agent_env\lib\site-packages (from coloredlogs->onnxruntime>=1.14.1->chromadb) (10.0)
Requirement already satisfied: pyreadline3 in d:\anaconda\envs\local_agent_env\lib\site-packages (from humanfriendly>=9.1->coloredlogs->onnxruntime>=1.14.1->chromadb) (3.5.4)
Requirement already satisfied: oauthlib>=3.0.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from requests-oauthlib->kubernetes>=28.1.0->chromadb) (3.3.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in d:\anaconda\envs\local_agent_env\lib\site-packages (from sympy->onnxruntime>=1.14.1->chromadb) (1.3.0)
Building wheels for collected packages: sgmllib3k
  Building wheel for sgmllib3k (pyproject.toml) ... done
  Created wheel for sgmllib3k: filename=sgmllib3k-1.0.0-py3-none-any.whl size=6104 sha256=2c0976e7a3d50419081fbb8c8cd193f4916d952c42af58727fcac4b3834d5293
  Stored in directory: c:\users\lenovo\appdata\local\pip\cache\wheels\50\20\4b\e95fc891917d652cb6ecbfea035cf3ce640259cf857aaa21a7
Successfully built sgmllib3k
Installing collected packages: sgmllib3k, scipy, feedparser, arxiv
Successfully installed arxiv-2.4.1 feedparser-6.0.12 scipy-1.15.3 sgmllib3k-1.0.0
(local_agent_env) PS D:\MyHyperspectralAgent> streamlit run app.py
```
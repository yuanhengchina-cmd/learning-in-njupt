小背景知识：
autogpt可以根据任务自动生成提示词自主运行
deep research自主web搜索和深度推理研究
minus多智能体协作
openclaw自动接管设备，自主学习技能完成任务
自动化工作流等目标
langchain提供了agent开发、调试、评估、部署的完整工具，目的是简化agent开发

langsmith agent builder0代码开发agent的脚手架
deep agents开发复杂的多智能体合作的高级智能体
langchain传统的基本智能体组件开发
langgraph关注智能体本身工作流的编排
v1.2后链式调用转为高级agent开发，chain其实已经不怎么用了，目的是提高agent开发效率

ai通识
**uv工具**，多线程包管理、依赖管理、工具安装、打包发布
防止出现冲突，准备独立的环境，为每个项目
uv创建新项目(uv init -p 3.13)init是文件名
```
uv init hello-world
cd hello-world
```
将创建
```
-python-version
-README.md
-main.py
-pyproject.toml
```
main中含有一个简单的hello world程序，使用
```
$ uv run main.py
hello from hello-world 
```
![[Pasted image 20260410201321.png|495]]
![[Pasted image 20260410201354.png]]![[Pasted image 20260410201526.png]]
在这个文件里可以安装依赖
![[Pasted image 20260410201733.png]]安装后会出现
![[Pasted image 20260410201950.png]]
是有独立的环境的![[Pasted image 20260410202018.png]]
可以直接打包好项目给别人，对方只要在自己本地运行一个uv sync自动配好所有的环境
![[Pasted image 20260410203642.png]]
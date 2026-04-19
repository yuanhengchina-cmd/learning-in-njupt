API Key正常是明文显示，容易泄露，别人如果知晓会访问我账号的模型，消耗token
环境变量法：隐藏明文API Key,记录在我们的操作系统的环境变量里，运行代码时会自动的从环境变量中去读取APIKey去使用
配置2个API Key
-OPENAI_API_KEY（用的openai库的环境变量名）
-DASHCOPE_API_KEY（langchain库使用的环境变量名）
利用环境变量名记录API_KEY的值，写代码不用去复制APIKEY
![[Pasted image 20260410174558.png]]![[Pasted image 20260410174710.png]]
重启Pycharm生效（查看方式:把api_key参数给注释掉）
![[Pasted image 20260410175138.png]]

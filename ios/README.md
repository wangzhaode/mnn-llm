# mnn-llm ios demo

🚀 本示例代码全部由`ChatGPT-4`生成。

## 速度

模型: Qwen-1.8b-int4
- iPhone 11    : pefill  52.00 tok/s, decode 16.23 tok/s
- iPhone 14 Pro: pefill 102.63 tok/s, decode 33.53 tok/s

模型: Qwen-1.8b-int8
- iPhone 11    : pefill  61.90 tok/s, decode 14.75 tok/s
- iPhone 14 Pro: pefill 105.41 tok/s, decode 25.45 tok/s

## 编译
1. 首先下载模型文件: [Qwen1.5-0.5B-Chat-MNN](https://modelscope.cn/models/zhaode/Qwen1.5-0.5B-Chat-MNN/files)
2. 将模型文件拷贝到`ios/mnn-llm/model/qwen1.5-0.5b-chat`目录下
3. 在xcode项目属性中`Signing & Capabilities` > `Team`输入自己的账号；`Bundle Identifier`可以重新命名；
4. 连接iPhone并编译执行，需要在手机端打开开发者模式，并在安装完成后在：`设置` > `通用` > `VPN与设备管理`中选择信任该账号；

备注：如测试其他模型，可以将`ios/mnn-llm/model/qwen1.5-0.5b-chat`替换为其他模型的文件夹；同时修改`LLMInferenceEngineWrapper.m +38`的模型路径；

## 测试
等待模型加载完成后即可发送信息，如下图所示：
![ios-app](./ios_app.jpg)
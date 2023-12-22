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
1. 首先下载MNN预编译的ios包: [mnn_2.8.0_ios_llm.zip](https://github.com/alibaba/MNN/releases/download/2.8.0/mnn_2.8.0_ios_llm.zip)
2. 解压该文件，得到`MNN.framework`目录；
3. 在xcode项目属性中`Build Phases` > `Link Binary With Libraries` > `+` > `Add Other` > `Add Files`选择上述解压的文件夹；
4. 在xcode中右键项目`mnn-llm` > `Add Files to` > 选择模型文件`qwen-1.8b-int4/8`；
5. 在xcode项目属性中`Signing & Capabilities` > `Team`输入自己的账号；`Bundle Identifier`可以重新命名；
6. 连接iPhone并编译执行，需要在手机端打开开发者模式，并在安装完成后在：`设置` > `通用` > `VPN与设备管理`中选择信任该账号；

## 测试
等待模型加载完成后即可发送信息，如下图所示：
![ios-app](./ios_app.jpg)
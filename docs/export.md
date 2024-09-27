# 模型导出

`mnn-llm`使用MNN模型导出工具[llmexport](https://github.com/wangzhaode/llm-export)将HuggingFace或ModelScope上的模型导出为MNN模型。

## 安装llmexport
```sh
# pip install
pip install llmexport

# git install
pip install git+https://github.com/wangzhaode/llm-export@master

# local install
git clone https://github.com/wangzhaode/llm-export && cd llm-export/
pip install .
```

## 用法

1. 下载模型
```sh
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
# 如果huggingface下载慢可以使用modelscope
git clone https://modelscope.cn/qwen/Qwen2-1.5B-Instruct.git
```
2. 测试模型
```sh
# 测试文本输入
llmexport --path Qwen2-1.5B-Instruct --test "你好"
# 测试图像文本
llmexport --path Qwen2-VL-2B-Instruct  --test "<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容"
```

3. 导出模型
```sh
# 将Qwen2-1.5B-Instruct导出为onnx模型
llmexport --path Qwen2-1.5B-Instruct --export onnx
# 将Qwen2-1.5B-Instruct导出为mnn模型, 量化参数为4bit, blokc-wise = 128
llmexport --path Qwen2-1.5B-Instruct --export mnn --quant_bit 4 --quant_block 128
```
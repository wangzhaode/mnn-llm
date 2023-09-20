# mnn-llm
[![License](https://img.shields.io/github/license/wangzhaode/mnn-llm)](LICENSE.txt)
[![Download](https://img.shields.io/github/downloads/wangzhaode/mnn-llm/total)](https://github.com/wangzhaode/mnn-llm/releases)

[Read me in english ](./README_en.md)

## 模型支持

llm模型导出onnx模型请使用[llm-export](https://github.com/wangzhaode/llm-export)

当前支持以模型：

| model | onnx-fp32 | mnn-int4 |
|-------|-----------|----------|
| chatglm-6b | [![Download][download-chatglm-6b-onnx]][release-chatglm-6b-onnx] | [![Download][download-chatglm-6b-mnn]][release-chatglm-6b-mnn] |
| chatglm2-6b | [![Download][download-chatglm2-6b-onnx]][release-chatglm2-6b-onnx] | [![Download][download-chatglm2-6b-mnn]][release-chatglm2-6b-mnn] |
| codegeex2-6b | [![Download][download-codegeex2-6b-onnx]][release-codegeex2-6b-onnx] | [![Download][download-codegeex2-6b-mnn]][release-codegeex2-6b-mnn] |
| Qwen-7B-Chat | [![Download][download-qwen-7b-chat-onnx]][release-qwen-7b-chat-onnx] | [![Download][download-qwen-7b-chat-mnn]][release-qwen-7b-chat-mnn] |
| Baichuan2-7B-Chat | [![Download][download-baichuan2-7b-chat-onnx]][release-baichuan2-7b-chat-onnx] | [![Download][download-baichuan2-7b-chat-mnn]][release-baichuan2-7b-chat-mnn] |
| Llama-2-7b-chat | [![Download][download-llama2-7b-chat-onnx]][release-llama2-7b-chat-onnx] | [![Download][download-llama2-7b-chat-mnn]][release-llama2-7b-chat-mnn] |

[download-chatglm-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/chatglm-6b-onnx/total
[download-chatglm2-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/chatglm2-6b-onnx/total
[download-codegeex2-6b-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/codegeex2-6b-onnx/total
[download-qwen-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/qwen-7b-chat-onnx/total
[download-baichuan2-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/baichuan2-7b-chat-onnx/total
[download-llama2-7b-chat-onnx]: https://img.shields.io/github/downloads/wangzhaode/llm-export/llama2-7b-chat-onnx/total
[release-chatglm-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/chatglm-6b-onnx
[release-chatglm2-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/chatglm2-6b-onnx
[release-codegeex2-6b-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/codegeex2-6b-onnx
[release-qwen-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/qwen-7b-chat-onnx
[release-baichuan2-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/baichuan2-7b-chat-onnx
[release-llama2-7b-chat-onnx]: https://github.com/wangzhaode/llm-export/releases/tag/llama2-7b-chat-onnx
[download-chatglm-6b-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-llm/chatglm-6b-mnn/total
[download-chatglm2-6b-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-llm/chatglm2-6b-mnn/total
[download-codegeex2-6b-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-llm/codegeex2-6b-mnn/total
[download-qwen-7b-chat-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-llm/qwen-7b-chat-mnn/total
[download-baichuan2-7b-chat-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-llm/baichuan2-7b-chat-mnn/total
[download-llama2-7b-chat-mnn]: https://img.shields.io/github/downloads/wangzhaode/mnn-llm/llama2-7b-chat-mnn/total
[release-chatglm-6b-mnn]: https://github.com/wangzhaode/mnn-llm/releases/tag/chatglm-6b-mnn
[release-chatglm2-6b-mnn]: https://github.com/wangzhaode/mnn-llm/releases/tag/chatglm2-6b-mnn
[release-codegeex2-6b-mnn]: https://github.com/wangzhaode/mnn-llm/releases/tag/codegeex2-6b-mnn
[release-qwen-7b-chat-mnn]: https://github.com/wangzhaode/mnn-llm/releases/tag/qwen-7b-chat-mnn
[release-baichuan2-7b-chat-mnn]: https://github.com/wangzhaode/mnn-llm/releases/tag/baichuan2-7b-chat-mnn
[release-llama2-7b-chat-mnn]: https://github.com/wangzhaode/mnn-llm/releases/tag/llama2-7b-chat-mnn


### 下载int4模型
```
# <model> like `chatglm-6b`
# linux/macos
./script/download_model.sh <model>

# windows
./script/download_model.ps1 <model>
```

## 构建

当前构建状态：

| System | CPU(x64) |
|--------|----------|
| Linux | [![Build Status][pass-linux-cpu]][ci-linux-cpu] |
| Macos | [![Build Status][pass-macos-cpu]][ci-macos-cpu] |
| Windows | [![Build Status][pass-windows-cpu]][ci-windows-cpu] |
| Android | [![Build Status][pass-android-cpu]][ci-android-cpu] |

[pass-linux-cpu]: https://img.shields.io/github/actions/workflow/status/wangzhaode/mnn-llm/linux-cpu.yml?branch=master
[pass-macos-cpu]: https://img.shields.io/github/actions/workflow/status/wangzhaode/mnn-llm/macos-cpu.yml?branch=master
[pass-windows-cpu]: https://img.shields.io/github/actions/workflow/status/wangzhaode/mnn-llm/windows-cpu.yml?branch=master
[pass-android-cpu]: https://img.shields.io/github/actions/workflow/status/wangzhaode/mnn-llm/android-cpu.yml?branch=master
[ci-linux-cpu]: https://github.com/wangzhaode/mnn-llm/actions?query=workflow%3Alinux-cpu
[ci-macos-cpu]: https://github.com/wangzhaode/mnn-llm/actions?query=workflow%3Amacos-cpu
[ci-windows-cpu]: https://github.com/wangzhaode/mnn-llm/actions?query=workflow%3Awindows-cpu
[ci-android-cpu]: https://github.com/wangzhaode/mnn-llm/actions?query=workflow%3Aandroid-cpu

### CPU-Only
```
# linux
./script/linux_build.sh

# macos
./script/macos_build.sh

# windows msvc
./script/windows_build.ps1

# android
./script/android_build.sh
```
### CUDA/OPENCL
`TODO`

### 4. 执行

```bash
# linux/macos
./cli_demo # cli demo
./web_demo # web ui demo

# windows
.\Debug\cli_demo.exe
.\Debug\web_demo.exe
```


## Reference
- [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
- [codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b)
- [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Qwen-7B-Chat](https://huggingface.co/tangger/Qwen-7B-Chat)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- [chatgpt-web](https://github.com/xqdoo00o/chatgpt-web)
- [cppjieba](https://github.com/yanyiwu/cppjieba)
- [ChatViewDemo](https://github.com/BrettFX/ChatViewDemo)

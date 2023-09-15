# mnn-llm

[Read me in english ](./README_en.md)

## 模型支持
当前支持以模型：

| model | onnx-fp32 | mnn-int4 |
|-------|-----------|----------|
| chatglm-6b | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/chatglm-6b-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/chatglm-6b-mnn) |
| chatglm2-6b | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/chatglm2-6b-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/chatglm2-6b-mnn) |
| codegeex2-6b | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/codegeex2-6b-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/codegeex2-6b-mnn) |
| Qwen-7B-Chat | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/qwen-7b-chat-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/qwen-7b-chat-mnn) |
| Baichuan2-7B-Chat | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/baichuan2-7b-chat-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/baichuan2-7b-chat-mnn) |

### 下载
```
# int4 model, <model> like `chatglm-6b`
./script/download_model.sh <model>
```


## 构建

当前构建状态：

| system | cpu |
|--------|-----|
| linux | [![Build Status][pass-linux-cpu]][ci-linux-cpu] |
| macos | [![Build Status][pass-macos-cpu]][ci-macos-cpu] |
| windows | [![Build Status][pass-windows-cpu]][ci-windows-cpu] |
| android | [![Build Status][pass-android-cpu]][ci-android-cpu] |

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
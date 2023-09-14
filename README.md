# mnn-llm

[Read me in english ](./README_en.md)

## 模型支持
该项目支持将主流llm模型转换到mnn模型部署推理，目前支持以下模型：

| 模型 | onnx-fp32 | mnn-int4 |
|------|-----------|----------|
| chatglm-6b | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/chatglm-6b-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/chatglm-6b-mnn) |
| chatglm2-6b | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/chatglm2-6b-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/chatglm2-6b-mnn) |
| codegeex2-6b | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/codegeex2-6b-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/untagged-93eea51bfbbd01f29a5f) |
| Qwen-7B-Chat | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/qwen-7b-chat-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/untagged-d109db4ac537bfce7a0b) |
| Baichuan2-7B-Chat | [onnx](https://github.com/wangzhaode/llm-export/releases/tag/baichuan2-7b-chat-onnx) | [mnn](https://github.com/wangzhaode/mnn-llm/releases/tag/untagged-6798382d6309a35e20d0) |


## 构建

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
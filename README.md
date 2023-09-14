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


## 用法
### 0. 模型导出与转换
可以使用[LLMExporter](https://github.com/wangzhaode/LLMExporter)将模型导出为`onnx`格式，然后使用`mnnconvert`转换为`mnn`模型。

### 1. 下载本项目
```bash
git clone https://github.com/wangzhaode/mnn-llm.git
```
### 2. 编译MNN库
- 克隆MNN项目，最新正式版是2.5.0
```bash
git clone https://github.com/alibaba/MNN.git -b 2.5.0
```

- 进入MNN项目, 并构建一个Build目录准备编译
```bash
cd MNN
mkdir build && cd build
```

- 正式编译，可选CPU/CUDA/OpenCL三种，推荐有英伟达显卡的选择CUDA，没显卡的选CPU,有AMD显卡的选择OpenCL

```bash
# CPU only（Suport Linux/Mac/Windows）
cmake -DCMAKE_BUILD_TYPE=Release ..

# using CUDA(Support Linux)
cmake -DCMAKE_BUILD_TYPE=Release -DMNN_CUDA=ON ..

# using OPENCL
cmake -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=ON -DMNN_SEP_BUILD=OFF ..

# start build(support Linux/Mac)
make -j$(nproc)

# start build(support Windows)
cmake --build . -- /m:8

```

- 回到mnn-llm

```bash
cd ../..
```

- 将MNN库的编译结果拷贝到`mnn-llm/libs`目录下
```bash
# for Linux/Mac
cp -r MNN/include/MNN include
cp MNN/build/libMNN.so libs/
cp MNN/build/express/*.so  libs/

# for windows
cp -r MNN/include/MNN include
cp MNN/build/Debug/MNN.dll libs/
cp MNN/build/Debug/MNN.lib libs/
```

- 对于Windows，还需要下载一下第三方库pthread，下载[地址](https://gigenet.dl.sourceforge.net/project/pthreads4w/pthreads-w32-2-9-1-release.zip),下载后解压，打开Pre-built.2\lib\x64， 将pthreadVC2.lib文件拷贝到ChatGLM-MNN的libs文件夹。打开Pre-built.2\include,将下面三个.h文件都放到ChatGLM-MNN的include文件夹。对于windows，项目的最终文件结构如下：
```bash
├───libs
│   ├───MNN.dll
│   ├───MNN.lib
│   └───pthreadVC2.lib
├───include
│   ├───cppjieba
│   ├───limonp
│   ├───MNN
│   ├───chat.hpp
│   ├───httplib.h
│   ├───pthread.h
│   ├───sched.h
│   └───semaphore.h
```


### 3. Download Models
从 `github release` 下载模型文件到 `/path/to/ChatGLM-MNN/resource/models`， 如下：
- 对于Linux/Mac
```bash
cd resource/models
# 下载fp16权值模型, 几乎没有精度损失
./download_models.sh fp16
# 对于中国用户，可以使用第三方服务加速下载fp16模型
./download_models.sh fp16 proxy

# 下载int8权值模型，极少精度损失，推荐使用
./download_models.sh int8
# 对于中国用户，可以使用第三方服务加速下载int8模型
./download_models.sh int8 proxy

# 下载int4权值模型，有一定精度损失
./download_models.sh int4 
# 对于中国用户，可以使用第三方服务加速下载int4模型
./download_models.sh int4 proxy
```
- 对于windows,将上面的`xxx.sh`替换为`xxx.ps1`文件即可，例如：
```powershell
cd resource/models

# 下载fp16权值模型, 几乎没有精度损失
./download_models.ps1 fp16
# 对于中国用户，可以使用第三方服务加速下载fp16模型
./download_models.ps1 fp16 proxy
```

### 4. Build and Run

##### Mac/Linux/Windows:
```bash
mkdir build && cd build
# for CPU
cmake ..
# for GPU
cmake -D WITH_CUDA=on ..
# for mini memory device
cmake -D BUILD_MINI_MEM_MODE=on ..

# start build(support Linux/Mac)
make -j$(nproc)

# start build(support Windows)
cmake --build . -- /m:8

# run (for Linux/Mac)
./cli_demo # cli demo
./web_demo # web ui demo

# run (for Windows)
.\Debug\cli_demo.exe
.\Debug\web_demo.exe
```


##### Android:
```
mkdir build
cd build
../android_build.sh
make -j8
```

## Reference
- [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- [chatgpt-web](https://github.com/xqdoo00o/chatgpt-web)
- [cppjieba](https://github.com/yanyiwu/cppjieba)
- [ChatViewDemo](https://github.com/BrettFX/ChatViewDemo)
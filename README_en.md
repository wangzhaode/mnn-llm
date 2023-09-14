# mnn-llm
## Support
- chatglm-6b
- chatglm2-6b
- codegeex2-6b
- Qwen-7B-Chat
- Baichuan2-7B-Chat

## Usage
### 0. Model export and convert
Using [LLMExporter](https://github.com/wangzhaode/LLMExporter) export llm model to `onnx` format，and then using `mnnconvert` convert to `mnn` model.


### 1. Download this project
```bash
git clone https://github.com/wangzhaode/mnn-llm.git
```

### 2. Compile MNN library
Compile MNN from source code, the latest release version is 2.5.0

```bash
git clone https://github.com/alibaba/MNN.git -b 2.5.0
```

- Enter the MNN project, and build a Build directory ready to compile
```bash
cd MNN
mkdir build && cd build
```

- Formally compiled, CPU/CUDA/OpenCL can be selected. It is recommended to choose CUDA if you have an NVIDIA graphics card, choose CPU if you don’t have a graphics card, and choose OpenCL if you have an AMD graphics card.
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

- Back to ChatGLM-MNN
```bash
cd ../..
```

- Copy the compilation result of the MNN library to `mnn-llm/libs`
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
- For Windows, you also need to download the third-party library pthread, download [address] (https://gigenet.dl.sourceforge.net/project/pthreads4w/pthreads-w32-2-9-1-release.zip), unzip after downloading, and open Pre-built.2libx64, Copy the pthreadVC2.lib file to the libs folder of ChatGLM-MNN. Open Pre-built.2include and place the following three .h files in the include folder of ChatGLM-MNN. For Windows, the final file structure of the project is as follows:
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
Download model files from github release to /path/to/ChatGLM-MNN/resource/models, as follows:   
- 对于Linux/Mac
```bash
cd resource/models
# download fp16(almost no loss of precision)
./download_models.sh fp16 
# For Chinese users, you can use third-party services to speed up downloading the fp16 model
./download_models.sh fp16 proxy

# download int8(little loss of precision,recommend)
./download_models.sh int8
# For Chinese users, you can use third-party services to speed up downloading the int8 model
./download_models.sh int8 proxy

# download int4(some precision loss)
./download_models.sh int4
# For Chinese users, you can use third-party services to speed up downloading the int4 model
./download_models.sh int4 proxy
```

- For Windows, replace 'xxx.sh' above with the 'xxx.ps1' file, for example:
```powershell
cd resource/models

# download fp16(almost no loss of precision)
./download_models.ps1 fp16 
# For Chinese users, you can use third-party services to speed up downloading the fp16 model
./download_models.ps1 fp16 proxy
```

### 4. Build and Run

##### Mac/Linux/Windows:
```bash
mkdir build
cd build
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
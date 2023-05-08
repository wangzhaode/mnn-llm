# ChatGLM-MNN
## Describe

This project converts the model ChatGLM-6B to MNN and performs inference using C++. The following optimizations were made in the implementation:

The vocabulary table was partially deleted, removing the first 20,000 image words not used in the model, reducing the size of vocab, embedding, and lm_head from 150,528 to 130,528.
The number of Embedding operations called is reduced by loading data using fseek and fread to reduce memory usage.
The lm_head operation is [num, 4096] @ [4096, 130528] and is converted to [130528, 4096] @ [4096, 1].
The original model requires a large amount of GPU memory. The model is split into 28 submodels, which can be dynamically allocated to GPU and CPU based on the user's GPU memory size to fully utilize the memory and computing power of both. Even small VRAM graphics cards can speed up generation.

For edge devices, calculations can be loaded progressively, and inference can be performed even on Android devices with only 2GB of memory (although slowly).

An ONNX export method and model are now available for deployment using other frameworks; click here to download
Currently, two types of demos are supported: command-line dialogue and web UI dialogue.

## Speed

Mobile devices: Segment models can be loaded one by one for inference on devices with memory larger than 2G, but experimental tests have shown poor performance, currently around 63 s/word.

PC test platform:

Memory: 32G (+32G Swap)
CPU: AMD Ryzen 9 3900X 12-Core Processor
GPU: GeForce RTX 2080 Ti

### FP Model

Only floating-point models were tested (CPU: fp32/ GPU: fp16). Given the input "你好" (hello in Chinese) and generating a reply with the same content, the time it takes to generate one word on average (s/word) is as follows:

|   impl  |   GPU + CPU   | CPU only  |
|---------|---------------|-----------|
|   MNN   |      0.292    |   0.877   |
| Pytorch | out of memory |   1.344   |



### Quantize Model
`TODO`

## Usage
### 1. Download this project
```bash
git clone https://github.com/wangzhaode/ChatGLM-MNN.git
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

- Copy the compilation result of the MNN library to ChatGLM-MNN
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

##### Docker
1. For Linux system, there are Nvidia graphics cards that support cuda, and those who want to run the fp16 model
```bash
# start compiling
docker build . -t chatglm-mnn:cuda_fp16_v0.1 -f dockerfiles/cuda_fp16.Dockerfile

# Run the web version directly, and then open the 5088 port of the corresponding ip in the browser to access the web page
docker run --gpus all -d -p 5088:5088 --restart always --name chatglm-mnn chatglm-mnn:cuda_fp16_v0.1

# Run directly (cli version)
docker run --gpus all -it \
  --name chatglm-mnn \
  chatglm-mnn:cuda_fp16_v0.1 \
  bash -c "cd /workspace/build && ./cli_demo"
```
2. For Linux system, there are Nvidia graphics cards that support cuda, and those who want to run the int8 model
```bash
# start compiling
docker build . -t chatglm-mnn:cuda_int8_v0.1 -f dockerfiles/cuda_int8.Dockerfile

# Run the web version directly
docker run --gpus all -d -p 5088:5088 --restart always --name chatglm-mnn chatglm-mnn:cuda_int8_v0.1

# Run directly (cli version)
docker run --gpus all -it \
  --name chatglm-mnn \
  chatglm-mnn:cuda_int8_v0.1 \
  bash -c "cd /workspace/build && ./cli_demo -d int8"
```

3. For Linux/Mac system, no Nvidia graphics card, want to run fp16 model (big concept requires 24G memory)。
```bash
# start compiling
docker build . -t chatglm-mnn:cpu_fp16_normal_v0.1 -f dockerfiles/cpu_fp16_normal.Dockerfile

# Run the web version directly
docker run -d -p 5088:5088 --restart always --name chatglm-mnn chatglm-mnn:cpu_fp16_normal_v0.1

# Run directly (cli version)
docker run -it \
  --name chatglm-mnn \
  chatglm-mnn:cpu_fp16_normal_v0.1 \
  bash -c "cd /workspace/build && ./cli_demo"
```
4. For Linux/Mac system, no Nvidia graphics card, want to run int8 model (big concept requires 24G memory)
```bash
# start compiling
docker build . -t chatglm-mnn:cpu_int8_normal_v0.1 -f dockerfiles/cpu_int8_normal.Dockerfile

# Run the web version directly
docker run -d -p 5088:5088 --restart always --name chatglm-mnn chatglm-mnn:cpu_int8_normal_v0.1

# Run directly (cli version)
docker run -it \
  --name chatglm-mnn \
  chatglm-mnn:cpu_int8_normal_v0.1 \
  bash -c "cd /workspace/build && ./cli_demo -d int8"

#### 5. Using GPU
The default usage is to use CPU. To use NVIDIA GPU, the macro -DMNN_CUDA=ON needs to be added when compiling MNN. To use other GPUs, the macro -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=ON -DMNN_SEP_BUILD=OFF needs to be added when compiling MNN.

Specify the size of the memory when creating ChatGLM as follows:

```cpp
// 8G CUDA Memory
ChatGLM chatglm(8);
// 命令式
std::cout << chatglm.response("Hello");
// 交互式
chatglm.chat();
```

## Example

Q: Can you help me write a quicksort code in Python?

A: Sure! Here is an example code for implementing the quicksort algorithm using Python:

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = []
        right = []
        for i in range(1, len(arr)):
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])
        return quick_sort(left) + [pivot] + quick_sort(right)
```
What is the meaning of the term "quantum supremacy"?

A: "Quantum supremacy" is a term used to describe the point at which a quantum computer can perform a calculation that is beyond the reach of even the most powerful classical computers. This means that the quantum computer can solve a problem in a time that is not feasible for a classical computer to solve, even if it were to run for millions or billions of years. Achieving quantum supremacy is seen as a significant milestone in the development of quantum computing, as it demonstrates the potential of quantum computers to solve problems that are currently intractable. However, it is important to note that quantum supremacy does not mean that quantum computers will be able to solve all problems faster than classical computers, but rather that they have unique capabilities that can be harnessed to solve specific problems more efficiently.

Q: How to locate memory leaks?

A: Memory leak refers to the situation where the occupied memory space of a program continuously increases during its runtime, eventually causing the program to fail or run inefficiently.

Here are some common methods for checking memory leaks:

Use memory analysis tools: such as Eclipse Memory Analyzer, VisualVM, etc. These tools can help detect memory space that has not been released in the program and abnormal memory allocation.

Check the lifecycle of variables: if a variable is not released during its lifecycle, its memory space will continue to be occupied, leading to memory leaks. Checking the lifecycle information of variables can help identify this issue.

Check for exceptions in the code: if the program encounters an exception during execution and does not release the occupied memory space, it can also lead to memory leaks. Debugging the program and finding and releasing such memory space can help resolve the issue.

Optimize the code: if the program has memory leak issues, it may be due to performance or excessive memory usage. Optimizing the code and reducing the frequency of memory allocation and release can help solve this problem.

Regularly clean up unused data: if there is a large amount of unused data in the program, it can also cause memory leaks. Regularly cleaning up such data and releasing memory space can help resolve this issue.

Solving memory leak issues requires multiple approaches. Continuously debugging and optimizing the program can help identify the root cause of memory leaks and effectively resolve the issue.

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
### 1. Compile MNN library
Compile MNN from source code

```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build && cd build
cmake .. # CPU only
cmake .. -DMNN_CUDA=ON # using CUDA
cmake .. -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=ON -DMNN_SEP_BUILD=OFF # using OPENCL
make -j8
cp -r ../include /path/to/ChatGLM-MNN/
cp libMNN.so /path/to/ChatGLM-MNN/libs
cp express/libMNN_Express.so /path/to/ChatGLM-MNN/libs
```
### 2. Download Models
Download model files from github release to /path/to/ChatGLM-MNN/resource/models, as follows:   
```
cd resource/models
./download_models_fp16.sh # 下载fp16权值模型
./download_models_int8.sh # 下载int8权值模型，推荐使用
./download_models_int4.sh # 下载int4权值模型，有精度损失
```
### 3. Build and Run
Mac/Linux/Windows:
```bash
mkdir build
cd build
cmake ..
make -j8
./cli_demo # cli demo
./web_demo # web ui demo
```

Android:
```
mkdir build
cd build
../android_build.sh
make -j8
```

#### 4. Using GPU
The default usage is to use CPU. To use NVIDIA GPU, the macro -DMNN_CUDA=ON needs to be added when compiling MNN. To use other GPUs, the macro -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=ON -DMNN_SEP_BUILD=OFF needs to be added when compiling MNN.

Specify the size of the memory when creating ChatGLM as follows:

```cpp
// 8G CUDA Memory
ChatGLM chatglm(8);
// 命令式
std::cout << chatglm.response("你好");
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

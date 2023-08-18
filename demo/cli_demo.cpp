//
//  cli_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "chat.hpp"
#include "CLI11.hpp"
#include <stdlib.h>
#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif // DEBUG

int main(int argc, const char* argv[]) {
    CLI::App app{"cli demo for chat.cpp"};
#ifdef WITH_CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device name: " << prop.name << std::endl; 
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    // Query device for FP16 support
    int fp16_support;
    cudaDeviceGetAttribute(&fp16_support, cudaDevAttrComputeCapabilityMajor, 0);
    if (fp16_support >= 7) {
        std::cout << "GPU Device supports FP16 computation" << std::endl;
    } else {
        std::cout << "GPU Device does not support FP16 computation" << std::endl;
    }
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double free_db_g = free_db / 1024.0 / 1024.0 / 1024.0;
    double total_db_g = total_db / 1024.0 / 1024.0 / 1024.0;
    std::cout << "Total global memory: " << total_db_g << "G" << std::endl;
    std::cout << "Total available memory: " << free_db_g << "G" << std::endl;
    float gpusize = free_db_g;
#else
    float gpusize = 0;
#endif
    float cpusize = 8;
    std::string model_dir = "../resource/models";
    std::string data_type = "fp16";
    std::string tokenizer_dir = "../resource/tokenizer";
    app.add_option("-c,--cpusize", cpusize,"cpu memory size(G), default is 8G.");
    app.add_option("-g,--gpusize", gpusize,"gpu memory size(G)");
	app.add_option("-m,--model_dir", model_dir, "model directory");
    app.add_option("-t,--tokenizer_dir", tokenizer_dir, "tokenizer directory");

	CLI11_PARSE(app, argc, argv);
    std::cout << "model path is " << model_dir << std::endl;
    ChatGLM chatglm;
    chatglm.load(cpusize, gpusize, model_dir, tokenizer_dir);
    // chatglm.chat();
    chatglm.response("你好");
    return 0;
}

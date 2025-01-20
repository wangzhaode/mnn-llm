//
//  cli_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

using namespace MNN::Transformer;

void benchmark(Llm* llm, std::string prompt_file) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        std::string::size_type pos = 0;
        while ((pos = prompt.find("\\n", pos)) != std::string::npos) {
            prompt.replace(pos, 2, "\n");
            pos += 1;
        }
        prompts.push_back(prompt);
    }
    int prompt_len = 0;
    int decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    auto& state = llm->getState();
    for (int i = 0; i < prompts.size(); i++) {
        const auto& prompt = prompts[i];
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        if (0) {
            llm->response(prompt, &std::cout, nullptr, 0);
            while (!llm->stoped() && state.gen_seq_len_ < 128) {
                llm->generate(1);
            }
        } else {
            llm->response(prompt);
        }
        prompt_len += state.prompt_len_;
        decode_len += state.gen_seq_len_;
        vision_time += state.vision_us_;
        audio_time += state.audio_us_;
        prefill_time += state.prefill_us_;
        decode_time += state.decode_us_;
    }
    float vision_s = vision_time / 1e6;
    float audio_s = audio_time / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num = %d\n", prompt_len);
    printf("decode tokens num = %d\n", decode_len);
    printf(" vision time = %.2f s\n", vision_s);
    printf("  audio time = %.2f s\n", audio_s);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " model_dir <prompt.txt>" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load();
    if (argc < 3) {
        llm->chat();
    }
    std::string prompt_file = argv[2];
    benchmark(llm.get(), prompt_file);
    return 0;
}

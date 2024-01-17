//
//  tokenizer_demo.cpp
//
//  Created by MNN on 2024/01/12.
//  ZhaodeWang
//

#include "tokenizer.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.txt" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    std::unique_ptr<Tokenizer> tokenizer_(new Tiktoken);
    tokenizer_->load(tokenizer_path);
    const std::string system_str = "Youare a helpful assistant.";
    const std::string user_str = "<|endoftext|>";
    // const std::string query = "\n<|im_start|>system\n" + system_str + "<|im_end|>\n<|im_start|>\n" + user_str + "<|im_end|>\n<|im_start|>assistant\n";
    const std::string query = system_str + "\n" + user_str;
    auto tokens = tokenizer_->encode(query);

    std::string decode_str;
    printf("encode tokens = [ ");
    for (auto token : tokens) {
        decode_str += tokenizer_->decode(token);
    }
    printf("]\n");
    printf("decode str = %s\n", decode_str.c_str());
    return 0;
}

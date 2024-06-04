//
//  tokenizer_demo.cpp
//
//  Created by MNN on 2024/01/12.
//  ZhaodeWang
//

#include "tokenizer.hpp"
#include <fstream>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.txt prompt.txt" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    std::string prompt_file = argv[2];
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(tokenizer_path));

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
        const std::string query = "\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        std::cout << query;
        auto tokens = tokenizer->encode(query);
        std::string decode_str;
        printf("encode tokens = [ ");
        for (auto token : tokens) {
            printf("%d, ", token);
            decode_str += tokenizer->decode(token);
        }
        printf("]\n");
        printf("decode str = %s\n", decode_str.c_str());
    }
    return 0;
}

//
//  pipeline_demo.cpp
//
//  Created by MNN on 2024/01/11.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

#include "json.hpp"

using json = nlohmann::json;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return 0;
    }
    std::string config_dir = argv[1];
    std::cout << "config_dir is " << config_dir << std::endl;
    std::unique_ptr<Pipeline> pipeline(Pipeline::load(config_dir));
    pipeline->invoke("你是谁？");
    pipeline->invoke("我曾经和你推荐过一部科幻电影，它的名字是？");
    return 0;
}

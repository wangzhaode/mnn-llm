//
//  knowledge_demo.cpp
//
//  Created by MNN on 2024/01/16.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " embedding.mnn knowledge.md " << std::endl;
        return 0;
    }
    std::string embedding_dir = argv[1];
    std::string knowledge_dir = argv[2];
    std::cout << "embedding file is " << embedding_dir << std::endl;
    std::cout << "knowledge file is " << knowledge_dir << std::endl;
    std::unique_ptr<Knowledge> knowledge(Knowledge::load(knowledge_dir));
    std::shared_ptr<Embedding> embedding(Embedding::createEmbedding(embedding_dir));
    knowledge->set_embedding(embedding);
    std::cout << "build vetcors for doc ...." << std::endl;
    knowledge->build_vectors();
    knowledge->save(knowledge_dir);
    const int k = 3;
    auto x = knowledge->search("请问我编译MNN之后，为什么build目录下没有找到`MNNConvert`呢？", k);
    for (int i = 0; i < k; i++) {
        std::cout << "top-" << i << " ### " << x[i] << std::endl;
    }
    return 0;
}
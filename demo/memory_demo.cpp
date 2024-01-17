//
//  memory_demo.cpp
//
//  Created by MNN on 2024/01/16.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " embedding.mnn history.json <llm.mnn>" << std::endl;
        return 0;
    }
    std::string embedding_dir = argv[1];
    std::string memory_dir = argv[2];
    std::cout << "embedding file is " << embedding_dir << std::endl;
    std::cout << "memory file is " << memory_dir << std::endl;
    std::unique_ptr<ChatMemory> chat_memory(ChatMemory::load(memory_dir));
    if (argc == 4) {
        auto llm_dir = argv[3];
        std::shared_ptr<Llm> llm(Llm::createLLM(llm_dir));
        llm->load(llm_dir);
        chat_memory->summarize(llm);
        chat_memory->save(memory_dir);
    }
    std::shared_ptr<Embedding> embedding(Embedding::createEmbedding(embedding_dir));
    chat_memory->set_embedding(embedding);
    chat_memory->build_vectors();
    chat_memory->save(memory_dir);
    auto querys = json::parse(R"(["我曾经和你推荐过一部科幻电影，它的名字是？", "我曾经在图书馆学习时看过一本小说，它的名字是？", "你曾经给我推荐过哪些画家？", "我曾经和你提到我去过绿禾公园，我在绿禾公园看到了什么景色？", "我曾经和你分享过一部文艺片《出租车司机》，它的内容是？", "我曾经在5月2日提到过我去了博物馆，你还记得我当时看了什么展览吗？", "我曾经在5月4日这天分享了一些我遇到的麻烦，是关于什么的？"])");
    for (auto& q : querys) {
        const int k = 1;
        auto x = chat_memory->search(q.dump(), k);
        std::cout << q << std::endl;
        for (int i = 0; i < k; i++) {
            std::cout << "top-" << i << " ### " << x[i] << std::endl;
        }
    }
    std::cout << chat_memory->get_latest("summary") << std::endl;
    std::cout << chat_memory->get_latest("personality") << std::endl;
    return 0;
}
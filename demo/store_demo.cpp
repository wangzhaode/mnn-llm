//
//  store_demo.cpp
//
//  Created by MNN on 2024/01/10.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

static void dumpVARP(VARP var) {
    auto size = var->getInfo()->size;
    auto ptr = var->readMap<float>();
    printf("[ ");
    for (int i = 0; i < 5; i++) {
        printf("%f, ", ptr[i]);
    }
    printf("... ");
    for (int i = size - 5; i < size; i++) {
        printf("%f, ", ptr[i]);
    }
    printf(" ]\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " embedding.mnn" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    std::shared_ptr<Embedding> embedding(Embedding::createEmbedding(model_dir));
    std::unique_ptr<TextVectorStore> store(new TextVectorStore(embedding));
    store->bench();
    std::vector<std::string> texts = {
        "在春暖花开的季节，走在樱花缤纷的道路上，人们纷纷拿出手机拍照留念。樱花树下，情侣手牵手享受着这绝美的春光。孩子们在树下追逐嬉戏，脸上洋溢着纯真的笑容。春天的气息在空气中弥漫，一切都显得那么生机勃勃，充满希望。",
        "春天到了，樱花树悄然绽放，吸引了众多游客前来观赏。小朋友们在花瓣飘落的树下玩耍，而恋人们则在这浪漫的景色中尽情享受二人世界。每个人的脸上都挂着幸福的笑容，仿佛整个世界都被春天温暖的阳光和满树的樱花渲染得更加美好。",
        "在炎热的夏日里，沙滩上的游客们穿着泳装享受着海水的清凉。孩子们在海边堆沙堡，大人们则在太阳伞下品尝冷饮，享受悠闲的时光。远处，冲浪者们挑战着波涛，体验着与海浪争斗的刺激。夏天的海滩，总是充满了活力和热情。"
    };
    store->add_texts(texts);
    std::string text = "春风轻拂过，公园里的花朵竞相开放，五彩斑斓地装点着大自然。游人如织，他们带着相机记录下这些美丽的瞬间。孩童们在花海中欢笑玩耍，无忧无虑地享受着春日的温暖。情侣们依偎在一起，沉醉于这迷人的季节。春天带来了新生与希望的讯息，让人心情愉悦，充满了对未来的美好憧憬。";
    auto similar_texts = store->search_similar_texts(text, 1);
    for (const auto& text : similar_texts) {
        std::cout << text << std::endl;
    }
    store->save("./tmp.mnn");
    store.reset(TextVectorStore::load("./tmp.mnn"));
    store->set_embedding(embedding);
    similar_texts = store->search_similar_texts(text, 2);
    for (const auto& text : similar_texts) {
        std::cout << text << std::endl;
    }
    return 0;
}

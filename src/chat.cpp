//
//  chat.cpp
//
//  Created by MNN on 2023/03/17.
//  ZhaodeWang
//

// #define MNN_OPEN_TIME_TRACE

#include <fstream>
#include <iostream>

#include "chat.hpp"
#include "cppjieba/Jieba.hpp"

void ChatGLM::chat() {
    while (true) {
        std::cout << "\nQ: ";
        std::string input_str;
        std::cin >> input_str;
        std::cout << "\nA: " << std::flush;
        response(input_str, true);
        std::cout << std::endl;
    }
}

std::string ChatGLM::response(const std::string& input_str, bool stream) {
    // init status
    mSeqLen = 0, mContextLen = -1, mMaskIdx = -1;
    if (mHistoryVars.empty()) mHistoryVars.resize(LAYER_SIZE);
    for (int i = 0; i < LAYER_SIZE; i++) {
        // init history
        mHistoryVars[i] = _Input({2, 0, 1, 32, 128}, NCHW);
    }
    // response
    auto input_ids = tokenizer_encode(input_str);
    int token = forward(input_ids);
    std::string output_str = decode(token);
    if (stream) std::cout << output_str << std::flush;
    while (token != EOS) {
        token = forward({token});
        auto word = decode(token);
        if (stream) std::cout << word << std::flush;
        output_str += word;
    }
    return output_str;
}

std::vector<int> ChatGLM::tokenizer_encode(std::string input_str) {
    std::vector<int> ids;
    std::vector<std::string> words;
    cppjieba::Jieba jieba(
        "../resource/tokenizer/jieba.dict.utf8",
        "../resource/tokenizer/hmm_model.utf8",
        "../resource/tokenizer/user.dict.utf8",
        "../resource/tokenizer/idf.utf8",
        "../resource/tokenizer/stop_words.utf8"
    );
    jieba.Cut(input_str, words, true);
    for (auto word : words) {
        const auto& iter = mWordEncode.find(word);
        if (iter != mWordEncode.end()) {
            ids.push_back(iter->second);
        }
    }
    ids.push_back(130001);
    ids.push_back(130004);
    return ids;
}

std::string ChatGLM::decode(int id) {
    auto word = mWordDecode[id];
    if (word == "<n>") return "\n";
    if (word == "<|tab|>") return "\t";
    int pos = word.find("<|blank_");
    if (pos != -1) {
        int space_num = atoi(word.substr(8, word.size() - 10).c_str());
        return std::string(space_num, ' ');
    }
    pos = word.find("▁");
    if (pos != -1) {
        word.replace(pos, pos + 3, " ");
    }
    return word;
}

void ChatGLM::init(float cuda_memory) {
    // 0. create runtime
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type          = MNN_FORWARD_CPU;
    config.numThread     = 4;
    config.backendConfig = &cpuBackendConfig;
    mCPURtmgr.reset(Executor::RuntimeManager::createRuntimeManager(config));
    BackendConfig cudaBackendConfig;
    config.type          = MNN_FORWARD_CUDA;
    cudaBackendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &cudaBackendConfig;
    mCUDARtmgr.reset(Executor::RuntimeManager::createRuntimeManager(config));
    // 1. load vocab
    printf("load ../resource/tokenizer/slim_vocab.txt ... ");
    std::ifstream dictFile("../resource/tokenizer/slim_vocab.txt");
    int index = 0;
    std::string word;
    while (dictFile >> word) {
        mWordDecode.push_back(word);
        mWordEncode.insert(std::make_pair<std::string, int>(std::move(word), index++));
    }
    printf("Done!\n");
    // 2. load models
    int cuda_run_layers = (cuda_memory - 2) * 1024.0 / 385.0;
    char buffer[50];
    for (int i = 0; i < LAYER_SIZE; i++) {
        sprintf(buffer, "../resource/models/glm_block_%d.mnn", i);
        loadModel(buffer, i <= cuda_run_layers);
    }
    // 3. load lm model
    loadModel("../resource/models/lm.mnn", false);
}

void ChatGLM::loadModel(const char* fileName, bool cuda) {
    printf("load %s model ... ", fileName);
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = true;
    auto rtmgr = cuda ? mCUDARtmgr : mCPURtmgr;
    std::shared_ptr<Module> net(Module::load({}, {}, fileName, rtmgr, &config));
    mModules.emplace_back(std::move(net));
    printf("Done!\n");
}

VARP ChatGLM::gen_embedding(const std::vector<int>& input_ids) {
    size_t seq_len = input_ids.size();
    auto embedding_var = _Input({static_cast<int>(seq_len), 1, HIDDEN_SIZE}, NCHW);
    constexpr size_t size = HIDDEN_SIZE * sizeof(float);
    FILE* file = fopen("../resource/models/slim_word_embeddings.bin", "rb");
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(embedding_var->writeMap<char>() + i * size, 1, size, file);
    }
    fclose(file);
    return embedding_var;
}

VARP ChatGLM::gen_attention_mask(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    // init mask
    if (seq_len > 1 && mMaskIdx == -1 && mContextLen == -1) {
        int gMaskIdx = -1;
        for (int i = 0; i < seq_len; i++) {
            if (input_ids[i] == MASK) {
                mMaskIdx = i;
            }
            if (input_ids[i] == gMASK) {
                gMaskIdx = i;
            }
            if (input_ids[i] == BOS) {
                mContextLen = i;
            }
        }
        if (mMaskIdx < 0) {
            mMaskIdx = gMaskIdx;
        }
    }
    // attention_mask
    auto attention_mask_var = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask_var->writeMap<int>();
    for (int i = 0; i < seq_len * seq_len; i++) {
        ptr[i] = 0;
    }
    if (seq_len > 1) {
        for (int i = 1; i < seq_len; i++) {
            ptr[seq_len * i - 1] = 1;
        }
    }
    return attention_mask_var;
}

VARP ChatGLM::gen_position_ids(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    // position_ids
    auto position_ids_var = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids_var->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = mMaskIdx;
        ptr[1] = mSeqLen - mContextLen;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
            ptr[seq_len + i] = 0;
        }
        ptr[2 * seq_len - 1] = 1;
    }
    return position_ids_var;
}

int ChatGLM::forward(const std::vector<int>& input_ids) {
    mSeqLen += input_ids.size();
    auto hidden_states = gen_embedding(input_ids);
    auto attention_mask = gen_attention_mask(input_ids);
    auto position_ids = gen_position_ids(input_ids);
    for (int i = 0; i < LAYER_SIZE; i++) {
        AUTOTIME;
        auto outputs = mModules[i]->onForward({hidden_states, attention_mask, position_ids, mHistoryVars[i]});
        hidden_states = outputs[0];
        mHistoryVars[i] = outputs[1];
    }
    return var_to_token(hidden_states);
}

int ChatGLM::var_to_token(VARP var) {
    AUTOTIME;
    int num = var->getInfo()->dim[0];
    if (num > 1) {
        var = _Gather(var, _Scalar<int>(num - 1));
    }
    var = _Reshape(var, {HIDDEN_SIZE, 1});
    auto outputs = mModules.back()->onForward({var});
    int id = outputs[0]->readMap<int>()[0];
    // printf("### %d\n", id);
    return id;
}
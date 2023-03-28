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
#include <sentencepiece_processor.h>

void ChatGLM::chat() {
    while (true) {
        std::cout << "\nQ: ";
        std::string input_str;
        std::cin >> input_str;
        std::cout << "\nA: " << std::flush;
        response(input_str);
        std::cout << std::endl;
    }
}

std::string ChatGLM::response(const std::string &input_str, std::ostream *os) {
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
    *os << output_str << std::flush;
    while (token != EOS) {
        token = forward({token});
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
    return output_str;
}

std::vector<int> ChatGLM::tokenizer_encode(const std::string &input_str) {
    std::string processed_text = input_str;
    size_t position = processed_text.find("\n");
    while (position != std::string::npos) {
        processed_text.replace(position, 1, "<n>");
        position = processed_text.find("\n", position + 3);
    }
    while ((position = processed_text.find('\t')) != std::string::npos) {
        processed_text.replace(position, 1, "<|tab|>");
    }
    for (int i = 80; i > 1; i--) {
        std::string spaces(i, ' ');
        std::string blank_token = "<|blank_" + std::to_string(i) + "|>";
        position = processed_text.find(spaces);
        while (position != std::string::npos) {
            processed_text.replace(position, i, blank_token);
            position = processed_text.find(spaces, position + blank_token.length());
        }
    }
    std::vector<int> ids;
    sp_processor.Encode(processed_text, &ids);
    ids.push_back(130001);
    ids.push_back(130004);
    return ids;
}

std::string ChatGLM::decode(int id) {
    std::vector<int> ids = {id};
    std::string word;
    sp_processor.Decode(ids, &word);
    if (word == "<n>") return "\n";
    if (word == "<|tab|>") return "\t";
    int pos = word.find("<|blank_");
    if (pos != -1) {
        int space_num = atoi(word.substr(8, word.size() - 10).c_str());
        return std::string(space_num, ' ');
    }
    return word;
}

void ChatGLM::init(float gpu_memory) {
    // 0. create runtime
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    config.backendConfig = &cpuBackendConfig;
    mCPURtmgr.reset(Executor::RuntimeManager::createRuntimeManager(config));
    BackendConfig gpuBackendConfig;
    config.type = MNN_FORWARD_CUDA;
    config.backupType = MNN_FORWARD_OPENCL;
    gpuBackendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &gpuBackendConfig;
    mGPURtmgr.reset(Executor::RuntimeManager::createRuntimeManager(config));
    // 1. load tokenizer
    printf("load ../resource/tokenizer/ice_text_new.model ... \n");
    int index = 0;
    const auto status = sp_processor.Load("../resource/tokenizer/ice_text_new.model");
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        // error
    }
    // 2. load models
    int gpu_run_layers = (gpu_memory - 2) * 1024.0 / 385.0;
    char buffer[50];
    for (int i = 0; i < LAYER_SIZE; i++) {
        sprintf(buffer, "../resource/models/glm_block_%d.mnn", i);
        loadModel(buffer, i <= gpu_run_layers);
    }
    // 3. load lm model
    loadModel("../resource/models/lm.mnn", false);
}

void ChatGLM::loadModel(const char *fileName, bool cuda) {
    printf("load %s model ... ", fileName);
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = true;
    auto rtmgr = cuda ? mGPURtmgr : mCPURtmgr;
    std::shared_ptr<Module> net(Module::load({}, {}, fileName, rtmgr, &config));
    mModules.emplace_back(std::move(net));
    printf("Done!\n");
}

VARP ChatGLM::gen_embedding(const std::vector<int> &input_ids) {
    size_t seq_len = input_ids.size();
    auto embedding_var = _Input({static_cast<int>(seq_len), 1, HIDDEN_SIZE}, NCHW);
    constexpr size_t size = HIDDEN_SIZE * sizeof(float);
    FILE *file = fopen("../resource/models/slim_word_embeddings.bin", "rb");
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(embedding_var->writeMap<char>() + i * size, 1, size, file);
    }
    fclose(file);
    return embedding_var;
}

VARP ChatGLM::gen_attention_mask(const std::vector<int> &input_ids) {
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

VARP ChatGLM::gen_position_ids(const std::vector<int> &input_ids) {
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

int ChatGLM::forward(const std::vector<int> &input_ids) {
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
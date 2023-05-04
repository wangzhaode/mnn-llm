//
//  chat.cpp
//
//  Created by MNN on 2023/03/17.
//  ZhaodeWang
//

// #define MNN_OPEN_TIME_TRACE

#include <fstream>
#include <iostream>
#ifdef MINI_MEM_MODE
#include <thread>
#endif

#include "chat.hpp"
#include "cppjieba/Jieba.hpp"
#include <MNN/expr/ExecutorScope.hpp>

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

void ChatGLM::reset() {
    mHistoryStr = "";
    mChatRound = 0;
}

std::string ChatGLM::response(const std::string& input_str, std::ostream* os) {
    AUTOTIME;
    // init status
    mSeqLen = 0, mContextLen = -1, mMaskIdx = -1;
    if (mHistoryVars.empty()) mHistoryVars.resize(LAYER_SIZE);
    for (int i = 0; i < LAYER_SIZE; i++) {
        // init history
        mHistoryVars[i] = _Input({2, 0, 1, 32, 128}, NCHW);
    }
    // response
    mHistoryStr += ("[Round " + std::to_string(mChatRound++) + "]\n问：" + input_str);
    auto prompt = mChatRound > 1 ? mHistoryStr : input_str;
    auto input_ids = tokenizer_encode(prompt);
    int token = forward(input_ids);
    std::string output_str = decode(token);
    *os << output_str << std::flush;
    while (token != EOS) {
        AUTOTIME;
        token = forward({token});
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
    mHistoryStr += ("\n答：" + output_str + "\n");
    return output_str;
}

std::vector<int> ChatGLM::tokenizer_encode(std::string input_str) {
    std::vector<int> ids;
    std::vector<std::string> words;
    std::string dict_path = mTokenizerDir + "/jieba.dict.utf8";
    std::string model_path = mTokenizerDir + "/hmm_model.utf8";
    std::string user_dict_path = mTokenizerDir + "/user.dict.utf8";
    std::string idf_path = mTokenizerDir + "/idf.utf8";
    std::string stopWord_path = mTokenizerDir + "/stop_words.utf8";
    cppjieba::Jieba jieba(
        dict_path,
        model_path,
        user_dict_path,
        idf_path,
        stopWord_path
    );
    jieba.Cut(input_str, words, true);
    for (auto word : words) {
        const auto& iter = mWordEncode.find(word);
        if (iter != mWordEncode.end()) {
            ids.push_back(iter->second);
        }
    }
    ids.push_back(gMASK);
    ids.push_back(BOS);
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
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length()-1] == '>' && word[1] == '0' && word[2] == 'x') {

        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word = static_cast<char>(num);
    }
    return word;
}

void ChatGLM::init(float gpu_memory) {
    // 0. create runtime
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type          = MNN_FORWARD_CPU;
    config.numThread     = 4;
    cpuBackendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &cpuBackendConfig;
    mCPURtmgr.reset(Executor::RuntimeManager::createRuntimeManager(config));
    BackendConfig gpuBackendConfig;
    config.type          = MNN_FORWARD_CUDA;
    config.backupType    = MNN_FORWARD_OPENCL;
    config.numThread     = 1;
    gpuBackendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &gpuBackendConfig;
    mGPURtmgr.reset(Executor::RuntimeManager::createRuntimeManager(config));
    // 1. load vocab
    std::string dictFilePath = mTokenizerDir + "/slim_vocab.txt";
    printf("load %s ... ", dictFilePath.c_str());
    std::ifstream dictFile(dictFilePath);
    int index = 0;
    std::string word;
    while (dictFile >> word) {
        mWordDecode.push_back(word);
        mWordEncode.insert(std::make_pair<std::string, int>(std::move(word), index++));
    }
    printf("Done!\n");
    // 2. load models
    mModules.resize(LAYER_SIZE + 1);
    int gpu_run_layers = (gpu_memory - 2) * 1024.0 / 385.0;
    char buffer[50];
#ifdef MINI_MEM_MODE
    std::string model0 = mModelDir + "/glm_block_0.mnn";
    loadModel(model0.c_str(), false, 0);
#else
    for (int i = 0; i < LAYER_SIZE; i++) {
        std::string model_path = mModelDir + "/glm_block_" + std::to_string(i) + ".mnn";
        printf("[%3.0f%% ] ", (i + 1) * 100.0 / LAYER_SIZE);
        loadModel(model_path.c_str(), i <= gpu_run_layers, i);
        fflush(stdout);
    }
    // 3. load lm model
    std::string lm_model_path = mModelDir + "/lm.mnn";
    loadModel(lm_model_path.c_str(), false, LAYER_SIZE);
#endif
}

void ChatGLM::loadModel(const char* fileName, bool cuda, int i) {
    // AUTOTIME;
#ifndef MINI_MEM_MODE
    printf("load %s model ... ", fileName);
#endif
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = true;
    auto rtmgr = cuda ? mGPURtmgr : mCPURtmgr;
    std::shared_ptr<Module> net(Module::load({}, {}, fileName, rtmgr, &config));
    mModules[i] = std::move(net);
#ifndef MINI_MEM_MODE
    printf("Done!\n");
#endif
}

VARP ChatGLM::gen_embedding(const std::vector<int>& input_ids) {
    size_t seq_len = input_ids.size();
    auto embedding_var = _Input({static_cast<int>(seq_len), 1, HIDDEN_SIZE}, NCHW);
    constexpr size_t size = HIDDEN_SIZE * sizeof(float);
    std::string file_path = mModelDir + "/slim_word_embeddings.bin";
    FILE* file = fopen(file_path.c_str(), "rb");
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
    AUTOTIME;
    mSeqLen += input_ids.size();
    auto hidden_states = gen_embedding(input_ids);
    auto attention_mask = gen_attention_mask(input_ids);
    auto position_ids = gen_position_ids(input_ids);
#ifdef MINI_MEM_MODE
    char buffer[50];
    for (int i = 0; i < LAYER_SIZE; i++) {
        int loadIdx = i < LAYER_SIZE - 1 ? i + 1 : 0;
        std::string model_path = mModelDir + "/glm_block_" + std::to_string(loadIdx) + ".mnn";
        std::thread load_next_model(&ChatGLM::loadModel, this, model_path.c_str(), false, loadIdx);
        {
            // AUTOTIME;
            auto outputs = mModules[i]->onForward({hidden_states, attention_mask, position_ids, mHistoryVars[i]});
            hidden_states = outputs[0];
            mHistoryVars[i] = outputs[1];
        }
        mModules[i].reset();
        load_next_model.join();
    }
#else
    for (int i = 0; i < LAYER_SIZE; i++) {
        // AUTOTIME;
        auto outputs = mModules[i]->onForward({hidden_states, attention_mask, position_ids, mHistoryVars[i]});
        hidden_states = outputs[0];
        mHistoryVars[i] = outputs[1];
    }
#endif
    return var_to_token(hidden_states);
}

int ChatGLM::var_to_token(VARP var) {
    // AUTOTIME;
    int num = var->getInfo()->dim[0];
    if (num > 1) {
        var = _Gather(var, _Scalar<int>(num - 1));
    }
    var = _Reshape(var, {HIDDEN_SIZE, 1});
#ifdef MINI_MEM_MODE
    // naive impl to save memory : gemm + argmax
    auto ptr = var->readMap<float>();
    constexpr int TILE = 512;
    std::string file_path = mModelDir + "/slim_lm.bin";
    FILE* file = fopen(file_path.c_str(), "rb");
    std::vector<float> buffer(TILE * HIDDEN_SIZE);
    int id = -1;
    float max_score = 0.f;
    for (size_t i = 0; i < VOCAB_SIZE / TILE; i++) {
        fseek(file, i * TILE * HIDDEN_SIZE * sizeof(float), SEEK_SET);
        fread(reinterpret_cast<char*>(buffer.data()), 1, TILE * HIDDEN_SIZE * sizeof(float), file);
        for (int j = 0; j < TILE; j++) {
            float sum = 0.f;
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += (buffer[j * HIDDEN_SIZE + k]) * ptr[k];
            }
            if (sum > max_score) {
                max_score = sum;
                id = i * TILE + j;
            }
        }
    }
    {
        int i = VOCAB_SIZE / TILE;
        constexpr int tile = VOCAB_SIZE % TILE;
        fseek(file, i * TILE * HIDDEN_SIZE * sizeof(float), SEEK_SET);
        fread(reinterpret_cast<char*>(buffer.data()), 1, tile * HIDDEN_SIZE * sizeof(float), file);
        for (int j = 0; j < tile; j++) {
            float sum = 0.f;
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += (buffer[j * HIDDEN_SIZE + k]) * ptr[k];
            }
            if (sum > max_score) {
                max_score = sum;
                id = i * TILE + j;
            }
        }
    }
    fclose(file);
#else
    auto outputs = mModules.back()->onForward({var});
    int id = outputs[0]->readMap<int>()[0];
#endif
    // printf("### %d\n", id);
    return id;
}

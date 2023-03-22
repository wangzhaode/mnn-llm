//
//  MNNV2Basic.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#define MNN_OPEN_TIME_TRACE

#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif
#include <MNN/MNNDefine.h>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include "cppjieba/Jieba.hpp"

using namespace MNN;

static void dumpTensor(const Tensor* t, const char* name = nullptr) {
    printf("%s :[ ", name ? name : "#");
    for (int i = 0; i < 5; i++) {
        printf("%f, ", t->host<float>()[i]);
    }
    printf(" ... ");
    int size = t->elementSize();
    for (int i = size - 5; i < size; i++) {
        printf("%f, ", t->host<float>()[i]);
    }
    printf("]\n");
}

class ChatGLM {
public:
    ChatGLM() {
        mConfig.type          = MNN_FORWARD_CPU;
        mConfig.numThread     = 4;
        mConfig.backendConfig = &mBackendConfig;
        load();
    }
    void chat();
    std::string forward(const std::string& input_str, bool print);
private:
    void load();
    void loadModel(const char* fileName);
    std::vector<int> tokenizer_encode(std::string input_str);
    std::string decode(const std::vector<int>& ids);
    std::vector<float> embedding(const std::vector<int>& input_ids);
    int to_token(const Tensor* tensor);
    int first_token(const std::vector<int>& input_ids);
    int next_token(int id);
private:
    static constexpr int \
    MASK = 150000, gMASK = 150001,
    BOS = 150004, EOS = 150005;
private:
    std::vector<std::string> mWordDecode;
    std::unordered_map<std::string, int> mWordEncode;
    // MNN Sessions
    ScheduleConfig mConfig;
    BackendConfig mBackendConfig;
    std::vector<std::shared_ptr<Interpreter>> mNets;
    std::vector<Session*> mSessions;
    // inputs
    std::vector<Tensor*> mInputsEmbeds, mAttentionMask, mPositionIds, mPastKeyValues;
    // outputs
    std::vector<Tensor*> mHiddenStates, mPresents;
    // history
    std::vector<std::vector<float>> mHistory;
    // mask info
    int mSeqLen, mContextLen, mMaskIdx;
};

void ChatGLM::chat() {
    while (true) {
        std::cout << "\n请输入 >>> ";
        std::string input_str;
        std::cin >> input_str;
        std::cout << forward(input_str, false) << std::endl;
    }
}

std::string ChatGLM::forward(const std::string& input_str, bool print) {
    auto input_ids = tokenizer_encode(input_str);
    printf("ids is : [ ");
    for (int i = 0; i < input_ids.size(); i++) {
        printf("%d, ", input_ids[i]);
    }
    printf("]\n");
    int token = first_token(input_ids);
    std::string output_str;
    output_str += mWordDecode[token];
    if (print) std::cout << mWordDecode[token];
    // 3. next tokens
    while (token != EOS) {
        token = next_token(token);
        output_str += mWordDecode[token];
        if (print) std::cout << mWordDecode[token];
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
    ids.push_back(20005);
    for (const auto& word : words) {
        const auto& iter = mWordEncode.find(word);
        if (iter != mWordEncode.end()) {
            ids.push_back(iter->second);
        }
    }
    ids.push_back(150001);
    ids.push_back(150004);
    return ids;
}

std::string ChatGLM::decode(const std::vector<int>& ids) {
    std::string response = "";
    for (int i = 0; i < ids.size(); i++) {
        response += mWordDecode[ids[i]];
    }
    return response;
}

std::vector<float> ChatGLM::embedding(const std::vector<int>& input_ids) {
    size_t word_nums = input_ids.size();
    std::vector<float> buffer(word_nums * 4096);
    constexpr size_t size = 4096 * sizeof(float);
    FILE* file = fopen("../resource/models/word_embeddings.bin", "rb");
    for (size_t i = 0; i < word_nums; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(reinterpret_cast<char*>(buffer.data()) + i * size, 1, size, file);
    }
    fclose(file);
    return buffer;
}

void ChatGLM::load() {
    // 1. load vocab
    printf("load ../resource/tokenizer/vocab.txt\n");
    std::ifstream dictFile("../resource/tokenizer/vocab.txt");
    int index = 0;
    std::string word;
    while (dictFile >> word) {
        mWordDecode.push_back(word);
        mWordEncode.insert(std::make_pair<std::string, int>(std::move(word), index++));
    }
    // 2. load models
    char buffer[50];
    for (int i = 0; i < 28; i++) {
        sprintf(buffer, "../resource/models/glm_block_%d.mnn", i);
        loadModel(buffer);
    }
}

void ChatGLM::loadModel(const char* fileName) {
    printf("load %s model\n", fileName);
    std::shared_ptr<Interpreter> net = std::shared_ptr<Interpreter>(Interpreter::createFromFile(fileName), Interpreter::destroy);
    net->setSessionMode(Interpreter::Session_Resize_Defer);
    net->setSessionMode(Interpreter::Session_Input_User);
    Session* session = net->createSession(mConfig);
    auto inputs_embeds = net->getSessionInput(session, "inputs_embeds");
    auto attention_mask = net->getSessionInput(session, "attention_mask");
    auto position_ids = net->getSessionInput(session, "position_ids");
    auto past_key_values = net->getSessionInput(session, "past_key_values");
    auto hidden_states = net->getSessionOutput(session, "hidden_states");
    auto presents = net->getSessionOutput(session, "presents");
    mInputsEmbeds.push_back(inputs_embeds);
    mAttentionMask.push_back(attention_mask);
    mPositionIds.push_back(position_ids);
    mPastKeyValues.push_back(past_key_values);
    mHiddenStates.push_back(hidden_states);
    mPresents.push_back(presents);
    mSessions.push_back(session);
    mNets.push_back(std::move(net));
}

int ChatGLM::first_token(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    // init mask info
    mSeqLen = seq_len;
    mMaskIdx = -1;
    mContextLen = -1;
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
    // inputs_embeds
    auto inputs_embeds_vals = embedding(input_ids);
    // attention_mask
    std::vector<int> attention_mask_vals(seq_len * seq_len, 0);
    if (seq_len > 1) {
        for (int i = 1; i < seq_len; i++) {
            attention_mask_vals[seq_len * i - 1] = 1;
        }
    }
    // position_ids
    std::vector<int> position_ids_vals(seq_len * 2);
    for (int i = 0; i < seq_len; i++) {
        position_ids_vals[i] = i;
        position_ids_vals[2*i] = 0;
    }
    position_ids_vals[2*seq_len-1] = 1;
    uint8_t* inputs_embeds_ptr = (uint8_t*)inputs_embeds_vals.data();
    uint8_t* attention_mask_ptr = (uint8_t*)attention_mask_vals.data();
    uint8_t* position_ids_ptr = (uint8_t*)position_ids_vals.data();
    const Tensor *hidden_states = nullptr, *presents = nullptr;
    for (int i = 0; i < mSessions.size(); i++) {
        AUTOTIME;
        // size
        mNets[i]->resizeTensor(mInputsEmbeds[i], {seq_len, 1, 4096});
        mNets[i]->resizeTensor(mAttentionMask[i], {1, 1, seq_len, seq_len});
        mNets[i]->resizeTensor(mPositionIds[i], {1, 2, seq_len});
        mNets[i]->resizeTensor(mPastKeyValues[i], {2, 0, 1, 32, 128});
        // set input
        mInputsEmbeds[i]->buffer().host = inputs_embeds_ptr;
        mAttentionMask[i]->buffer().host = attention_mask_ptr;
        mPositionIds[i]->buffer().host = position_ids_ptr;
        mNets[i]->resizeSession(mSessions[i]);
        mNets[i]->runSession(mSessions[i]);
        hidden_states = mHiddenStates[i];
        presents = mPresents[i];
        std::vector<float> presents_copy(presents->elementSize());
        ::memcpy(presents_copy.data(), presents->host<float>(), presents_copy.size() * sizeof(float));
        mHistory.emplace_back(std::move(presents_copy));
        inputs_embeds_ptr = hidden_states->host<uint8_t>();
    }
    return to_token(hidden_states);
}

int ChatGLM::next_token(int id) {
    // seqlen add this token
    mSeqLen++;
    // inputs_embeds
    auto inputs_embeds_vals = embedding({id});
    // attention_mask
    std::vector<int> attention_mask_vals { 0 };
    // position_ids
    std::vector<int> position_ids_vals {mMaskIdx, mSeqLen - mContextLen};
    uint8_t* inputs_embeds_ptr = (uint8_t*)inputs_embeds_vals.data();
    uint8_t* attention_mask_ptr = (uint8_t*)attention_mask_vals.data();
    uint8_t* position_ids_ptr = (uint8_t*)position_ids_vals.data();
    const Tensor *hidden_states = nullptr, *presents = nullptr;
    for (int i = 0; i < mSessions.size(); i++) {
        AUTOTIME;
        mNets[i]->resizeTensor(mInputsEmbeds[i], {1, 1, 4096});
        mNets[i]->resizeTensor(mAttentionMask[i], {1, 1});
        mNets[i]->resizeTensor(mPositionIds[i], {1, 2, 1});
        mNets[i]->resizeTensor(mPastKeyValues[i], {2, mSeqLen-1, 1, 32, 128});
        // set input
        mInputsEmbeds[i]->buffer().host = inputs_embeds_ptr;
        mAttentionMask[i]->buffer().host = attention_mask_ptr;
        mPositionIds[i]->buffer().host = position_ids_ptr;
        mPastKeyValues[i]->buffer().host = (uint8_t*)(mHistory[i].data());
        mNets[i]->resizeSession(mSessions[i]);
        mNets[i]->runSession(mSessions[i]);
        hidden_states = mHiddenStates[i];
        presents = mPresents[i];
        std::vector<float> presents_copy(presents->elementSize());
        ::memcpy(presents_copy.data(), presents->host<float>(), presents_copy.size() * sizeof(float));
        mHistory[i] = std::move(presents_copy);
        inputs_embeds_ptr = hidden_states->host<uint8_t>();
    }
    return to_token(hidden_states);
}

int ChatGLM::to_token(const Tensor* tensor) {
    AUTOTIME;
    auto num = tensor->shape()[0];
    auto ptr = tensor->host<float>() + (num - 1) * 4096;
    // [150528, 4096] -> [147, 1024, 4096]
    FILE* file = fopen("../resource/models/lm.bin", "rb");
    std::vector<float> buffer(1024 * 4096);
    int id = -1;
    float max_score = 0.f;
    // TODO: naive gemm impl + argmax
    for (size_t i = 0; i < 147; i++) {
        fseek(file, i * 1024 * 4096 * sizeof(float), SEEK_SET);
        fread(reinterpret_cast<char*>(buffer.data()), 1, 1024 * 4096 * sizeof(float), file);
        for (int j = 0; j < 1024; j++) {
            double sum = 0.f;
            for (int k = 0; k < 4096; k++) {
                sum += (double)(buffer[j * 4096 + k]) * ptr[k];
            }
            if (sum > max_score) {
                max_score = sum;
                id = i * 1024 + j;
            }
        }
    }
    fclose(file);
    printf("### %d : %s\n", id, mWordDecode[id].c_str());
    return id;
}

int main(int argc, const char* argv[]) {
    ChatGLM chatglm;
    chatglm.chat();
    return 0;
}
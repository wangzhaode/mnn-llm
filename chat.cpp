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
#include <core/Backend.hpp>
#include <core/TensorUtils.hpp>
#include <MNN_generated.h>

//#define FEED_INPUT_NAME_VALUE

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
    }
    void load(const char* fileName);
    void forward();
    std::vector<float> embedding(const std::vector<int>& input_ids);
private:
    ScheduleConfig mConfig;
    BackendConfig mBackendConfig;
    std::vector<std::shared_ptr<Interpreter>> mNets;
    std::vector<Session*> mSessions;
    // inputs
    std::vector<Tensor*> mInputsEmbeds, mAttentionMask, mPositionIds, mPastKeyValues;
    // outputs
    std::vector<Tensor*> mHiddenStates, mPresents;
};

void ChatGLM::load(const char* fileName) {
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

void ChatGLM::forward() {
    // inputs_embeds
    std::vector<int> input_ids { 20005,  94874, 150001, 150004 };
    auto inputs_embeds_vals = embedding(input_ids);
    // attention_mask
    std::vector<int> attention_mask_vals {
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,0
    };
    // position_ids
    std::vector<int> position_ids_vals {
        0,1,2,3, 0,0,0,1
    };
    uint8_t* inputs_embeds_ptr = (uint8_t*)inputs_embeds_vals.data();
    uint8_t* attention_mask_ptr = (uint8_t*)attention_mask_vals.data();
    uint8_t* position_ids_ptr = (uint8_t*)position_ids_vals.data();
    const Tensor *hidden_states = nullptr, *presents = nullptr;
    for (int i = 0; i < mSessions.size(); i++) {
        // set input
        mInputsEmbeds[i]->buffer().host = inputs_embeds_ptr;
        mAttentionMask[i]->buffer().host = attention_mask_ptr;
        mPositionIds[i]->buffer().host = position_ids_ptr;
        mNets[i]->resizeSession(mSessions[i]);
        {
            AUTOTIME;
            mNets[i]->runSession(mSessions[i]);
        }
        hidden_states = mHiddenStates[i];
        dumpTensor(hidden_states, "hidden_states");
        presents = mPresents[i];
        inputs_embeds_ptr = hidden_states->host<uint8_t>();
    }
    dumpTensor(hidden_states, "hidden_states");
    // dumpTensor(presents, "presents");
}

std::vector<float> ChatGLM::embedding(const std::vector<int>& input_ids) {
    size_t word_nums = input_ids.size();
    std::vector<float> buffer(word_nums * 4096);
    constexpr size_t size = 4096 * sizeof(float);
    FILE* file = fopen("/home/zhaode/ChatGLM-MNN/chat_model/transformer.word_embeddings.weight", "rb");
    for (size_t i = 0; i < word_nums; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(reinterpret_cast<char*>(buffer.data()) + i * size, 1, size, file);
    }
    fclose(file);
    printf("embedding value is : [");
    for (int i = 0; i < 5; i++) {
        printf("%f, ", buffer[i]);
    }
    printf(" ... ");
    for (int i = word_nums * 4096 - 5; i < word_nums * 4096; i++) {
        printf("%f, ", buffer[i]);
    }
    printf("]\n");
}

int main(int argc, const char* argv[]) {
    ChatGLM chatglm;
    char buffer[50];
    for (int i = 0; i < 28; i++) {
        sprintf(buffer, "model/glm_block_%d.mnn", i);
        chatglm.load(buffer);
    }
    printf("wait key input to continue ...\n");
    getchar();
    chatglm.forward();
    return 0;
}
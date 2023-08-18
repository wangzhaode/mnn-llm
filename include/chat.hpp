//
//  chat.hpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#ifndef CHAT_hpp
#define CHAT_hpp

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

using namespace MNN;
using namespace Express;

#ifdef CHATGLM_1
static constexpr int MASK = 130000;
static constexpr int gMASK = 130001;
static constexpr int BOS = 130004;
static constexpr int EOS = 130005;
static constexpr int VOCAB_SIZE = 130528;
#else
static constexpr int MASK = 130000;
static constexpr int gMASK = 130001;
static constexpr int BOS = 1;
static constexpr int EOS = 2;
static constexpr int VOCAB_SIZE = 65024;
#endif
static constexpr int HIDDEN_SIZE = 4096;
static constexpr int LAYER_SIZE = 28;

class ChatGLM {
public:
    ChatGLM() {}
    // set gpu memory size (G)
    void load(float cpu_memory = 8, float gpu_memory = 0) {
        init(cpu_memory, gpu_memory);
    }
    void load(float cpu_memory, float gpu_memory, const std::string& model_dir, const std::string& tokenizer_dir) {
        mModelDir = model_dir;
        mTokenizerDir = tokenizer_dir;
        init(cpu_memory, gpu_memory);
    };
    float loadProgress() { return mLoadProgress; }
    void chat();
    void reset();
    std::string response(const std::string& input_str, std::ostream* os = &std::cout);
private:
    void init(float cpu_memory, float gpu_memory);
    void loadModel(const char* fileName, bool cuda, int index);
    std::vector<int> tokenizer_encode(std::string input_str);
    std::string decode(int id);
    VARP gen_embedding(const std::vector<int>& input_ids);
    VARP gen_attention_mask(const std::vector<int>& input_ids);
    VARP gen_position_ids(const std::vector<int>& input_ids);
    int var_to_token(VARP var);
    int forward(const std::vector<int>& input_ids);
private:
    float mLoadProgress = 0;
    std::vector<std::string> mWordDecode;
    std::unordered_map<std::string, int> mWordEncode;
    // MNN Modules
    std::shared_ptr<Executor::RuntimeManager> mCPURtmgr;
    std::shared_ptr<Executor::RuntimeManager> mGPURtmgr;
    std::vector<std::shared_ptr<Module>> mModules;
    std::vector<VARP> mHistoryVars;
    // mask info
    int mSeqLen = 0, mContextLen = -1, mMaskIdx = -1;
    int mToks = 0;
    // model dir
    std::string mModelDir = "../resource/models/fp16";
    std::string mTokenizerDir = "../resource/tokenize";

    // history
    int mChatRound = 0;
    std::string mHistoryStr = "";
};

#endif // CHAT_hpp

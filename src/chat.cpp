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
#include <MNN/expr/ExecutorScope.hpp>

typedef unsigned char BYTE;
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(BYTE c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  BYTE char_array_4[4], char_array_3[3];
  // std::vector<BYTE> ret;
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
          ret.push_back(char_array_3[i]);
      i = 0;
    }
  }

  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
  }

  return ret;
}

void ChatGLM::chat() {
    while (true) {
        std::cout << "\nQ: ";
        std::string input_str;
        std::cin >> input_str;
        if (input_str == "/reset") {
            reset();
            continue;
        }
        if (input_str == "/quit") {
            break;
        }
        std::cout << "\nA: " << std::flush;
        response(input_str);
        std::cout << std::endl;
    }
}

void ChatGLM::reset() {
    mHistoryStr.clear();
    mChatRound = 0;
}

std::string ChatGLM::response(const std::string& input_str, std::ostream* os) {
    // AUTOTIME;
    // init status
    mSeqLen = 0, mContextLen = -1, mMaskIdx = -1, mToks = 0;
    if (mHistoryVars.empty()) mHistoryVars.resize(LAYER_SIZE);
    for (int i = 0; i < LAYER_SIZE; i++) {
        // init history
        //mHistoryVars[i] = _Input({2, 0, 1, 32, 128}, NCHW);
        mHistoryVars[i] = _Input({2, 1, 0, 32, 128}, NCHW);
    }
    // response
    mHistoryStr += ("[Round " + std::to_string(mChatRound++) + "]\n问：" + input_str);
    auto prompt = mChatRound > 1 ? mHistoryStr : input_str;
    auto st = std::chrono::system_clock::now();
    mToks = 1;
    auto input_ids = tokenizer_encode(prompt);
    int token = forward(input_ids);
    std::string output_str = decode(token);
    *os << output_str << std::flush;
    while (token != EOS) {
        // AUTOTIME;
        mToks++;
        token = forward({token});
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
    auto et = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(et - st);
    printf("\nspeed: %f tok/s\n", mToks / (duration.count() * 1e-6));
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
    /*
    ids = {151644, 8948, 198, 2610, 525, 264,  10950,  17847,     13,
                151645,    198, 151644,    872,    198, 108386, 151645,    198, 151644,
                77091,    198};
    */
    printf("ids = { ");
    for (auto id : ids) {
        printf("%d, ", id);
    }
    printf(" }\n");
    // ids.push_back(gMASK);
    // ids.push_back(BOS);
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

void ChatGLM::init(float cpu_memory, float gpu_memory) {
    // AUTOTIME;
    // 0. create runtime
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type          = MNN_FORWARD_CPU;
    config.numThread     = 4;
    cpuBackendConfig.precision = BackendConfig::Precision_Low;
    if (cpu_memory < 24) {
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
    }
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
    {
        AUTOTIME;
        // std::string dictFilePath = mTokenizerDir + "/slim_vocab.txt";
        std::string dictFilePath = mTokenizerDir + "/vocab_3.txt";
        printf("load %s ... ", dictFilePath.c_str());
        std::ifstream dictFile(dictFilePath);
        int index = 0;
        std::string word;
        while (dictFile >> word) {
            word = base64_decode(word);
            mWordDecode.push_back(word);
            mWordEncode.insert(std::make_pair<std::string, int>(std::move(word), index++));
        }
        printf("Done!\n");
    }
    mLoadProgress = 1.5;
    // 2. load models
    mModules.resize(LAYER_SIZE + 1);
    int gpu_run_layers = (gpu_memory - 2) * 1024.0 / 385.0;
    char buffer[50];
    // load lm model
    std::string lm_model_path = mModelDir + "/lm.mnn";
    loadModel(lm_model_path.c_str(), false, LAYER_SIZE);
    mLoadProgress = 8.46;
    printf("[%3.0f%% ] ", mLoadProgress);
    // load glm_block models
    for (int i = 0; i < LAYER_SIZE; i++) {
        std::string model_path = mModelDir + "/glm_block_" + std::to_string(i) + ".mnn";
        loadModel(model_path.c_str(), i <= gpu_run_layers, i);
        mLoadProgress += 3.27;
        printf("[%3.0f%% ] ", mLoadProgress);
        fflush(stdout);
    }
}

void ChatGLM::loadModel(const char* fileName, bool cuda, int i) {
    AUTOTIME;
    printf("load %s model ... ", fileName);
    Module::Config config;
    config.shapeMutable = true;
    config.rearrange = true;
    auto rtmgr = cuda ? mGPURtmgr : mCPURtmgr;
    std::vector<std::string> input_names {};
    std::shared_ptr<Module> net(Module::load({}, {}, fileName, rtmgr, &config));
    mModules[i] = std::move(net);
    printf("Done!\n");
}

VARP ChatGLM::gen_embedding(const std::vector<int>& input_ids) {
    size_t seq_len = input_ids.size();
    auto embedding_var = _Input({1, static_cast<int>(seq_len), HIDDEN_SIZE}, NCHW);
    constexpr size_t size = HIDDEN_SIZE * sizeof(int16_t);
    std::string file_path = mModelDir + "/slim_word_embeddings_bf16.bin";
    FILE* file = fopen(file_path.c_str(), "rb");
    std::unique_ptr<int16_t[]> buffer(new int16_t[HIDDEN_SIZE]);
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(buffer.get(), 1, size, file);
        auto ptr = embedding_var->writeMap<int16_t>() + i * HIDDEN_SIZE * 2;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            ptr[j * 2] = 0;
            ptr[j * 2 + 1] = buffer[j];
        }
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
        ptr[i] = 1;
    }
    if (seq_len > 1) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                ptr[seq_len * i + j] = j <= i;
            }
        }
    }
    return attention_mask_var;
}

VARP ChatGLM::gen_position_ids(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    // position_ids
#ifdef CHATHLM_1
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
#else
    auto position_ids_var = _Input({seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids_var->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = mSeqLen - 1;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
#endif
    return position_ids_var;
}

int ChatGLM::forward(const std::vector<int>& input_ids) {
    // AUTOTIME;
    mSeqLen += input_ids.size();
    auto hidden_states = gen_embedding(input_ids);
    auto ptr = hidden_states->readMap<float>();
// #define DEBUG_DUMP
#ifdef DEBUG_DUMP
    printf("embeding : [\n");
    for (int n = 0; n < 5; n++) {
        for (int i = 0; i < 3; i++) {
            printf("%f, ", ptr[n * 4096 + i]);
        }
        printf(" ..., ");
        for (int i = 4093; i < 4096; i++) {
            printf("%f, ", ptr[n * 4096 + i]);
        }
        printf("\n");
    }
    printf("]\n");
#endif
    auto attention_mask = gen_attention_mask(input_ids);
    auto position_ids = gen_position_ids(input_ids);
    for (int i = 0; i < LAYER_SIZE; i++) {
        // AUTOTIME;
        // auto outputs = mModules[i]->onForward({hidden_states, attention_mask, position_ids, mHistoryVars[i]});
        auto outputs = mModules[i]->onForward({attention_mask, hidden_states, position_ids, mHistoryVars[i]});
// #define DEBUG_DUMP
#ifdef DEBUG_DUMP
        if (mHistoryVars[i]->getInfo()->size > 1 && 0) {
            auto kv = outputs[1];
            auto size = kv->getInfo()->size;
            ptr = kv->readMap<float>();
            printf("[%d] pastkv : [\n", i);
            for (int i = 0; i < 3; i++) {
                printf("%f, ", ptr[i]);
            }
            printf(" ..., ");
            for (int i = size - 3; i < size; i++) {
                printf("%f, ", ptr[i]);
            }
            printf("]\n");
        }
#endif
        hidden_states = outputs[0];
#ifdef DEBUG_DUMP
        if (mHistoryVars[i]->getInfo()->size > 1) {
            ptr = hidden_states->readMap<float>();
            printf("[%d] hidden_states : [\n", i);
            for (int n = 0; n < 1; n++) {
                for (int i = 0; i < 3; i++) {
                    printf("%f, ", ptr[n * 4096 + i]);
                }
                printf(" ..., ");
                for (int i = 4093; i < 4096; i++) {
                    printf("%f, ", ptr[n * 4096 + i]);
                }
                printf("\n");
            }
            printf("...\n");
            for (int n = input_ids.size() - 3; n < input_ids.size(); n++) {
                for (int i = 0; i < 3; i++) {
                    printf("%f, ", ptr[n * 4096 + i]);
                }
                printf(" ..., ");
                for (int i = 4093; i < 4096; i++) {
                    printf("%f, ", ptr[n * 4096 + i]);
                }
                printf("\n");
            }
            printf("]\n");
            // exit(0);
        }
#endif
        hidden_states = outputs[0];
        mHistoryVars[i] = outputs[1];
    }
    return var_to_token(hidden_states);
}

int ChatGLM::var_to_token(VARP var) {
    // AUTOTIME;
    var = _Reshape(var, {-1, HIDDEN_SIZE});
    int num = var->getInfo()->dim[0];
    if (num > 1) {
        var = _Gather(var, _Scalar<int>(num - 1));
    }
    var = _Reshape(var, {1, HIDDEN_SIZE, 1, 1});
    if (0) {
        auto ptr = var->readMap<float>();
        printf("hidden_states : [\n");
        for (int n = 0; n < 1; n++) {
            for (int i = 0; i < 3; i++) {
                printf("%f, ", ptr[n * 4096 + i]);
            }
            printf(" ..., ");
            for (int i = 4093; i < 4096; i++) {
                printf("%f, ", ptr[n * 4096 + i]);
            }
            printf("\n");
        }
        printf("]\n");
    }
    auto outputs = mModules.back()->onForward({var});
    int id = outputs[0]->readMap<int>()[0];
    return id;
}

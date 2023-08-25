//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#include <fstream>
#include <iostream>

#include "llm.hpp"
#include "cppjieba/Jieba.hpp"
#include <MNN/expr/ExecutorScope.hpp>

// base64
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


std::string Llm::response(const std::string& query, std::ostream* os) {
    // init status
    for (int i = 0; i < layer_nums_; i++) {
        past_key_values_.push_back(_Input(key_value_shape_, NCHW));
    }
    // response
    auto st = std::chrono::system_clock::now();
    auto input_ids = tokenizer(query);
    /*
    printf("input_ids = { ");
    for (auto id : input_ids) {
        printf("%d, ", id);
    }
    printf(" }\n");
    */
    int token = forward(input_ids);
    std::string output_str = decode(token);
    *os << output_str << std::flush;
    while (true) {
        token = forward({token});
        if (is_stop(token)) {
            *os << std::endl << std::flush;
            break;
        }
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
    auto et = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(et - st);
    printf("\n[speed: %f tok/s]\n", gen_seq_len_ / (duration.count() * 1e-6));
    return output_str;
}

void Llm::load(const std::string& model_dir, const std::string& tokenizer_dir) {
    model_dir_ = model_dir;
    tokenizer_dir_ = tokenizer_dir;
    // init
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type          = MNN_FORWARD_CPU;
    config.numThread     = 4;
    cpuBackendConfig.precision = BackendConfig::Precision_Low;
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
    config.backendConfig = &cpuBackendConfig;
    cpu_runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    // 1. load vocab
    {
        std::string vocab_path = tokenizer_dir + "/" + model_name_ + "_vocab.txt";
        printf("load %s ... ", vocab_path.c_str());
        std::ifstream vocab_file(vocab_path);
        int index = 0;
        std::string word;
        while (vocab_file >> word) {
            word = base64_decode(word);
            word_decoder_.push_back(word);
            word_encoder_.insert(std::make_pair<std::string, int>(std::move(word), index++));
        }
        printf("Done!\n");
    }
    // 2. load models
    modules_.resize(layer_nums_ + 1);
    char buffer[50];
    // load lm model
    std::string lm_model_path = model_dir + "/lm.mnn";
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;
    float load_progress = 0.f;
    printf("[%3.0f%% ] load %s model ... ", load_progress, lm_model_path.c_str());
    modules_[layer_nums_].reset(Module::load({}, {}, lm_model_path.c_str(), cpu_runtime_manager_, &module_config));
    printf("Done!\n");
    float step = 100.0 / (layer_nums_ + 1);
    // load glm_block models
    for (int i = 0; i < layer_nums_; i++) {
        load_progress += step;
        std::string model_path = model_dir + "/glm_block_" + std::to_string(i) + ".mnn";
        printf("[%3.0f%% ] load %s model ... ", load_progress, model_path.c_str());
        modules_[i].reset(Module::load(
            {"inputs_embeds", "attention_mask", "position_ids", "past_key_values"},
            {"hidden_states", "presents"}, model_path.c_str(), cpu_runtime_manager_, &module_config));
        printf("Done!\n");
        fflush(stdout);
    }
}

int Llm::forward(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    auto hidden_states = gen_embedding(input_ids);
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    for (int i = 0; i < layer_nums_; i++) {
        auto outputs = modules_[i]->onForward({hidden_states, attention_mask, position_ids, past_key_values_[i]});
        hidden_states = outputs[0];
        past_key_values_[i] = outputs[1];
    }
    // TODO: merge to module
    {
        hidden_states = _Reshape(hidden_states, {-1, hidden_size_});
        int num = hidden_states->getInfo()->dim[0];
        if (num > 1) {
            hidden_states = _Gather(hidden_states, _Scalar<int>(num - 1));
        }
        // hidden_states = _Reshape(hidden_states, {1, hidden_size_, 1, 1});
        hidden_states = _Reshape(hidden_states, {1, hidden_size_});
    }
    auto outputs = modules_.back()->onForward({hidden_states});
    int id = outputs[0]->readMap<int>()[0];
    all_seq_len_ += seq_len;
    gen_seq_len_++; 
    return id;
}

VARP Llm::gen_embedding(const std::vector<int>& input_ids) {
    size_t seq_len = input_ids.size();
    auto embedding = _Input({static_cast<int>(seq_len), 1, hidden_size_}, NCHW);
    size_t size = hidden_size_ * sizeof(int16_t);
    std::string file_path = model_dir_ + "/slim_word_embeddings_bf16.bin";
    FILE* file = fopen(file_path.c_str(), "rb");
    std::unique_ptr<int16_t[]> buffer(new int16_t[hidden_size_]);
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(buffer.get(), 1, size, file);
        auto ptr = embedding->writeMap<int16_t>() + i * hidden_size_ * 2;
        for (int j = 0; j < hidden_size_; j++) {
            ptr[j * 2] = 0;
            ptr[j * 2 + 1] = buffer[j];
        }
    }
    fclose(file);
    return embedding;
}

std::vector<int> Llm::tokenizer_encode(std::string input_str) {
    std::vector<int> ids;
    std::vector<std::string> words;
    std::string dict_path = tokenizer_dir_ + "/jieba.dict.utf8";
    std::string model_path = tokenizer_dir_ + "/hmm_model.utf8";
    std::string user_dict_path = tokenizer_dir_ + "/user.dict.utf8";
    std::string idf_path = tokenizer_dir_ + "/idf.utf8";
    std::string stopWord_path = tokenizer_dir_ + "/stop_words.utf8";
    cppjieba::Jieba jieba(
        dict_path,
        model_path,
        user_dict_path,
        idf_path,
        stopWord_path
    );
    jieba.Cut(input_str, words, true);
    for (auto word : words) {
        const auto& iter = word_encoder_.find(word);
        if (iter != word_encoder_.end()) {
            ids.push_back(iter->second);
        }
    }
    return ids;
}

std::string Llm::decode(int id) {
    auto word = word_decoder_[id];
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length()-1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word = static_cast<char>(num);
    }
    return word;
}

// Chatglm_6b
std::vector<int> Chatglm_6b::tokenizer(const std::string& query) {
    auto ids = tokenizer_encode(query);
    context_len_ = ids.size();
    ids.push_back(130001);
    ids.push_back(130004);
    return ids;
}

VARP Chatglm_6b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len * seq_len; i++) {
        ptr[i] = 0;
    }
    if (seq_len > 1) {
        for (int i = 1; i < seq_len; i++) {
            ptr[seq_len * i - 1] = 1;
        }
    }
    return attention_mask;
}

VARP Chatglm_6b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = 1;
        ptr[1] = all_seq_len_ - context_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
            ptr[seq_len + i] = 0;
        }
        ptr[2 * seq_len - 1] = 1;
    }
    return position_ids;
}

bool Chatglm_6b::is_stop(int token_id) {
    return token_id == 130005;
}

// Chatglm2_6b
std::vector<int> Chatglm2_6b::tokenizer(const std::string& query) {
    auto ids = tokenizer_encode(query);
    ids.insert(ids.begin(), 64792);
    ids.insert(ids.begin(), 64790);
    return ids;
}

VARP Chatglm2_6b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    if (seq_len > 1) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                ptr[seq_len * i + j] = j > i;
            }
        }
    } else {
        ptr[0] = 0;
    }
    return attention_mask;
}

VARP Chatglm2_6b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = gen_seq_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
    return position_ids;
}

bool Chatglm2_6b::is_stop(int token_id) {
    return token_id <= 2;
}

// Qwen_7b
std::vector<int> Qwen_7b::tokenizer(const std::string& query) {
    auto prompt = "\n<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n";
    auto ids = tokenizer_encode(prompt);
    return ids;
}

VARP Qwen_7b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            ptr[seq_len * i + j] = j <= i;
        }
    }
    return attention_mask;
}

VARP Qwen_7b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = all_seq_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
    return position_ids;
}

bool Qwen_7b::is_stop(int token_id) {
    return token_id >= 151645;
}
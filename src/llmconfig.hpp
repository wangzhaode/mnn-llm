//
//  llmconfig.hpp
//
//  Created by MNN on 2024/07/19.
//  ZhaodeWang
//
#include "llm.hpp"

static inline bool has_suffix(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static inline std::string base_dir(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "./";
    } else {
        return path.substr(0, pos + 1);
    }
}

static inline std::string file_name(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

class LlmConfig {
public:
    std::string base_dir_;
    json config_, llm_config_;
    LlmConfig() {}
    LlmConfig(const std::string& path) {
        // load config
        if (has_suffix(path, ".json")) {
            std::ifstream config_file(path);
            if (config_file.is_open()) {
                config_ = json::parse(config_file);
            } else {
                std::cerr << "Unable to open config file: " << path << std::endl;
            }
            base_dir_ = base_dir(path);
        } else {
            // compatibility with the original usage
            if (has_suffix(path, ".mnn")) {
                auto model_name = file_name(path);
                config_ = {
                    {"llm_model", model_name},
                    {"llm_weight", model_name + ".weight"}
                };
                base_dir_ = base_dir(path);
            } else {
                config_ = {};
                base_dir_ = path;
            }
        }
        // using config's base_dir
        base_dir_ = config_.value("base_dir", base_dir_);
        // load llm_config for model info
        std::ifstream llm_config_file(llm_config());
        if (llm_config_file.is_open()) {
            llm_config_ = json::parse(llm_config_file);
        } else {
            std::cerr << "Unable to open llm_config file: " << llm_config() << std::endl;
        }
    }

    #define DEFINE_CONFIG_PATH_ACCESSOR(name, defaultValue) \
    std::string name() const { return base_dir_ + config_.value(#name, defaultValue); }

    #define DEFINE_CONFIG_ACCESSOR(name, type, defaultValue) \
    type name() const { return config_.value(#name, defaultValue); }

    #define DEFINE_LLM_CONFIG_ACCESSOR(name, type, defaultValue) \
    type name() const { return llm_config_.value(#name, defaultValue); }

    // < model file config start
    DEFINE_CONFIG_PATH_ACCESSOR(llm_config, "llm_config.json")
    DEFINE_CONFIG_PATH_ACCESSOR(llm_model, "llm.mnn")
    DEFINE_CONFIG_PATH_ACCESSOR(llm_weight, "llm.mnn.weight")
    DEFINE_CONFIG_PATH_ACCESSOR(lm_model, "lm.mnn")
    DEFINE_CONFIG_PATH_ACCESSOR(embedding_model, "embedding.mnn")
    DEFINE_CONFIG_PATH_ACCESSOR(embedding_file, "embeddings_bf16.bin")
    DEFINE_CONFIG_PATH_ACCESSOR(tokenizer_file, "tokenizer.txt")
    DEFINE_CONFIG_PATH_ACCESSOR(visual_model, "visual.mnn")
    // model file config end >

    // < generate config start
    DEFINE_CONFIG_ACCESSOR(max_new_tokens, int, 512)
    DEFINE_CONFIG_ACCESSOR(reuse_kv, bool, false)
    DEFINE_CONFIG_ACCESSOR(backend_type, std::string, "cpu")
    DEFINE_CONFIG_ACCESSOR(thread_num, int, 4)
    DEFINE_CONFIG_ACCESSOR(precision, std::string, "low")
    DEFINE_CONFIG_ACCESSOR(power, std::string, "normal")
    DEFINE_CONFIG_ACCESSOR(memory, std::string, "low")
    DEFINE_CONFIG_ACCESSOR(quant_qkv, int, 0)
    DEFINE_CONFIG_ACCESSOR(kvcache_limit, int, -1)
    DEFINE_CONFIG_ACCESSOR(use_mmap, bool, false)
    DEFINE_CONFIG_ACCESSOR(kvcache_mmap, bool, false)
    DEFINE_CONFIG_ACCESSOR(tmp_path, std::string, "")
    // generate config end >

    // < llm model config start
    DEFINE_LLM_CONFIG_ACCESSOR(is_single, bool, true)
    DEFINE_LLM_CONFIG_ACCESSOR(is_visual, bool, false)
    DEFINE_LLM_CONFIG_ACCESSOR(hidden_size, int, 4096)
    DEFINE_LLM_CONFIG_ACCESSOR(layer_nums, int, 32)
    DEFINE_LLM_CONFIG_ACCESSOR(key_value_shape, std::vector<int>, std::vector<int>{})
    DEFINE_LLM_CONFIG_ACCESSOR(attention_mask, std::string, "int")
    DEFINE_LLM_CONFIG_ACCESSOR(attention_fused, bool, true)
    DEFINE_LLM_CONFIG_ACCESSOR(chat_template, std::string, "")
    DEFINE_LLM_CONFIG_ACCESSOR(prompt_template, std::string, "")
    // llm model config end >
};
//
//  llm.hpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#ifndef LLM_hpp
#define LLM_hpp

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <functional>
#include <unordered_map>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "tokenizer.hpp"
#include "json.hpp"

using namespace MNN;
using namespace Express;
using json = nlohmann::json;
class Tokenizer;
class Pipeline;

// Llm start
// llm stream buffer with callback
class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    LlmStreamBuffer(CallBack callback) : callback_(callback) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }

private:
    CallBack callback_ = nullptr;
};

enum PROMPT_TYPE {
    SYSTEM = 0,
    ATTACHMENT = 1,
    USER = 2,
    ASSISTANT = 3,
    OTHER = 4
};

struct Prompt {
    PROMPT_TYPE type;
    std::string str;
    std::vector<int> tokens;
};

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

    // < model file config start
    std::string llm_config() const {
        return base_dir_ + config_.value("llm_config", "llm_config.json");
    }

    std::string llm_model() const {
        return base_dir_ + config_.value("llm_model", "llm.mnn");
    }

    std::string llm_weight() const {
        return base_dir_ + config_.value("llm_weight", "llm.mnn.weight");
    }

    std::string block_model(int index) const {
        return base_dir_ + config_.value("block_model", "block_") + std::to_string(index) + ".mnn";
    }

    std::string lm_model() const {
        return base_dir_ + config_.value("lm_model", "lm.mnn");
    }

    std::string embedding_model() const {
        return base_dir_ + config_.value("embedding_model", "embedding.mnn");
    }

    std::string embedding_file() const {
        return base_dir_ + config_.value("embedding_file", "embeddings_bf16.bin");
    }

    std::string tokenizer_file() const {
        return base_dir_ + config_.value("tokenizer_file", "tokenizer.txt");
    }

    std::string visual_model() const {
        return base_dir_ + config_.value("visual_model", "visual.mnn");
    }
    // model file config end >

    // < generate config start
    int max_new_tokens() const {
        return config_.value("max_new_tokens", 512);
    }
    // generate config end >

    // < backend config start
    std::string backend_type() const {
        return config_.value("backend_type", "cpu");
    }

    int thread_num() const {
        return config_.value("thread_num", 4);
    }

    std::string precision() const {
        return config_.value("precision", "low");
    }

    std::string memory() const {
        return config_.value("memory", "low");
    }
    // backend config end >

    // < llm model config start
    bool is_single() const {
        return llm_config_.value("is_single", true);
    }

    bool is_visual() const {
        return llm_config_.value("is_visual", false);
    }

    int hidden_size() const {
        return llm_config_.value("hidden_size", 4096);
    }

    int layer_nums() const {
        return llm_config_.value("layer_nums", 32);
    }

    std::vector<int> key_value_shape() const {
        return llm_config_.value("key_value_shape", std::vector<int>{});
    }

    std::string attention_mask() const {
        return llm_config_.value("attention_mask", "int");
    }

    std::string prompt_template() const {
        return llm_config_.value("prompt_template", "");
    }
    // llm model config end >
};

class Llm {
public:
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm() {
        modules_.clear();
        runtime_manager_.reset();
    }
    void chat();
    static Llm* createLLM(const std::string& config_path);
    virtual void load();
    VARP forward(const std::vector<int>& input_ids);
    int sample(VARP logits, const std::vector<int>& pre_ids);
    std::string apply_chat_template(const std::string& input_str) const;
    std::string response(const std::string& input_str, std::ostream* os = &std::cout, const char* end_with = nullptr);
    void generate_init();
    std::string generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    void print_speed();
    friend class Pipeline;
public:
    // forward info
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
    float load_progress_ = 0.f;
    bool is_single_ = true;
    bool is_disk_embedding_ = true;
    std::shared_ptr<LlmConfig> config_;
    std::unique_ptr<Tokenizer> tokenizer_;
protected:
    std::vector<int> key_value_shape_ = {};
    std::vector<VARP> past_key_values_;
    VARP inputs_embeds_, attention_mask_, position_ids_;
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> modules_;
    void init_runtime();
    std::string decode(int id);
    bool is_stop(int token_id);
    virtual std::vector<int> tokenizer(const std::string& query);
    virtual VARP embedding(const std::vector<int>& input_ids);
    virtual VARP gen_attention_mask(int seq_len);
    virtual VARP gen_position_ids(int seq_len);
};

class Lvlm : public Llm {
public:
    Lvlm(std::shared_ptr<LlmConfig> config) : Llm(config) {
        img_size_ = config->llm_config_.value("img_size", img_size_);
        imgpad_len_ = config->llm_config_.value("imgpad_len", imgpad_len_);
        img_start_ = config->llm_config_.value("img_start", img_start_);
        img_end_ = config->llm_config_.value("img_end", img_end_);
        img_pad_ = config->llm_config_.value("img_pad", img_pad_);
    }
    ~Lvlm() { visual_module_.reset(); }
    virtual void load() override;
private:
    int img_size_ = 448, imgpad_len_ = 256, img_start_ = 151857, img_end_ = 151858, img_pad_ = 151859;
    std::shared_ptr<Module> visual_module_;
    VARP visual_embedding(const std::vector<int>& input_ids);
    std::vector<int> url_encode(const std::string& url);
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP embedding(const std::vector<int>& input_ids) override;
};
// Llm end

// Embedding start
class Embedding : public Llm {
public:
    Embedding(std::shared_ptr<LlmConfig> config) : Llm(config) {}
    static Embedding* createEmbedding(const std::string& config_path);
    static float dist(VARP var0, VARP var1);
    virtual void load() override;
    VARP embedding(const std::string& txt);
    int dim() { return config_->hidden_size(); }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
};
// Embedding end

// TextVectorStore strat
class TextVectorStore {
public:
    TextVectorStore() : embedding_(nullptr) {}
    TextVectorStore(std::shared_ptr<Embedding> embedding) : embedding_(embedding) {}
    ~TextVectorStore() {}
    static TextVectorStore* load(const std::string& path, const std::string& embedding_path = "");
    void set_embedding(std::shared_ptr<Embedding> embedding) {
        embedding_ = embedding;
    }
    void save(const std::string& path);
    void add_text(const std::string& text);
    void add_texts(const std::vector<std::string>& texts);
    std::vector<std::string> search_similar_texts(const std::string& txt, int topk = 1);
    void bench();
protected:
    inline VARP text2vector(const std::string& text);
// private:
public:
    std::shared_ptr<Embedding> embedding_;
    VARP vectors_;
    std::vector<std::string> texts_;
    int dim_ = 1024;
};
// TextVectorStore end

// Document start
class Document {
public:
    enum DOCTYPE {
        AUTO = 0,
        TXT  = 1,
        MD   = 2,
        HTML = 3,
        PDF  = 4
    };
    Document(const std::string& path, DOCTYPE type = AUTO) : path_(path), type_(type) {}
    ~Document() = default;
    std::vector<std::string> split(int chunk_size = -1);
private:
    DOCTYPE type_;
    std::string path_;
    std::vector<std::string> load_txt();
    std::vector<std::string> load_pdf();
};
// Document end

// MemoryBase start
class MemoryBase {
public:
    MemoryBase() {}
    virtual ~MemoryBase() {}
    void set_embedding(std::shared_ptr<Embedding> embedding) {
        store_->set_embedding(embedding);
    }
    virtual std::vector<std::string> search(const std::string& query, int topk);
    virtual void save(const std::string& path) = 0;
    virtual void build_vectors() = 0;
protected:
    void load_store(const std::string& path);
    void save_store(const std::string& path);
public:
    std::shared_ptr<TextVectorStore> store_;
};

class ChatMemory : public MemoryBase {
public:
    ChatMemory() {}
    ~ChatMemory() override {}
    static ChatMemory* load(const std::string& path);
    void save(const std::string& path) override;
    void build_vectors() override;
    std::string get_latest(std::string key);
    void add(const std::vector<Prompt>& prompts);
    void summarize(std::shared_ptr<Llm> llm);
private:
    json memory_;
};

class Knowledge : public MemoryBase {
public:
    Knowledge() {}
    ~Knowledge() override {}
    static Knowledge* load(const std::string& path);
    void save(const std::string& path) override;
    void build_vectors() override;
private:
    std::unique_ptr<Document> document_;
};
// MemoryBase end

// Pipeline start
class Pipeline {
public:
    Pipeline() {}
    ~Pipeline() {}
    static Pipeline* load(const std::string& path);
    void invoke(const std::string& str);
private:
    bool need_memory(const std::string& str);
    bool need_knowledge(const std::string& str);
    std::string build_prompt(const std::string& str);
    std::unique_ptr<Llm> llm_;
    std::shared_ptr<Embedding> embedding_;
    std::unique_ptr<Knowledge> knowledge_;
    std::unique_ptr<ChatMemory> memory_;
    std::string system_, user_, assistant_;
    std::vector<Prompt> prompts_;
    json config_;
};
// Pipeline end

#endif // LLM_hpp

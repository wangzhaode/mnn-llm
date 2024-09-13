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
class LlmConfig;

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

class Llm {
public:
    using PromptItem = std::pair<std::string, std::string>; // <role, content>
    Llm(std::shared_ptr<LlmConfig> config) : config_(config) {}
    virtual ~Llm();
    void chat();
    void reset();
    static Llm* createLLM(const std::string& config_path);
    virtual void load();
    VARP forward(const std::vector<int>& input_ids);
    int sample(VARP logits, const std::vector<int>& pre_ids);
    std::string apply_prompt_template(const std::string& user_content) const;
    std::string apply_chat_template(const std::vector<PromptItem>& chat_prompts) const;
    std::string response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr);
    std::string response(const std::vector<PromptItem>& chat_prompts, std::ostream* os = &std::cout, const char* end_with = nullptr);
    void generate_init();
    std::string generate(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    std::vector<int> generate(const std::vector<int>& input_ids, int max_new_tokens = -1);
    void print_speed();
    // config function
    std::string dump_config();
    bool set_config(const std::string& content);
    // lora function
    size_t apply_lora(const std::string& lora_path);
    Llm* create_lora(const std::string& lora_path);
    bool release_module(size_t index);
    bool select_module(size_t index);
    friend class Pipeline;
public:
    // forward info
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    std::vector<int> history_ids_;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
    float load_progress_ = 0.f;
    bool is_single_ = true;
    bool attention_fused_ = true;
protected:
    std::shared_ptr<LlmConfig> config_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::vector<int> key_value_shape_ = {};
    std::vector<MNN::Express::VARP> past_key_values_;
    MNN::Express::VARP inputs_embeds_, attention_mask_, position_ids_;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<MNN::Express::Module>> modules_;
    std::vector<std::shared_ptr<MNN::Express::Module>> prefill_modules_, decode_modules_, current_modules_;
    const MNN::Express::Module* base_module_ = nullptr;
    void init_runtime();
    std::string decode(int id);
    bool is_stop(int token_id);
    virtual std::vector<int> tokenizer(const std::string& query);
    virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids);
    virtual MNN::Express::VARP gen_attention_mask(int seq_len);
    virtual MNN::Express::VARP gen_position_ids(int seq_len);
};
// Llm end

// Embedding start
class Embedding : public Llm {
public:
    Embedding(std::shared_ptr<LlmConfig> config);
    static Embedding* createEmbedding(const std::string& config_path, bool load = true);
    static float dist(MNN::Express::VARP var0, MNN::Express::VARP var1);
    virtual void load() override;
    MNN::Express::VARP ids_embedding(const std::vector<int>& ids);
    MNN::Express::VARP txt_embedding(const std::string& txt);
    int dim() const;
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual MNN::Express::VARP gen_attention_mask(int seq_len) override;
    virtual MNN::Express::VARP gen_position_ids(int seq_len) override;
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

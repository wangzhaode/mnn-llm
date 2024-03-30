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

class Llm {
public:
    Llm() {
        // default tokenier is senrencepiece
        tokenizer_.reset(new Sentencepiece);
    }
    virtual ~Llm() {
        modules_.clear();
        visual_module_.reset();
        runtime_manager_.reset();
    }
    static Llm* createLLM(const std::string& path, std::string model_type = "auto");
    void load(const std::string& model_dir);
    void chat();
    void warmup();
    std::string response(const std::string& input_str, std::ostream* os = &std::cout, const char* end_with = nullptr);
    std::string response_nohistory(const std::string& input_str, std::ostream* os = &std::cout, const char* end_with = nullptr);
    float load_progress() { return load_progress_; }
    void reset();
    void print_speed();
    friend class Pipeline;
public:
    std::vector<int> history_;
    // forward info
    int max_seq_len_ = 1024;
    int prompt_len_ = 0;
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    // time
    int64_t prefill_us_ = 0;
    int64_t decode_us_ = 0;
protected:
    void response_init();
    std::string response_impl(const std::vector<int>& input_ids, std::ostream* os, const char* end_with);
    VARP embedding(const std::vector<int>& input_ids);
    VARP txt_embedding(const std::vector<int>& input_ids);
    int forward(const std::vector<int>& input_ids);
    std::vector<int> tokenizer_encode(const std::string& input_str);
    std::string decode(int id);
protected:
    VARP inputs_embeds_, attention_mask_, position_ids_;
    // model configs
    bool is_single_ = false;
    bool is_disk_embedding_ = false;
    bool is_visual_ = false;
    int layer_nums_ = 0;
    int hidden_size_ = 4096;
    std::vector<int> key_value_shape_ = {};
    std::string model_name_ = "";
    std::string disk_embedding_file_ = "";
    // gen info
    float load_progress_ = 0.f;
    // tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;
    std::shared_ptr<Module> visual_module_;
private:
    virtual VARP visual_embedding(const std::vector<int>& input_ids) { return nullptr; }
    virtual std::vector<int> tokenizer(const std::string& query) = 0;
    virtual VARP gen_attention_mask(int seq_len) = 0;
    virtual VARP gen_position_ids(int seq_len) = 0;
    virtual bool is_stop(int token_id) = 0;
private:
    // MNN Modules
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> modules_;
    std::vector<VARP> past_key_values_;
    // model dir
    std::string model_dir_;
};

// some llm models
class Chatglm_6b : public Llm {
public:
    Chatglm_6b() {
        model_name_ = "Chatglm_6b";
        layer_nums_ = 28;
        key_value_shape_ = {2, 0, 1, 32, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
    int context_len_ = 0;
};

class Chatglm2_6b : public Llm {
public:
    Chatglm2_6b() {
        model_name_ = "Chatglm2_6b";
        layer_nums_ = 28;
        key_value_shape_ = {2, 0, 1, 2, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
};

class Phi_2 : public Chatglm2_6b {
public:
    Phi_2() {
        model_name_ = "Phi_2";
        layer_nums_ = 32;
        key_value_shape_ = {1, 0, 2, 32, 80};
        hidden_size_ = 2560;
        tokenizer_.reset(new Tiktoken);
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual bool is_stop(int token_id) override;
};

class Qwen_7b : public Llm {
public:
    Qwen_7b() {
        model_name_ = "Qwen_7b";
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 0, 32, 128};
        hidden_size_ = 4096;
        tokenizer_.reset(new Tiktoken);
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
};

class Qwen_vl : public Qwen_7b {
public:
    Qwen_vl() {
        model_name_ = "Qwen_vl";
        is_visual_ = true;
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 0, 32, 128};
        hidden_size_ = 4096;
        tokenizer_.reset(new Tiktoken);
    }
private:
    const int img_size_ = 448;
    const int imgpad_len_ = 256;
    const int img_start_ = 151857;
    const int img_end_ = 151858;
    const int img_pad_ = 151859;
private:
    std::vector<int> url_encode(const std::string& url);
    virtual VARP visual_embedding(const std::vector<int>& input_ids) override;
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
};

class Qwen_1_8b : public Qwen_7b {
public:
    Qwen_1_8b() {
        model_name_ = "Qwen_1.8b";
        layer_nums_ = 24;
        key_value_shape_ = {2, 1, 0, 16, 128};
        hidden_size_ = 2048;
        tokenizer_.reset(new Tiktoken);
    }
};

class Llama2_7b : public Llm {
public:
    Llama2_7b() {
        model_name_ = "Llama2_7b";
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 32, 0, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
};

class Qwen2 : public Llama2_7b {
public:
    Qwen2() {
        model_name_ = "Qwen2";
        tokenizer_.reset(new HuggingfaceTokenizer);
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual bool is_stop(int token_id) override;
};

class Qwen2_0_5b : public Qwen2 {
public:
    Qwen2_0_5b() {
        model_name_ = "Qwen2_0.5b";
        layer_nums_ = 24;
        key_value_shape_ = {2, 1, 16, 0, 64};
        hidden_size_ = 1024;
    }
};

class Qwen2_1_8b : public Qwen2 {
public:
    Qwen2_1_8b() {
        model_name_ = "Qwen2_1.8b";
        layer_nums_ = 24;
        key_value_shape_ = {2, 1, 16, 0, 128};
        hidden_size_ = 2048;
    }
};

class Qwen2_4b : public Qwen2 {
public:
    Qwen2_4b() {
        model_name_ = "Qwen2_4b";
        layer_nums_ = 40;
        key_value_shape_ = {2, 1, 20, 0, 128};
        hidden_size_ = 2560;
    }
};

class Qwen2_7b : public Qwen2 {
public:
    Qwen2_7b() {
        model_name_ = "Qwen2_7b";
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 32, 0, 128};
        hidden_size_ = 4096;
    }
};

class TinyLlama : public Llama2_7b {
public:
    TinyLlama() {
        model_name_ = "TinyLlama";
        layer_nums_ = 22;
        key_value_shape_ = {2, 1, 4, 0, 64};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
};

class Yi_6b : public Llama2_7b {
public:
    Yi_6b() {
        model_name_ = "Yi_6b";
        key_value_shape_ = {2, 1, 4, 0, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual bool is_stop(int token_id) override;
};
// Llm end

// Embedding start
class Embedding {
public:
    Embedding() {
        // default tokenier is Bert
        tokenizer_.reset(new BertTokenizer);
    }
    virtual ~Embedding() {
        module_.reset();
        runtime_manager_.reset();
    }
    static Embedding* createEmbedding(const std::string& path, std::string model_type = "auto");
    static float dist(VARP var0, VARP var1);
    void load(const std::string& model_dir);
    VARP embedding(const std::string& txt);
    void print_speed();
    int dim() { return hidden_size_; }
public:
    // time
    int64_t embedding_us_ = 0;
    int prompt_len_ = 0;
protected:
    std::vector<int> tokenizer_encode(const std::string& input_str);
protected:
    // model configs
    int layer_nums_ = 0;
    int hidden_size_ = 1024;
    std::string model_name_ = "";
    // tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;
private:
    virtual std::vector<int> tokenizer(const std::string& query) = 0;
    virtual VARP gen_attention_mask(int seq_len) = 0;
    virtual VARP gen_position_ids(int seq_len) = 0;
private:
    // MNN Modules
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::shared_ptr<Module> module_;
    // model dir
    std::string model_dir_;
};

// some embedding models
class Bge : public Embedding {
public:
    Bge() {
        model_name_ = "Bge";
        layer_nums_ = 24;
        hidden_size_ = 1024;
    }
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

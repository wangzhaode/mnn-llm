# C++ API

## Tokenizer 类

### 描述
`Tokenizer` 类负责文本的编码和解码，支持不同类型的分词器。

### 常量
#### `static constexpr int MAGIC_NUMBER`
- **描述：** 魔法数字，用于识别分词器类型。

### 枚举类型
#### `enum TokenizerType`
- `SENTENCEPIECE`：SentencePiece 分词器。
- `TIKTOIKEN`：TikToken 分词器。
- `BERT`：BERT 分词器。
- `HUGGINGFACE`：Hugging Face 分词器。

### 构造函数
#### `Tokenizer()`
- **描述：** 默认构造函数。

### 析构函数
#### `virtual ~Tokenizer()`
- **描述：** 默认析构函数。

### 公共方法
#### `static Tokenizer* createTokenizer(const std::string& filename)`
- **描述：** 根据给定文件名创建相应类型的分词器实例。
- **参数：**
  - `filename`：分词器配置文件路径。
- **返回：** 创建的 `Tokenizer` 实例指针。

#### `bool is_stop(int token)`
- **描述：** 判断给定的 token 是否为停止 token。
- **参数：**
  - `token`：待检查的 token。
- **返回：** 如果是停止 token，则返回 `true`；否则返回 `false`。

#### `bool is_special(int token)`
- **描述：** 判断给定的 token 是否为特殊 token。
- **参数：**
  - `token`：待检查的 token。
- **返回：** 如果是特殊 token，则返回 `true`；否则返回 `false`。

#### `std::vector<int> encode(const std::string& str)`
- **描述：** 将输入字符串编码为 token 数组。

---

## LlmStreamBuffer 类

### 描述
`LlmStreamBuffer` 是一个自定义流缓冲区，允许通过回调机制进行数据流传输。

### 构造函数
#### `LlmStreamBuffer(CallBack callback)`
- **参数：**
  - `callback`：用于处理流数据的函数。

### 保护方法
#### `std::streamsize xsputn(const char* s, std::streamsize n) override`
- **描述：** 将 `n` 个字符从 `s` 指向的数组写入。
- **返回：** 写入的字符数量。

### 私有成员
- `CallBack callback_`：存储回调函数。

---

## PROMPT_TYPE 枚举

### 描述
定义了可以在 `Llm` 类中使用的提示类型。

- `SYSTEM`：系统提示。
- `ATTACHMENT`：附件提示。
- `USER`：用户提示。
- `ASSISTANT`：助手提示。
- `OTHER`：其他提示类型。

---

## Prompt 结构体

### 描述
表示一个提示，包含其类型、字符串内容及相关标记。

### 成员
- `PROMPT_TYPE type`：提示类型。
- `std::string str`：提示内容。
- `std::vector<int> tokens`：与提示相关的标记。

---

## Llm 类

### 描述
`Llm` 类封装了一个大型语言模型，用于交互和处理提示。

### 构造函数
#### `Llm(std::shared_ptr<LlmConfig> config)`
- **参数：**
  - `config`：指向配置的共享指针。

### 析构函数
#### `~Llm()`
- **描述：** 清理资源。

### 公共方法
#### `void chat()`
- **描述：** 启动聊天会话。

#### `void reset()`
- **描述：** 重置模型状态。

#### `static Llm* createLLM(const std::string& config_path)`
- **描述：** 工厂方法，创建 `Llm` 实例。
- **返回：** 创建的 `Llm` 实例的指针。

#### `void load()`
- **描述：** 加载模型配置。

#### `VARP forward(const std::vector<int>& input_ids)`
- **描述：** 处理输入标记并返回输出变量。

#### `int sample(VARP logits, const std::vector<int>& pre_ids)`
- **描述：** 根据提供的 ID 从 logits 中抽样。
- **返回：** 抽样的整数。

#### `std::string apply_prompt_template(const std::string& user_content) const`
- **描述：** 将提示模板应用于用户内容。

#### `std::string apply_chat_template(const std::vector<PromptItem>& chat_prompts) const`
- **描述：** 将聊天模板应用于提示列表。

#### `std::string response(const std::string& user_content, std::ostream* os = &std::cout, const char* end_with = nullptr)`
- **描述：** 根据用户内容生成响应。
- **参数：**
  - `os`：输出流（默认为 `std::cout`）。
  - `end_with`：可选的附加字符串。
- **返回：** 生成的响应。

#### `void print_speed()`
- **描述：** 打印处理速度指标。

### 配置方法
#### `std::string dump_config()`
- **描述：** 以字符串形式转储当前配置。

#### `bool set_config(const std::string& content)`
- **描述：** 从字符串设置配置。
- **返回：** 如果成功，返回 `true`；否则返回 `false`。

### LoRA 方法
#### `size_t apply_lora(const std::string& lora_path)`
- **描述：** 从指定路径应用 LoRA 模型。

#### `Llm* create_lora(const std::string& lora_path)`
- **描述：** 创建 LoRA 实例。

#### `bool release_module(size_t index)`
- **描述：** 释放指定模块。
- **返回：** 如果成功，返回 `true`。

#### `bool select_module(size_t index)`
- **描述：** 选择指定模块。
- **返回：** 如果成功，返回 `true`。

---

## 公共成员
- `int prompt_len_`：提示长度。
- `int gen_seq_len_`：生成序列的长度。
- `int all_seq_len_`：总序列长度。
- `std::vector<int> history_ids_`：历史 ID 列表。
- `int64_t visual_us_`：视觉处理所用时间。
- `int64_t prefill_us_`：预填充所用时间。
- `int64_t decode_us_`：解码所用时间。
- `float load_progress_`：加载进度。
- `bool is_single_`：指示是否为单一模型。
- `bool attention_fused_`：指示是否融合了注意力机制。

## 保护成员
- `std::shared_ptr<LlmConfig> config_`：模型配置。
- `std::shared_ptr<Tokenizer> tokenizer_`：文本处理的分词器。
- `std::vector<int> key_value_shape_`：键值对的形状。
- `std::vector<MNN::Express::VARP> past_key_values_`：历史键值对。
- `MNN::Express::VARP inputs_embeds_, attention_mask_, position_ids_`：模型输入。
- `std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_manager_`：管理运行时执行。
- `std::vector<std::shared_ptr<MNN::Express::Module>> modules_`：加载的模块。
- `const MNN::Express::Module* base_module_`：基础模块指针。

---

## 保护方法
#### `void init_runtime()`
- **描述：** 初始化运行时参数。

#### `std::string decode(int id)`
- **描述：** 将给定 ID 解码为字符串。

#### `bool is_stop(int token_id)`
- **描述：** 检查令牌是否为停止令牌。

#### `virtual std::vector<int> tokenizer(const std::string& query)`
- **描述：** 对输入查询进行分词。

#### `virtual MNN::Express::VARP embedding(const std::vector<int>& input_ids)`
- **描述：** 为输入 ID 生成嵌入。

#### `virtual MNN::Express::VARP gen_attention_mask(int seq_len)`
- **描述：** 为序列生成注意力掩码。

#### `virtual MNN::Express::VARP gen_position_ids(int seq_len)`
- **描述：** 为序列生成位置 ID。

---

## Embedding 类

### 描述
`Embedding` 类负责生成文本和 ID 的嵌入表示，并提供相应的操作。

### 构造函数
#### `Embedding(std::shared_ptr<LlmConfig> config)`
- **参数：**
  - `config`：指向配置的共享指针。

### 公共方法
#### `static Embedding* createEmbedding(const std::string& config_path, bool load = true)`
- **描述：** 工厂方法，创建 `Embedding` 实例。
- **参数：**
  - `config_path`：配置文件路径。
  - `load`：是否加载模型（默认为 `true`）。
- **返回：** 创建的 `Embedding` 实例的指针。

#### `static float dist(MNN::Express::VARP var0, MNN::Express::VARP var1)`
- **描述：** 计算两个嵌入的距离。
- **返回：** 嵌入的距离值。

#### `virtual void load() override`
- **描述：** 加载嵌入模型。

#### `MNN::Express::VARP ids_embedding(const std::vector<int>& ids)`
- **描述：** 根据 ID 生成嵌入表示。
- **返回：** 生成的嵌入变量。

#### `MNN::Express::VARP txt_embedding(const std::string& txt)`
- **描述：** 根据文本生成嵌入表示。
- **返回：** 生成的嵌入变量。

#### `int dim() const`
- **描述：** 获取嵌入的维度。
- **返回：** 嵌入维度。

### 私有方法
#### `virtual std::vector<int> tokenizer(const std::string& query) override`
- **描述：** 对输入查询进行分词。

#### `virtual MNN::Express::VARP gen_attention_mask(int seq_len) override`
- **描述：** 为序列生成注意力掩码。

#### `virtual MNN::Express::VARP gen_position_ids(int seq_len) override`
- **描述：** 为序列生成位置 ID。

---

## TextVectorStore 类

### 描述
`TextVectorStore` 类用于管理文本的向量存储，支持文本的添加和搜索。

### 构造函数
#### `TextVectorStore()`
- **描述：** 默认构造函数。

#### `TextVectorStore(std::shared_ptr<Embedding> embedding)`
- **参数：**
  - `embedding`：指向 `Embedding` 实例的共享指针。

### 析构函数
#### `~TextVectorStore()`
- **描述：** 清理资源。

### 公共方法
#### `static TextVectorStore* load(const std::string& path, const std::string& embedding_path = "")`
- **描述：** 从路径加载 `TextVectorStore` 实例。
- **返回：** 加载的实例指针。

#### `void set_embedding(std::shared_ptr<Embedding> embedding)`
- **描述：** 设置嵌入实例。

#### `void save(const std::string& path)`
- **描述：** 保存文本向量存储到指定路径。

#### `void add_text(const std::string& text)`
- **描述：** 添加单个文本到存储。

#### `void add_texts(const std::vector<std::string>& texts)`
- **描述：** 添加多个文本到存储。

#### `std::vector<std::string> search_similar_texts(const std::string& txt, int topk = 1)`
- **描述：** 搜索与给定文本相似的文本。
- **参数：**
  - `topk`：返回的相似文本数量（默认为 1）。
- **返回：** 相似文本的列表。

#### `void bench()`
- **描述：** 进行基准测试。

### 保护方法
#### `inline VARP text2vector(const std::string& text)`
- **描述：** 将文本转换为向量。

---

## Document 类

### 描述
`Document` 类表示一个文档，支持根据类型加载和分割文档内容。

### 枚举类型
#### `enum DOCTYPE`
- `AUTO`：自动识别类型。
- `TXT`：文本文件。
- `MD`：Markdown 文件。
- `HTML`：HTML 文件。
- `PDF`：PDF 文件。

### 构造函数
#### `Document(const std::string& path, DOCTYPE type = AUTO)`
- **参数：**
  - `path`：文档路径。
  - `type`：文档类型（默认为 `AUTO`）。

### 析构函数
#### `~Document()`
- **描述：** 清理资源。

### 公共方法
#### `std::vector<std::string> split(int chunk_size = -1)`
- **描述：** 将文档分割为多个部分。
- **参数：**
  - `chunk_size`：每部分的大小（默认为 -1，表示不限制）。
- **返回：** 分割后的字符串列表。

### 私有方法
#### `std::vector<std::string> load_txt()`
- **描述：** 加载文本文件内容。

#### `std::vector<std::string> load_pdf()`
- **描述：** 加载 PDF 文件内容。

---

## MemoryBase 类

### 描述
`MemoryBase` 类是一个抽象基类，提供文本向量存储的基础功能。

### 构造函数
#### `MemoryBase()`
- **描述：** 默认构造函数。

### 析构函数
#### `virtual ~MemoryBase()`
- **描述：** 清理资源。

### 公共方法
#### `void set_embedding(std::shared_ptr<Embedding> embedding)`
- **描述：** 设置嵌入实例。

#### `virtual std::vector<std::string> search(const std::string& query, int topk)`
- **描述：** 根据查询搜索文本。
- **参数：**
  - `topk`：返回的结果数量。
- **返回：** 搜索结果列表。

#### `virtual void save(const std::string& path) = 0`
- **描述：** 保存存储（纯虚函数）。

#### `virtual void build_vectors() = 0`
- **描述：** 构建文本向量（纯虚函数）。

### 保护方法
#### `void load_store(const std::string& path)`
- **描述：** 从指定路径加载存储。

#### `void save_store(const std::string& path)`
- **描述：** 将存储保存到指定路径。

---

## ChatMemory 类

### 描述
`ChatMemory` 类继承自 `MemoryBase`，用于管理聊天记忆。

### 构造函数
#### `ChatMemory()`
- **描述：** 默认构造函数。

### 析构函数
#### `~ChatMemory() override`
- **描述：** 清理资源。

### 公共方法
#### `static ChatMemory* load(const std::string& path)`
- **描述：** 从路径加载 `ChatMemory` 实例。
- **返回：** 加载的实例指针。

#### `void save(const std::string& path) override`
- **描述：** 保存聊天记忆到指定路径。

#### `void build_vectors() override`
- **描述：** 构建聊天记忆的向量。

#### `std::string get_latest(std::string key)`
- **描述：** 获取指定键的最新记忆。

#### `void add(const std::vector<Prompt>& prompts)`
- **描述：** 添加提示到记忆中。

#### `void summarize(std::shared_ptr<Llm> llm)`
- **描述：** 对记忆进行总结。

### 私有成员
- `json memory_`：存储记忆的 JSON 对象。

---

## Knowledge 类

### 描述
`Knowledge` 类继承自 `MemoryBase`，用于管理知识库。

### 构造函数
#### `Knowledge()`
- **描述：** 默认构造函数。

### 析构函数
#### `~Knowledge() override`
- **描述：** 清理资源。

### 公共方法
#### `static Knowledge* load(const std::string& path)`
- **描述：** 从路径加载 `Knowledge` 实例。
- **返回：** 加载的实例指针。

#### `void save(const std::string& path) override`
- **描述：** 保存知识库到指定路径。

#### `void build_vectors() override`
- **描述：** 构建知识库的向量。

### 私有成员
- `std::unique_ptr<Document> document_`：指向文档的智能指针。

---

## Pipeline 类

### 描述
`Pipeline` 类用于处理输入字符串，并管理不同的模块，包括 LLM、嵌入、知识库和聊天记忆。

### 构造函数
#### `Pipeline()`
- **描述：** 默认构造函数，用于初始化 `Pipeline` 实例。

### 析构函数
#### `~Pipeline()`
- **描述：** 析构函数，用于清理资源。

### 公共方法
#### `static Pipeline* load(const std::string& path)`
- **描述：** 从指定路径加载 `Pipeline` 实例。
- **参数：**
  - `path`：配置文件路径。
- **返回：** 加载的 `Pipeline` 实例指针。

#### `void invoke(const std::string& str)`
- **描述：** 处理输入字符串并执行相应操作。
- **参数：**
  - `str`：输入字符串。

### 私有方法
#### `bool need_memory(const std::string& str)`
- **描述：** 判断输入字符串是否需要记忆模块。

#### `bool need_knowledge(const std::string& str)`
- **描述：** 判断输入字符串是否需要知识库模块。

#### `std::string build_prompt(const std::string& str)`
- **描述：** 根据输入字符串构建提示。

### 私有成员
- `std::unique_ptr<Llm> llm_`：指向 LLM 实例的智能指针。
- `std::shared_ptr<Embedding> embedding_`：指向嵌入实例的共享指针。
- `std::unique_ptr<Knowledge> knowledge_`：指向知识库实例的智能指针。
- `std::unique_ptr<ChatMemory> memory_`：指向聊天记忆实例的智能指针。
- `std::string system_`：系统信息字符串。
- `std::string user_`：用户信息字符串。
- `std::string assistant_`：助手信息字符串。
- `std::vector<Prompt> prompts_`：提示的向量。
- `json config_`：配置的 JSON 对象。

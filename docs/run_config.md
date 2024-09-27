# 推理配置文件

## 配置项
- max_new_tokens: 生成时最大token数，默认为`512`
- reuse_kv: 多轮对话时是否复用之前对话的`kv cache`，默认为`false`
- quant_qkv: CPU attention 算子中`query, key, value`是否量化，可选为：`0, 1, 2, 3, 4`，默认为`0`，含义如下：
  - 0: key和value都不量化
  - 1: 使用非对称8bit量化存储key
  - 2: 使用fp8格式量化存储value
  - 3: 使用非对称8bit量化存储key，使用fp8格式量化存储value
  - 4: 量化kv的同时使用非对称8bit量化query，并使用int8矩阵乘计算Q*K
- use_mmap: 是否使用mmap方式，在内存不足时将权重写入磁盘，避免溢出，默认为false，手机上建议设成true
- kvcache_mmap: 是否使用mmap方式，在内存不足时将在KV Cache 写入磁盘，避免溢出，默认为false
- tmp_path: 启用 mmap 相关功能时，写入磁盘的缓存目录
  - iOS 上可用如下语句创建临时目录并设置：`NSString *tempDirectory = NSTemporaryDirectory();llm->set_config("{\"tmp_path\":\"" + std::string([tempDirectory UTF8String]) + "\"}")`
- 硬件配置
  - backend_type: 推理使用硬件后端类型，默认为：`"cpu"`
  - thread_num: CPU推理使用硬件线程数，默认为：`4`; OpenCL推理时使用`68`
  - precision: 推理使用精度策略，默认为：`"low"`，尽量使用`fp16`
  - memory: 推理使用内存策略，默认为：`"low"`，开启运行时量化

## 示例

```json
{
    "llm_model": "qwen2-1.5b-int4.mnn",
    "llm_weight": "qwen2-1.5b-int4.mnn.weight",

    "backend_type": "cpu",
    "thread_num": 4,
    "precision": "low",
    "memory": "low"
}
```

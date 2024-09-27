# 模型配置文件

## 配置项
- base_dir: 模型文件加载的文件夹目录，默认为config.json的所在目录，或模型所在目录；
- llm_config: `llm_config.json`的实际名称路径为`base_dir + llm_config`，默认为`base_dir + 'config.json'`
- llm_model: `llm.mnn`的实际名称路径为`base_dir + llm_model`，默认为`base_dir + 'llm.mnn'`
- llm_weight: `llm.mnn.weight`的实际名称路径为`base_dir + llm_weight`，默认为`base_dir + 'llm.mnn.weight'`
- block_model: 分段模型时`block_{idx}.mnn`的实际路径为`base_dir + block_model`，默认为`base_dir + 'block_{idx}.mnn'`
- lm_model: 分段模型时`lm.mnn`的实际路径为`base_dir + lm_model`，默认为`base_dir + 'lm.mnn'`
- embedding_model: 当embedding使用模型时，embedding的实际路径为`base_dir + embedding_model`，默认为`base_dir + 'embedding.mnn'`
- embedding_file: 当embedding使用二进制时，embedding的实际路径为`base_dir + embedding_file`，默认为`base_dir + 'embeddings_bf16.bin'`
- tokenizer_file: `tokenizer.txt`的实际名称路径为`base_dir + tokenizer_file`，默认为`base_dir + 'tokenizer.txt'`
- visual_model: 当使用VL模型时，visual_model的实际路径为`base_dir + visual_model`，默认为`base_dir + 'visual.mnn'`

## 示例

```json
{
    "hidden_size": 1536,
    "layer_nums": 28,
    "attention_mask": "float",
    "key_value_shape": [
        2,
        1,
        0,
        2,
        128
    ],
    "prompt_template": "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "is_visual": false,
    "is_single": true
}
```
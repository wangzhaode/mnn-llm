# demo

## cli_demo

功能：交互式对话，性能测试示例。

用法：`Usage: ./cli_demo model_dir <prompt.txt>`

示例：
```
# chat mode, 交互式对话
./cli_demo ../qwen-1.8b-int4
# benchmark, 性能测试
./cli_demo ../qwen-1.8b-int4 ../resource/prompt.txt
```

## tokenizer_demo

功能：文本与token_id转换示例。

用法：`Usage: ./tokenizer_demo tokenizer.txt`

示例：
```
./tokenizer_demo ../qwen-1.8b-int4/tokenizer.txt
```

## embedding_demo

功能：文本转向量，向量距离计算示例。

用法：`Usage: ./embedding_demo embedding.mnn`

示例：
```
./embedding_demo ../bge-large-zh-int8.mnn
```

## store_demo

功能：文本向量存储，搜索示例。

用法：`Usage: ./store_demo embedding.mnn`

示例：
```
./store_demo ../bge-large-zh-int8.mnn
```

## document_demo

功能：文本读取，分段示例。

用法：`Usage: ./document_demo doc.txt`

示例：
```
./document_demo demo.md
```

## memory_demo

功能：记忆库示例。

用法：`Usage: ./memory_demo embedding.mnn history.json <llm.mnn>`

示例：
```
# 记忆向量化，查询
./memory_demo ../bge-large-zh-int8.mnn ../resource/memory_demo.json
# 记忆总结，向量化，查询
./memory_demo ../bge-large-zh-int8.mnn ../resource/memory_demo.json ../qwen-7b-int4
```

## knowledge_demo

功能：知识库示例。

用法：`Usage: ./knowledge_demo embedding.mnn knowledge.md`

示例：
```
# 知识库构建，查询
./knowledge_demo ../bge-large-zh-int8.mnn demo.md
```

# pipeline_demo

功能：流水线聊天示例。

用法：`Usage: ./pipeline_demo config.json`

示例：
```
./pipeline_demo ../resource/config.json
```
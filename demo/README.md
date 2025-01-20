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
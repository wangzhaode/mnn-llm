# Python API

## LLM 类

### 描述
`LLM` 类用于加载模型并生成输出。它继承自 `cmnnllm.LLM`。

### 方法

#### `load(model_dir: str)`
- **描述：** 从指定路径加载模型。
- **参数：**
  - `model_dir`：模型路径（分割）或模型名称（单一）。
- **返回：** `None`
- **示例：**
  ```python
  >>> llm.load('../qwen-1.8b-in4/config.json')
  ```

#### `generate(input_ids: list)`
- **描述：** 根据输入 token ID 生成输出。
- **参数：**
  - `input_ids`：输入 token ID 列表（整型）。
- **返回：** 输出 token ID 列表（整型）。
- **示例：**
  ```python
  >>> input_ids = [151644, 872, 198, 108386, 151645, 198, 151644, 77091]
  >>> output_ids = qwen.generate(input_ids)
  ```

#### `response(prompt: str, stream: bool = False)`
- **描述：** 根据输入提示生成响应。
- **参数：**
  - `prompt`：输入提示字符串。
  - `stream`：是否生成字符串流，默认为 `False`。
- **返回：** 输出字符串。
- **示例：**
  ```python
  >>> res = qwen.response('Hello', True)
  ```

## create 函数

### 描述
创建 `LLM` 实例。

### 参数
- **config_path :** 配置文件路径或模型路径。

### 返回
- **llm :** `LLM` 实例。

### 示例
```python
>>> qwen = mnnllm.create('./qwen-1.8b-int4/config.json')
```
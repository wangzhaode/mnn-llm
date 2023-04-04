# 模型的导出与转换

## ChatGLM分层导出ONNX模型

### 分层导出glm_block
1. 使用该文件夹下的`modeling_chatglm.py`作为ChatGLM-6B的执行代码；该代码已将embedding层移出forward函数，并且注释掉了模型最后的`lm_head`层，并且仅在forward函数中执行一层block；
2. 修改`export_layers.sh`脚本中`code`变量为`modeling_chatglm.py`的地址；执行脚本即可完成0-26层导出：
   ```
   ./export_layers.sh
   ```
3. 修改`modeling_chatglm.py`，将832行下标改为27，取消843行注释；执行如下命令可以导出第27层：
   ```
   python export.py 27
   ```
### 导出embedding层
取消1021行注释，同时注释掉`forward`函数中的其他计算逻辑；执行export.py中的`embedding_to_onnx`函数即可；

### 导出lm_head层
取消1022行注释，同时注释掉`forward`函数中的其他计算逻辑；执行export.py中的`lm_to_onnx`函数即可；

## ONNX模型转换到MNN
1. 在编译MNN时打开宏`-DMNN_BUILD_CONVERTER=ON`；
2. 修改`onnx2mnn.sh`中的路径， quantbit，在`build`目录下执行；
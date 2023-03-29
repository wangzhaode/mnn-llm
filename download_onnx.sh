for i in `seq 0 27`
do
    wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.4/glm_block_$i.onnx
done

wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.4/embedding.onnx
wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.4/embedding_weight.zip
wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.4/lm_head.onnx
wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.4/lm_head_weight.zip


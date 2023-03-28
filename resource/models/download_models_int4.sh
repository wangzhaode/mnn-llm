for i in `seq 0 27`
do
    wget https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.3/glm_block_$i.mnn
done

wget https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn
wget https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin

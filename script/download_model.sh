model=$1
mkdir $model
cd $model
# download models
wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/embedding.mnn
wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/lm.mnn
for i in `seq 0 31`
do
    wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/block_$i.mnn
done
cd ..
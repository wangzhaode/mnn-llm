model=$1
mkdir $model
cd $model
# download models
wget -c https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/embedding.mnn
wget -c https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/lm.mnn
for i in `seq 0 27`
do
    wget -c https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/block_$i.mnn
done
cd ..
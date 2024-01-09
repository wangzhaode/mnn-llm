if [ $# -lt 1 ]; then
    echo 'Usage: ./download_model.sh $model'
    exit 1
fi

model=$1
mkdir $model
cd $model
is_7b=`echo $model | grep '7b'`
is_1_8b=`echo $model | grep '1.8b'`
block_num=27
if [ $is_7b ]; then
    block_num=31
fi
if [ $is_1_8b ]; then
    block_num=23
fi
# download models
wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/tokenizer.txt
wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/embedding.mnn
wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/lm.mnn
for i in `seq 0 $block_num`
do
    wget -c -nv https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/block_$i.mnn
done
cd ..

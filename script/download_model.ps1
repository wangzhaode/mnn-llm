param(
     $model
)
mkdir $model
cd $model
wget -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/embedding.mnn -OutFile embedding.mnn
wget -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/lm.mnn -OutFile lm.mnn
for($i=1; $i -lt 32; $i=$i+1)
{   
    wget -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/block_$i.mnn  -OutFile  block_$i.mnn
}
cd ..

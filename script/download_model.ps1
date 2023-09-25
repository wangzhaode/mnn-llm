param(
     $model
)
mkdir $model
cd $model
$block_num = 28
if ($model.Contains('7b')) {
    $block_num = 32
}
Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/tokenizer.txt -OutFile tokenizer.txt
Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/embedding.mnn -OutFile embedding.mnn
Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/lm.mnn -OutFile lm.mnn
for ($i=1; $i -lt $block_num; $i=$i+1) {
    Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/block_$i.mnn  -OutFile  block_$i.mnn
}
cd ..

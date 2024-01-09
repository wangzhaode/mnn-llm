param(
     $model
)
mkdir $model
cd $model
$block_num = 27
if ($model.Contains('7b')) {
    $block_num = 31
}
if ($model.Contains('1.8b')) {
    $block_num = 23
}
Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/tokenizer.txt -OutFile tokenizer.txt
Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/embedding.mnn -OutFile embedding.mnn
Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/lm.mnn -OutFile lm.mnn
for ($i=0; $i -lt $block_num; $i=$i+1) {
    Invoke-WebRequest -Uri https://github.com/wangzhaode/mnn-llm/releases/download/$model-mnn/block_$i.mnn  -OutFile  block_$i.mnn
}
cd ..

for i in `seq 0 27`
do
    echo $i
    src=./glm_block_$i.onnx
    dst=./models/glm_block_$i.mnn
    ./MNNConvert -f ONNX --modelFile $src --MNNModel $dst --bizCode mnn --weightQuantBits 4 --weightQuantAsymmetric
done
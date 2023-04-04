code=src/2449bdc9d85103734ae987bdc94eafa7fec2145d/modeling_chatglm.py
for i in `seq 0 26`
do
    j=$((i+1))
    echo $j
    sed -i 's/layer_ret = self.layers\['$i'\]/layer_ret = self.layers\['$j'\]/g' $code
    sed -i 's/layer_id=torch.tensor('$i')/layer_id=torch.tensor('$j')/g' $code
    python export.py $j
done
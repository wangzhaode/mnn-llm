# mnnllm python

## Build & Install
```sh
# install
python setup.py install

# build wheel
python setup.py bdist_wheel
```

## Usage
```python
import mnnllm

# config path
config_path = './qwen2-1.5b-int4/config.json'

# create model
qwen = mnnllm.create(config_path)
# load model
qwen.load()

# response stream
out = qwen.response('你好', True)

# generate
input_ids = [151644, 872, 198, 108386, 151645, 198, 151644, 77091]
output_ids = qwen.generate(input_ids)
print(output_ids)
```
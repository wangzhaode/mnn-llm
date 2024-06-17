import mnnllm
import sys

config_path = sys.argv[1]
# create model
qwen = mnnllm.create(config_path)
# load model
qwen.load()

# response stream
out = qwen.response('你好', True)
print(out)

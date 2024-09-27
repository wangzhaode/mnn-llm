# 运行

## 1. 下载模型
```
huggingface-cli download --resume-download zhaode/Qwen2-1.5B-Instruct-MNN --local-dir Qwen2-1.5B-Instruct-MNN
```

## 2. 运行

```bash
# linux/macos
./cli_demo ./Qwen2-1.5B-Instruct-MNN/config.json # cli demo
./web_demo ./Qwen2-1.5B-Instruct-MNN/config.json ../web # web ui demo

# windows
.\Debug\cli_demo.exe ./Qwen2-1.5B-Instruct-MNN/config.json
.\Debug\web_demo.exe ./Qwen2-1.5B-Instruct-MNN/config.json ../web

# android
adb push libs/*.so build/libllm.so build/cli_demo /data/local/tmp
adb push model_dir /data/local/tmp
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./cli_demo ./Qwen2-1.5B-Instruct-MNN/config.json"
```
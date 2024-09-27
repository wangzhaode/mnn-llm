# 编译

## 编译宏
- `BUILD_FOR_ANDROID`: 编译到Android设备；
- `LLM_SUPPORT_VISION`: 是否支持视觉处理能力；
- `DUMP_PROFILE_INFO`: 每次对话后dump出性能数据到命令行中；

默认使用`CPU`，如果使用其他后端或能力，可以在编译MNN时添加`MNN`编译宏
- cuda: `-DMNN_CUDA=ON`
- opencl: `-DMNN_OPENCL=ON`
- metal: `-DMNN_METAL=ON`

## 编译脚本
```
# clone
git clone --recurse-submodules https://github.com/wangzhaode/mnn-llm.git
cd mnn-llm

# linux
./script/build.sh

# macos
./script/build.sh

# windows msvc
./script/build.ps1

# python wheel
./script/py_build.sh

# android
./script/android_build.sh

# android apk
./script/android_app_build.sh

# ios
./script/ios_build.sh
```
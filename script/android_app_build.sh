# 1. clone MNN
git clone https://github.com/alibaba/MNN.git --depth=1

# 2. build MNN
cd MNN/project/android
mkdir build
cd build
../build_64.sh
cd ../../../..

# 3. copy headers and libs
cp -r MNN/include/MNN include
cp MNN/project/android/build/libMNN.so MNN/project/android/build/libMNN_Express.so android/app/src/main/jni/libs/arm64-v8a

# 4. build mnn-llm android apk
cd android
./gradlew assembleDebug
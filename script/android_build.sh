# 1. clone MNN
git clone https://github.com/alibaba/MNN.git --depth=1

# 2. build MNN
cd MNN/project/android
mkdir build
cd build
../build_64.sh -DMNN_LOW_MEMORY=ON
cd ../../../..

# 3. copy headers and libs
cp -r MNN/include/MNN include
cp MNN/project/android/build/libMNN.so MNN/project/android/build/libMNN_Express.so libs

# 4. build mnn-llm android
mkdir android_build
cd android_build
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DANDROID_STL=c++_static \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_FOR_ANDROID=ON
make -j4
cd ..
cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DANDROID_STL=c++_static \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_FOR_ANDROID=ON

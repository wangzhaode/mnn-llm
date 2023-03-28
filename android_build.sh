cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DANDROID_STL=c++_static \
-DCMAKE_BUILD_TYPE=Release \
-DMINI_MEM_MODE=ON

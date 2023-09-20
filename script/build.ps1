# 1. clone MNN
git clone https://github.com/alibaba/MNN.git --depth=1

# 2. build MNN
cd MNN
mkdir build
cd build
cmake -DMNN_LOW_MEMORY=ON -DMNN_WIN_RUNTIME_MT=ON ..
cmake --build . --config Release -j 4
cd ../..

# 3. copy headers and libs
cp -r MNN\include\MNN include
cp MNN\build\Release\MNN.lib libs
cp MNN\build\Release\MNN.dll libs

# 4. copy pthread
Expand-Archive .\resource\win_pthreads.zip
cp .\win_pthreads\Pre-built.2\lib\x64\pthreadVC2.lib libs
cp .\win_pthreads\Pre-built.2\include\*.h .\include\

# 5. build mnn-llm
mkdir build
cd build
cmake ..
cmake --build . --config Release -j 4
cd ..
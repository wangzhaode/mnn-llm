# 1. clone MNN
git clone https://github.com/alibaba/MNN.git -b 2.6.3 --depth=1

# 2. build MNN
cd MNN
mkdir build
cd build
# cmake -DMNN_LOW_MEMORY=ON ..
cmake ..
cmake --build . -j 4
cd ../..

# 3. copy headers and libs
cp -r MNN/include/MNN include
mv MNN/build/libMNN.lib libs

# 4. download pthread
wget -Uri https://gigenet.dl.sourceforge.net/project/pthreads4w/pthreads-w32-2-9-1-release.zip -OutFile "pthreads.zip"
Expand-Archive .\pthreads.zip
mv .\pthreads\Pre-built.2\lib\x64\pthreadVC2.lib libs
cp .\pthreads\Pre-built.2\include\*.h .\include\

# 5. build mnn-llm
mkdir build
cd build
cmake ..
cmake --build .  -j 4
cd ..
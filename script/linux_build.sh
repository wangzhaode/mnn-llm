# 1. clone MNN
git clone https://github.com/alibaba/MNN.git -b 2.7.1 --depth=1

# 2. build MNN
cd MNN
mkdir build
cd build
cmake -DMNN_LOW_MEMORY=ON ..
make -j4
cd ../..

# 3. copy headers and libs
cp -r MNN/include/MNN include
cp MNN/build/libMNN.so MNN/build/express/libMNN_Express.so libs

# 4. build mnn-llm
mkdir build
cd build
cmake ..
make -j4
cd ..
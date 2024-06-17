# 1. build lib
mkdir build
cd build
cmake ..
make -j4
cd ..
# 2. install python wheel
cd python
python setup.py bdist_wheel
cd ..
mkdir build

Expand-Archive .\resource\win_pthreads.zip
cp .\win_pthreads\Pre-built.2\lib\x64\pthreadVC2.lib build
cp .\win_pthreads\Pre-built.2\include\*.h .\include\

cd build
cmake ..
cmake --build . --config Release -j 4
cd ..
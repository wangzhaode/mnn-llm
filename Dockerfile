FROM nvcr.io/nvidia/tensorrt:23.03-py3

# update timezone
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install dependencies
RUN sed -i "s@http://ftp.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://security.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    apt update  -y && \
    apt install lsof vim gcc git -y && \
    apt install tzdata -y


# setting workspace
WORKDIR /workspace

COPY include include
COPY resources resources
COPY CMakeLists.txt CMakeLists.txt
COPY src src
COPY libs libs
COPY demo demo

# build MNN
RUN git clone https://github.com/alibaba/MNN.git -b 2.4.0 && \
  cd MNN && \
  mkdir build && cd build && \
  cmake .. -DMNN_CUDA=ON && \
  make -j$(nproc) && \
  cd .. && \
  cp -r MNN/include/MNN include && \
  cp MNN/build/libMNN.so libs/ && \
  cp MNN/build/express/*.so  libs/ 

# build ChatGLM-MNN
RUN mkdir build && cd build && \
  cmake -D WITH_CUDA=on .. && \
  make -j$(nproc) && \
  cd ..

EXPOSE 5088

# run
CMD ["cd build && ./web_demo.cpp"]



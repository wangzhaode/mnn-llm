project(chatglm_mnn)

cmake_minimum_required(VERSION 3.10)

# set(MNN_DIR ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/lib/cmake/ncnn)
# find_package(MNN REQUIRED)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../../../include/)
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs/arm64-v8a)

add_definitions(-DUSING_DISK_EMBED)
FILE(GLOB SRCS ../../../../../src/*.cpp)
add_library(llm_mnn SHARED llm_mnn_jni.cpp ${SRCS})

target_link_libraries(llm_mnn MNN MNN_Express log)
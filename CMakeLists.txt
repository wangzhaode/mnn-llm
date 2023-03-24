cmake_minimum_required(VERSION 3.0)
project(chatglm-mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# compile dynamic lib
add_library(chat SHARED ${SRCS})
target_link_libraries(chat MNN MNN_Express)

# demo target
add_executable(demo ${CMAKE_CURRENT_LIST_DIR}/demo/main.cpp)
target_link_libraries(demo chat)
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
# target_link_libraries(chat MNN MNN_Express MNN_CL) # if using OPENCL

# demo targets
add_executable(cli_demo ${CMAKE_CURRENT_LIST_DIR}/demo/cli_demo.cpp)
target_link_libraries(cli_demo chat)

add_executable(web_demo ${CMAKE_CURRENT_LIST_DIR}/demo/web_demo.cpp)
# target_link_libraries(web_demo chat pthread ssl crypto)
target_link_libraries(web_demo chat pthread)

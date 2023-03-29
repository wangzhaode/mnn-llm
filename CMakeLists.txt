cmake_minimum_required(VERSION 3.0)
project(chatglm-mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option(BUILD_FOR_ANDROID "Build for android whith mini memory mode." OFF)
option(BUILD_MINI_MEM_MODE "Build whith mini memory mode." OFF)

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

if (BUILD_FOR_ANDROID)
    target_compile_options(chat PRIVATE -DMINI_MEM_MODE)
    add_library(MNN SHARED IMPORTED)
    add_library(MNN_Express SHARED IMPORTED)
    set_target_properties(
        MNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/libs/libMNN.so
        )
    set_target_properties(
        MNN_Express
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/libs/libMNN_Express.so
        )
    # just cli demo
    add_executable(cli_demo ${CMAKE_CURRENT_LIST_DIR}/demo/cli_demo.cpp)
    target_link_libraries(cli_demo chat log)
else()
    if (BUILD_MINI_MEM_MODE)
        target_compile_options(chat PRIVATE -DMINI_MEM_MODE)
    endif()
    # cli demo
    add_executable(cli_demo ${CMAKE_CURRENT_LIST_DIR}/demo/cli_demo.cpp)
    target_link_libraries(cli_demo chat)
    # web demo
    add_executable(web_demo ${CMAKE_CURRENT_LIST_DIR}/demo/web_demo.cpp)
    target_link_libraries(web_demo chat pthread)
endif()

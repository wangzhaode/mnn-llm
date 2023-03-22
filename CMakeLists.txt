cmake_minimum_required(VERSION 3.0)
project(chatglm-mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# target
add_executable(chat ${SRCS})

if (MSVC)
    target_link_libraries(chat MNN)
else()
    target_link_libraries(chat MNN)
endif()
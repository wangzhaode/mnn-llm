cmake_minimum_required(VERSION 3.0)
project(mnn-llm)

option(BUILD_FOR_ANDROID "Build for android whith mini memory mode." OFF)
option(USING_VISUAL_MODEL "Using visual model will need dpes: MNNOpenCV and httplib." OFF)
option(USING_DISK_EMBED "Using disk embedding to save memory." OFF)
option(DUMP_PROFILE_INFO "Dump profile info when chat." OFF)

if (USING_DISK_EMBED)
    add_definitions(-DUSING_DISK_EMBED)
endif()

if (USING_VISUAL_MODEL)
    add_definitions(-DUSING_VISUAL_MODEL)
endif()

if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    # compile static lib, surrpot Winwows
    add_library(llm STATIC ${SRCS})
    target_link_libraries(llm MNN.lib)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    # compile dynamic so, support Linux/Mac
    add_library(llm SHARED ${SRCS})
    set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

    target_link_libraries(llm MNN MNN_Express)
    if (USING_VISUAL_MODEL)
        target_link_libraries(llm MNNOpenCV)
    endif()
endif()

if (BUILD_FOR_ANDROID)
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
    target_link_libraries(cli_demo llm log)
else()
    # cli demo
    add_executable(cli_demo ${CMAKE_CURRENT_LIST_DIR}/demo/cli_demo.cpp)
    add_executable(web_demo ${CMAKE_CURRENT_LIST_DIR}/demo/web_demo.cpp)
    add_executable(embedding_demo ${CMAKE_CURRENT_LIST_DIR}/demo/embedding_demo.cpp)
    if (MSVC)
        target_link_libraries(cli_demo llm)
        target_link_libraries(web_demo llm pthreadVC2)
        # copy all lib to target dir
        file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/libs/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug/)
    else()
        target_link_libraries(cli_demo llm)
        target_link_libraries(embedding_demo llm)
        target_link_libraries(web_demo llm pthread)
    endif()
endif()

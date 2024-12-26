cmake_minimum_required(VERSION 3.5)
project(mnn-llm)

option(BUILD_FOR_ANDROID "Build for android whith mini memory mode." OFF)
option(LLM_SUPPORT_VISION "Llm model support vision input." ON)
option(DUMP_PROFILE_INFO "Dump profile info when chat." OFF)
option(BUILD_JNI "Build JNI for android app." OFF)

if (DUMP_PROFILE_INFO)
    add_definitions(-DDUMP_PROFILE_INFO)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# compile MNN first
set(MNN_SEP_BUILD OFF CACHE BOOL "Close MNN_SEP_BUILD" FORCE)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "Close MNN_BUILD_SHARED_LIBS" FORCE)
set(MNN_BUILD_LLM ON CACHE BOOL "Open MNN_BUILD_LLM" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "Open MNN_LOW_MEMORY" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "Open MNN_SUPPORT_TRANSFORMER_FUSE" FORCE)
if (BUILD_FOR_ANDROID)
    set(MNN_ARM82 ON CACHE BOOL "Open MNN_ARM82" FORCE)
endif()
if (LLM_SUPPORT_VISION)
    add_definitions(-DLLM_SUPPORT_VISION)
    set(MNN_BUILD_OPENCV ON CACHE BOOL "Open MNN_BUILD_OPENCV" FORCE)
    set(MNN_IMGCODECS ON CACHE BOOL "Open MNN_IMGCODECS" FORCE)
endif()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/MNN)

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/tools/cv/include/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/MNN/3rd_party/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/source/
                    ${CMAKE_CURRENT_LIST_DIR}/MNN/transformers/llm/engine/include
                    )

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

# jni file
if (BUILD_JNI)
    list(APPEND SRCS "${CMAKE_CURRENT_LIST_DIR}/android/app/src/main/jni/llm_mnn_jni.cpp")
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

add_executable(mllm ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp)
target_link_libraries(mllm MNN)
if (0)
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++11")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    # compile static lib, surrpot Winwows
    add_library(llm STATIC ${SRCS})
    link_directories(${CMAKE_BINARY_DIR}/MNN/Release)
    target_link_libraries(llm MNN.lib)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    # compile dynamic so, support Linux/Mac
    add_library(llm SHARED ${SRCS})
    set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

    target_link_libraries(llm MNN MNN_Express)
    if (MNN_OPENCL)
        target_link_libraries(llm MNN_CL)
    endif()
    if (LLM_SUPPORT_VISION)
        target_link_libraries(llm MNNOpenCV)
    endif()
endif()

if (NOT (BUILD_JNI))
    add_executable(cli_demo ${CMAKE_CURRENT_LIST_DIR}/demo/cli_demo.cpp)
    add_executable(tokenizer_demo ${CMAKE_CURRENT_LIST_DIR}/demo/tokenizer_demo.cpp)
    add_executable(embedding_demo ${CMAKE_CURRENT_LIST_DIR}/demo/embedding_demo.cpp)
    add_executable(store_demo ${CMAKE_CURRENT_LIST_DIR}/demo/store_demo.cpp)
    add_executable(document_demo ${CMAKE_CURRENT_LIST_DIR}/demo/document_demo.cpp)
    add_executable(memory_demo ${CMAKE_CURRENT_LIST_DIR}/demo/memory_demo.cpp)
    add_executable(knowledge_demo ${CMAKE_CURRENT_LIST_DIR}/demo/knowledge_demo.cpp)
    add_executable(pipeline_demo ${CMAKE_CURRENT_LIST_DIR}/demo/pipeline_demo.cpp)

    if (BUILD_FOR_ANDROID)
        # add_library(MNN SHARED IMPORTED)
        # add_library(MNN_Express SHARED IMPORTED)
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
        target_link_libraries(cli_demo llm log)
        target_link_libraries(tokenizer_demo llm log)
        target_link_libraries(embedding_demo llm log)
        target_link_libraries(store_demo llm log)
        target_link_libraries(document_demo llm log)
        target_link_libraries(memory_demo llm log)
        target_link_libraries(knowledge_demo llm log)
        target_link_libraries(pipeline_demo llm log)
    else()
        target_link_libraries(cli_demo llm)
        target_link_libraries(tokenizer_demo llm)
        target_link_libraries(embedding_demo llm)
        target_link_libraries(store_demo llm)
        target_link_libraries(document_demo llm)
        target_link_libraries(memory_demo llm)
        target_link_libraries(knowledge_demo llm)
        target_link_libraries(pipeline_demo llm)
        # web demo
        add_executable(web_demo ${CMAKE_CURRENT_LIST_DIR}/demo/web_demo.cpp)
        if (MSVC)
            target_link_libraries(web_demo llm pthreadVC2)
            add_custom_command(
                TARGET cli_demo POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_SOURCE_DIR}/build/MNN/Release/MNN.dll
                    $<TARGET_FILE_DIR:cli_demo>
            )
        else()
            target_link_libraries(web_demo llm pthread)
        endif()
    endif()
else()
    set(LLM_LIBS "")
    list(APPEND LLM_LIBS $<TARGET_FILE:MNN>)
    list(APPEND LLM_LIBS $<TARGET_FILE:MNN_Express>)
    if (LLM_SUPPORT_VISION)
        #list(APPEND LLM_LIBS $<TARGET_FILE:MNNOpenCV>)
    endif()
    if (MNN_OPENCL)
        list(APPEND LLM_LIBS $<TARGET_FILE:MNN_CL>)
    endif()
    add_custom_command(TARGET llm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${LLM_LIBS}
                ${CMAKE_CURRENT_LIST_DIR}/android/app/src/main/jni/libs/arm64-v8a)
endif()
endif()
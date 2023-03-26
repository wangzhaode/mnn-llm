//
//  cli_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "chat.hpp"

int main(int argc, const char* argv[]) {
    ChatGLM chatglm(8);
    chatglm.chat();
    return 0;
}
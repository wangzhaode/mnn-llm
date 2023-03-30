//
//  cli_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "chat.hpp"
#include "CLI11.hpp"
#include <stdlib.h>

int main(int argc, const char* argv[]) {
    CLI::App app{"cli demo for chat.cpp"};

    float gpusize = 8.0;
	std::string modeldir = "../resource/models";
    std::string tokenizerdir = "../resource/tokenizer";
    app.add_option("-g,--gpusize", gpusize,"gpu memory size(G)");
	app.add_option("-m,--model", modeldir, "model directory");
    app.add_option("-t,--tokenizer", tokenizerdir, "tokenizer directory");

	CLI11_PARSE(app, argc, argv);

    ChatGLM chatglm(gpusize,modeldir,tokenizerdir);
    chatglm.chat();
    return 0;
}
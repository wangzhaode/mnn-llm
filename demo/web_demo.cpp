//
//  web_demo.cpp
//
//  Created by MNN on 2023/03/25.
//  ZhaodeWang
//

#include "llm.hpp"
#include "httplib.h"
#include <iostream>
#include <thread>

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " model_dir web_dir" << std::endl;
        std::cout << "Example: " << argv[0] << " ../qwen-1.8b-int4 ../web" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::string web_dir = argv[2];
    std::cout << "model path is " << model_dir << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load(model_dir);
    
    std::stringstream ss;
    httplib::Server svr;
    std::atomic_bool waiting;
    waiting = false;
    std::string last_request = "";
    auto chat = [&](std::string str) {
        waiting = true;
        llm->response(str, &ss, "<eop>");
        std::cout << "### response : " << ss.str() << std::endl;
        waiting = false;
    };
    svr.Post("/chat", [&](const httplib::Request &req, httplib::Response &res) {
        if (req.body == last_request) {
            res.set_content(ss.str(), "text/plain");
            return;
        }
        if (waiting) {
            res.set_content(ss.str(), "text/plain");
        } else {
            ss.str("");
            std::cout << "### request : " << req.body << std::endl;
            last_request = req.body;
            std::thread chat_thread(chat, last_request);
            chat_thread.detach();
        }
    });
    svr.set_mount_point("/", web_dir);
    printf(">>> please open http://0.0.0.0:8080 or http://localhost:8080\n");
    fflush(stdout);
    svr.listen("0.0.0.0", 8080);
    printf(">>> end\n");
    return 0;
}

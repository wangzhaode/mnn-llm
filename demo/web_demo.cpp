//
//  web_demo.cpp
//
//  Created by MNN on 2023/03/25.
//  ZhaodeWang
//

#include "chat.hpp"
#include "httplib.h"
#include <iostream>
#include <thread>

int main(int argc, const char* argv[]) {
    ChatGLM chatglm(8);
    std::stringstream ss;
    httplib::Server svr;
    std::atomic_bool waiting;
    waiting = false;
    std::string last_request = "";
    auto chat = [&](std::string str) {
        waiting = true;
        chatglm.response(str, &ss);
        waiting = false;
        std::cout << "### response : " << ss.str() << std::endl;
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
    svr.set_mount_point("/", "../resource/web");
    printf(">>> please open http://localhost:8081\n");
    svr.listen("localhost", 8081);
    printf(">>> end\n");
    return 0;
}
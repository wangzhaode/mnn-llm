//
//  LLMInferenceEngineWrapper.m
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

#import "LLMInferenceEngineWrapper.h"
#include "llm.hpp"

const char* GetMainBundleDirectory() {
    NSString *bundleDirectory = [[NSBundle mainBundle] bundlePath];
    return [bundleDirectory UTF8String];
}

@implementation LLMInferenceEngineWrapper {
    Llm* llm;
}

- (instancetype)initWithCompletionHandler:(ModelLoadingCompletionHandler)completionHandler {
    self = [super init];
    if (self) {
        // 在后台线程异步加载模型
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            BOOL success = [self loadModel]; // 假设loadModel方法加载模型并返回加载的成功或失败
            // 切回主线程回调
            dispatch_async(dispatch_get_main_queue(), ^{
                completionHandler(success);
            });
        });
    }
    return self;
}

- (BOOL)loadModel {
    if (!llm) {
        auto model_dir = GetMainBundleDirectory();
        llm = Llm::createLLM(model_dir, "qwen-1.8b");
        llm->load(model_dir);
    }
    return YES;
}

- (void)processInput:(NSString *)input withStreamHandler:(StreamOutputHandler)handler {
    LlmStreamBuffer::CallBack callback = [handler](const char* str, size_t len) {
        if (handler) {
            NSString *nsOutput = [NSString stringWithUTF8String:str];
            handler(nsOutput);
        }
    };
    LlmStreamBuffer streambuf(callback);
    std::ostream os(&streambuf);
    llm->response([input UTF8String], &os, "<eop>");
}

- (void)dealloc {
    delete llm;
}
@end

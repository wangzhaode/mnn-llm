//
//  document_demo.cpp
//
//  Created by MNN on 2024/01/11.
//  ZhaodeWang
//

#include "llm.hpp"
#include <fstream>
#include <stdlib.h>

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " doc.txt" << std::endl;
        return 0;
    }
    std::string doc_dir = argv[1];
    std::cout << "doc path is " << doc_dir << std::endl;
    std::unique_ptr<Document> document(new Document(doc_dir));
    auto segments = document->split();
    printf("segment size: %ld\n", segments.size());
    for (int i = 0; i < segments.size() && i < 3; i++) {
        printf("# %d segment [%ld]: \n%s\n", i, segments[i].size(), segments[i].c_str());
    }
    return 0;
}

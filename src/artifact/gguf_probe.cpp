#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    try {
        strata::GgufFile g(argv[1]);
        std::printf("parsed OK\n");
        const char* probe[] = {"general.architecture", "qwen4exp.block_count", "qwen4exp.expert_count",
                               "qwen4exp.embedding_length", "split.count", "split.no"};
        for (auto k : probe) {
            const auto* v = g.get(k);
            if (!v) std::printf("MISSING %s\n", k);
            else std::printf("found %s\n", k);
        }
    } catch (const std::exception& e) {
        std::printf("EXC: %s\n", e.what());
    }
    return 0;
}

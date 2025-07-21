#include "lootinator/template/kernel_template.h"

namespace loot {
    KernelTemplate::KernelTemplate(const TemplateParameters& params) : Template(params) {}

    void KernelTemplate::generate(std::ostream& out) const {
        generatePreamble(out);
        generateLootLookupTable(out);
        generateLootProcessors(out);
        generateKernelHeader(out);
        generateKernelBody(out);
        generateHostController(out);
    }

    // Generates a preamble containing code that will be shared by
    // all loot finding / loot cracking kernels:
    // 1. Includes for common CUDA / C++ headers. 
    // 2. GPU error handling macro 
    // 3. __managed__ memory storage for found loot seeds.
    // 4. TODO __device__ PRNG functions for Java Random and Xoroshiro.
    void KernelTemplate::generatePreamble(std::ostream& out) const {
        // 1. Includes for common CUDA / C++ headers.
        out <<  "#include \"cuda_runtime.h\"\n"
                "#include \"device_launch_parameters.h\"\n"
                "#include <cinttypes>\n"
                "#include <cstdlib>\n"
                "#include <cstdint>\n"
                "#include <cstdio>\n"
                "#include <chrono>\n\n";

        // 2. GPU error handling macro.
        out <<  "#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)\n"
                "inline void gpuAssert(cudaError_t code, const char* file, int line) {\n"
                "    if (code != cudaSuccess) {\n"
                "        fprintf(stderr, \"GPU ERROR: %s (code %d) in file %s, line %d\n\", cudaGetErrorString(code), code, file, line);\n"
                "        exit(code);\n"
                "    }\n"
                "}\n\n";

        // 3. __managed__ memory storage for found loot seeds.
        out <<  "constexpr size_t RESULT_ARRAY_CAPACITY = " << KernelTemplate::DEFAULT_RESULT_ARRAY_CAPACITY << "\n"
                "__managed__ uint64_t results[RESULT_ARRAY_CAPACITY];\n"
                "__managed__ uint32_t resultCount;\n\n";

        // 4. __device__ PRNG functions.
        // TODO there's going to be a lot of PRNG code to be added here; 
        // perhaps it should be stored in a separate file?
        // or perhaps the entire preamble should be stored in a separate file?
    }

    void KernelTemplate::generateLootLookupTable(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateLootProcessors(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateKernelHeader(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateKernelBody(std::ostream& out) const {
        // TODO
    }

    void KernelTemplate::generateHostController(std::ostream& out) const {
        // TODO
    }
}
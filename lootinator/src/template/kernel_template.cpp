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
    // 4. __device__ PRNG functions for Java Random (Xoroshiro support can come in the future).
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
                "        std::fprintf(stderr, \"GPU ERROR: %s (code %d) in file %s, line %d\n\", cudaGetErrorString(code), code, file, line);\n"
                "        std::exit(code);\n"
                "    }\n"
                "}\n\n";

        // 3. __managed__ memory storage for found loot seeds.
        out <<  "constexpr size_t RESULT_ARRAY_CAPACITY = " << KernelTemplate::DEFAULT_RESULT_ARRAY_CAPACITY << ";\n"
                "__managed__ uint64_t results[RESULT_ARRAY_CAPACITY];\n"
                "__managed__ uint32_t resultCount = 0;\n\n";

        // 4. __device__ PRNG functions.
        out <<  "__device__ inline void setSeed(uint64_t* rand, uint64_t value){ *rand = (value ^ 0x5deece66d) & ((1ULL << 48) - 1); }\n"
                "__device__ inline int next(uint64_t* rand, const int bits){ *rand = (*rand * 0x5deece66d + 0xb) & ((1ULL << 48) - 1); return (int)((int64_t)*rand >> (48 - bits)); }\n"
                "__device__ inline int nextInt(uint64_t* rand, const int n){ if ((n-1 & n) == 0) {uint64_t x = n * (uint64_t)next(rand, 31); return (int)((int64_t)x >> 31);} else {return (int)(next(rand, 31) % n);} }\n"
                "__device__ inline float nextFloat(uint64_t* rand){ return next(rand, 24) / (float)(1 << 24); }\n\n";
    }

    void KernelTemplate::generateLootLookupTable(std::ostream& out) const {
        // TODO
        // Copy the precomputed loot table entry indices and place them in device (or managed) memory
    }

    void KernelTemplate::generateLootProcessors(std::ostream& out) const {
        // TODO
        // Based on the loot functions listed in the computed loot table, generate CUDA
        // __device__ function representations of all the functions.
    }

    void KernelTemplate::generateKernelHeader(std::ostream& out) const {
        // TODO
        // Generate a kernel header - create unique thread index, copy necessary data to __shared__ memory.
        // Which of the data used by kernels should get copied to shared memory like this?
    }

    void KernelTemplate::generateHostController(std::ostream& out) const {
        // TODO
        // Generate host-side code for launching the kernel in approppriate batches. The host controller
        // will provide users program state and progress information. 
    }
}
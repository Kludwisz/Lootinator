#include "lootinator/template/kernel_template.h"

namespace loot {
    KernelTemplate::KernelTemplate(const TemplateParameters& params) : Template(params) {}

    void KernelTemplate::generate(std::ostream& out) const {
        generate_preamble(out);
        generate_loot_lookup_table(out);
        generate_loot_processors(out);
        generate_kernel_header(out);
        generate_kernel_body(out);
        generate_kernel_launch_information(out);
        generate_host_controller(out);
    }

    // Generates a preamble containing code that will be shared by
    // all loot finding / loot cracking kernels:
    // 1. Includes for common CUDA / C++ headers. 
    // 2. GPU error handling macro 
    // 3. __managed__ memory storage for found loot seeds.
    // 4. __device__ PRNG functions for Java Random (Xoroshiro support can come in the future).
    void KernelTemplate::generate_preamble(std::ostream& out) const {
        // 1. Includes for common CUDA / C++ headers.
        out <<  "#include \"cuda_runtime.h\"\n"
                "#include \"device_launch_parameters.h\"\n"
                "#include <cinttypes>\n"
                "#include <cstdlib>\n"
                "#include <cstdint>\n"
                "#include <ostream>\n"
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

    void KernelTemplate::generate_loot_lookup_table(std::ostream& out) const {
        // TODO
        // Copy the precomputed loot table entry indices and place them in device (or managed) memory
    }

    void KernelTemplate::generate_loot_processors(std::ostream& out) const {
        // TODO
        // Based on the loot functions listed in the computed loot table, generate CUDA
        // __device__ function representations of all the functions.
    }

    void KernelTemplate::generate_kernel_header(std::ostream& out) const {
        // TODO
        // Generate a kernel header - create unique thread index, copy necessary data to __shared__ memory.
        // Which of the data used by kernels should get copied to shared memory like this?
    }

    // Generates kernel configuration constants, such as the number of threads per block,
    // or number of batches in the entire seedspace.
    void KernelTemplate::generate_kernel_launch_information(std::ostream& out) const {
        out <<  "constexpr uint32_t NUM_BATCHES = 1U << 16;\n"
                "constexpr uint64_t THREADS_PER_BATCH = 1ULL << 32;\n"
                "constexpr uint32_t THREADS_PER_BLOCK = 256;\n"
                "constexpr uint32_t NUM_BLOCKS = THREADS_PER_BATCH / THREADS_PER_BLOCK;\n";
    }

    // Generates host-side code for launching the kernel in approppriate batches. 
    // The controller additionally provides progress information and estimated time of completion. 
    void KernelTemplate::generate_host_controller(std::ostream& out) const {
        out <<  "int main() {\n"
                "    GPU_ASSERT(cudaSetDevice(0));\n"
                "    for (uint32_t batch_no = 0; batch_no < NUM_BATCHES) {\n"
                "        const auto t1 = std::chrono::steady_clock::now();\n"
                "        " << kernel_name << " <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (THREADS_PER_BATCH * batch_no);\n"
                "        GPU_ASSERT(cudaDeviceSynchronize());\n"
                "        for (uint32_t i = 0; i < result_count; i++) {\n"
                "            std::cout << results[i] << \'\n\';\n"
                "        }\n"
                "        std::cout << std::flush;\n"
                "        const auto t2 = std::chrono::steady_clock::now();\n"
                "        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);\n"
                "        const double eta_minutes = (double)elapsed_ms * (NUM_BATCHES - 1 - batch_no) / 60000.0;\n"
                "        std::cerr << \"ETA: \" << eta_minutes << \" minutes. Tasks done: \" << (batch_no+1) << \"/\" << (NUM_BATCHES+1) << std::endl;\n";
                "    }\n"
                "    GPU_ASSERT(cudaDeviceReset());\n"
                "    return 0;\n"
                "}\n";
    }
}
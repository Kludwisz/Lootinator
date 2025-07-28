#ifndef LAUNCHER_DATA
#define LAUNCHER_DATA

#include <string>
#include <vector>

namespace launcher {
    // in nvrtc-compiled code, the stdint.h header is not directly available, 
    // so this approach should be better for portability across standard systems
    typedef unsigned long long u64;
    typedef unsigned int u32;
    typedef int i32;

    // these can be command line args if necessary
    constexpr const char* BENCHMARK_RESULTS_FILE = "benchmark_results.txt";
    constexpr const char* SOURCE_CODE_OUTPUT_FILE = "source.cu";

    constexpr i32 UNSPECIFIED = -1;
    constexpr u32 RESULT_BUFFER_SIZE = 16u * 1024u; // max results per kernel launch

    enum AppMode {
        NONE,
        BENCHMARK,
        RUN_SINGLE
    };

    struct AppParameters {
        AppMode mode;
        bool debug_info;
        bool generate_cuda_source;
    };

    struct LaunchParameters {
        // internal
        std::string kernel_name;
        std::string kernel_source_file;
        std::string kernel_code; // can be either PTX for NVRTC-based runs or CUDA for source code gen mode
        std::vector<u32> kernel_shared_memory;
        u64 threads_total;
        u64 threads_per_batch;

        // user-provided
        u32 threads_per_block;
        u32 device_id;
        
        // non-kernel
        i32 start_batch;
        i32 end_batch;
    };
}

#endif
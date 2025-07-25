/*
This file will be compiled as a standalone application.
It's going to be an executable that reads PTX code obtained using nvrtc
and launches that as a kernel with additional (user-provided?)
parameters, such as # of threads, device ID, etc.
*/

#include <cuda.h>  // driver API
#include <nvrtc.h> // runtime compilation API

#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <string>
#include <vector>

#include <iostream>
#include <sstream>
#include <fstream>


namespace launcher {
    constexpr int32_t UNSPECIFIED = -1;
    constexpr uint32_t RESULT_BUFFER_SIZE = UINT32_C(16) * 1024;
    constexpr uint32_t SHARED_MEM_SIZE = UINT32_C(8) * 1024;

    #define CUDA_CHECK(ans) { launcher::gpuAssert((ans), __FILE__, __LINE__); }
    void gpuAssert(CUresult code, const char *file, int line) {
        if (code != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(code, &errStr);
            std::cerr << "CUDA error: " << errStr << " at " << file << ":" << line << std::endl;
            exit(1);
        }
    }

    #define NVRTC_CHECK(ans) { launcher::nvrtcAssert((ans), __FILE__, __LINE__); }
    void nvrtcAssert(nvrtcResult res, const char* file, int line) {
        if (res != NVRTC_SUCCESS) {
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(res) << " at " << file << ":" << line << std::endl;
            exit(1);
        }
    }

    struct LaunchParameters {
        // internal
        std::string kernel_name;
        std::string kernel_ptx;
        std::vector<uint32_t> kernel_shared_memory;
        uint64_t threads_total;
        uint64_t threads_per_batch;

        // user-provided
        uint32_t threads_per_block;
        uint32_t device_id;
        
        // non-kernel
        int32_t start_batch;
        int32_t end_batch;
    };

    // Uses cuda driver api to turn the provided parameters into a functional kernel and launches it.
    int launch_kernel(const LaunchParameters& config) {
        // assert that grid size is ok
        const uint32_t n_blocks = static_cast<uint32_t>(config.threads_per_batch / config.threads_per_block);
        const uint64_t n_blocks_long = config.threads_per_batch / config.threads_per_block;
        if (static_cast<uint64_t>(n_blocks) != n_blocks_long) {
            std::cerr << "Fatal error (launch_kernel): number of blocks exceeded 2^32 for provided thread block size: " << config.threads_per_block << std::endl;
            return 1;
        }

        CUDA_CHECK(cuInit(0));
        CUdevice device;
        CUDA_CHECK(cuDeviceGet(&device, config.device_id));
        CUcontext context;
        CUDA_CHECK(cuCtxCreate(&context, 0, device));
        CUmodule module;
        CUDA_CHECK(cuModuleLoadData(&module, config.kernel_ptx.c_str()));
        CUfunction kernel;
        CUDA_CHECK(cuModuleGetFunction(&kernel, module, config.kernel_name.c_str()));

        CUdeviceptr d_result_array, d_result_count;
        CUdeviceptr d_shared_mem_contents, d_shared_mem_contents_length;
        // note the use of 'unsigned long long' and 'unsigned int' instead of uint64_t and uint32_t:
        // in nvrtc-compiled code, the stdint.h header is not directly available, so this approach
        // should be better for portability across standard systems
        CUDA_CHECK(cuMemAlloc(&d_result_array, RESULT_BUFFER_SIZE * sizeof(unsigned long long)));
        CUDA_CHECK(cuMemAlloc(&d_result_count, sizeof(unsigned int)));
        CUDA_CHECK(cuMemAlloc(&d_shared_mem_contents, config.kernel_shared_memory.size() * sizeof(unsigned int)));
        CUDA_CHECK(cuMemAlloc(&d_shared_mem_contents_length, sizeof(unsigned int)));

        for (int32_t batch = config.start_batch; batch < config.end_batch; batch++) {
            // __global__ kernel_name(u64* result_array, u32* result_count, u32* shared_mem_contents, u32 shared_mem_contents_length, u64 offset)
            uint64_t thread_offset = batch * config.threads_per_batch;
            void* kernel_args[] = {
                &d_result_array,
                &d_result_count,
                &d_shared_mem_contents,
                &d_shared_mem_contents_length,
                &thread_offset
            };

            CUDA_CHECK(cuLaunchKernel(
                kernel,
                n_blocks, 1, 1, // grid size
                config.threads_per_block, 1, 1, // block size
                launcher::SHARED_MEM_SIZE,
                NULL, // use current context to launch
                kernel_args,
                NULL // no extra parameters
            ));

            CUDA_CHECK(cuCtxSynchronize());
        }

        // Cleanup
        cuMemFree(d_result_array);
        cuMemFree(d_result_count);
        cuMemFree(d_shared_mem_contents);
        cuMemFree(d_shared_mem_contents_length);
        cuModuleUnload(module);
        cuCtxDestroy(context);

        return 0;
    }

    int compile_nvrtc(const char* source_filename, LaunchParameters& config) {
        std::ifstream file(source_filename);
        if (!file) {
            std::cerr << "Fatal error (compile_nvrtc): Failed to open source code file '" << source_filename << "'." << std::endl;
            return 1;
        }

        std::ostringstream ss;
        ss << file.rdbuf();
        if (file.fail() && !file.eof()) {
            std::cerr << "Fatal error (compile_nvrtc): Failed while reading source code file '" << source_filename << "'." << std::endl;
            return 1;
        }
        std::string kernel_source = ss.str();

        nvrtcProgram prog;
        NVRTC_CHECK(nvrtcCreateProgram(&prog, kernel_source.c_str(), "internal.cu", 0, nullptr, nullptr));
        NVRTC_CHECK(nvrtcCompileProgram(prog, 0, nullptr));
        size_t ptxSize;
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
        std::string ptx(ptxSize, '\0');
        NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
        nvrtcDestroyProgram(&prog);

        config.kernel_ptx = ptx;
        return 0;
    }

    int read_shared_memory(const char* shared_mem_filename, LaunchParameters& config) {
        std::ifstream file(shared_mem_filename);
        if (!file) {
            std::cerr << "Fatal error (read_shared_memory): Failed to open file '" << shared_mem_filename << "'." << std::endl;
            return 1;
        }

        uint32_t val;
        while (file >> val) {
            config.kernel_shared_memory.push_back(val);
        }
        return 0;
    }

    int parse_args(int argc, char** argv, LaunchParameters& config) {
        bool have_name = false, have_ptx = false, have_shared = false;

        for (int i = 1; i < argc-1; i++) {
            if (strcmp(argv[i], "--kernel-name") == 0) {
                config.kernel_name = std::string(argv[i+1]);
                have_name = true;
            }
            else if (strcmp(argv[i], "--kernel-source") == 0) {
                if (compile_nvrtc(argv[i+1], config))
                    return 1;
                have_ptx = true;
            }
            else if (strcmp(argv[i], "--shared-memory") == 0) {
                if (read_shared_memory(argv[i+1], config))
                    return 1;
                have_shared = true;
            }
            else if (strcmp(argv[i], "--threads-total") == 0) {
                sscanf(argv[i+1], "%" PRIu64, &(config.threads_total));
                i++;
            }
            else if (strcmp(argv[i], "--threads-per-batch") == 0) {
                sscanf(argv[i+1], "%" PRIu64, &(config.threads_per_batch));
                i++;
            }
            else if (strcmp(argv[i], "--threads-per-block") == 0) {
                sscanf(argv[i+1], "%" PRIu32, &(config.threads_per_block));
                i++;
            }
            else if (strcmp(argv[i], "--device-id") == 0) {
                sscanf(argv[i+1], "%" PRIu32, &(config.device_id));
                i++;
            }
            else if (strcmp(argv[i], "--start-batch") == 0) {
                sscanf(argv[i+1], "%" PRId32, &(config.start_batch));
                i++;
            }
            else if (strcmp(argv[i], "--end-batch") == 0) {
                sscanf(argv[i+1], "%" PRId32, &(config.end_batch));
                i++;
            }
        }

        return (have_ptx && have_name && have_shared) ? 0 : 1;
    }
};


int main(int argc, char** argv) {
    // default config
    launcher::LaunchParameters config {
        std::string(), // kernel_name
        std::string(), // kernel_ptx
        std::vector<uint32_t>(), // kernel_shared_memory
        UINT64_C(1) << 48, // threads_total
        UINT64_C(1) << 32, // threads_per_batch
        256, // threads per block
        0,   // device id
        launcher::UNSPECIFIED, // (optional) start batch
        launcher::UNSPECIFIED  // (optional) end batch
    };

    if (launcher::parse_args(argc, argv, config)) {
        std::cerr << "Fatal error: kernel name or source code file missing or invalid." << std::endl;
        return 1;
    }

    // calculate or keep provided start and end batches.
    if (config.start_batch == launcher::UNSPECIFIED || config.start_batch < 0) {
        std::cerr << "Start batch unspecified or too small, defaulting to start-batch=0" << std::endl;
        config.start_batch = 0;
    }
    const int32_t end_exclusive = (config.threads_total + config.threads_per_batch - 1) / config.threads_per_batch;
    if (config.end_batch == launcher::UNSPECIFIED || config.end_batch > end_exclusive) {
        std::cerr << "End batch unspecified or too large, defaulting to end-batch=" << end_exclusive << std::endl;
        config.end_batch = end_exclusive;
    }

    return launch_kernel(config);
}

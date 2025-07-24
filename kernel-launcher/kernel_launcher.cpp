/*
This file will be compiled as a standalone application.
It's going to be an executable that reads PTX code obtained using nvrtc
and launches that as a kernel with additional (user-provided?)
parameters, such as # of threads, device ID, etc.
*/

#include <cuda.h> // driver API
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <string>

#include <iostream>
#include <sstream>
#include <fstream>


namespace launcher {
    constexpr int32_t UNSPECIFIED = -1;
    constexpr uint32_t RESULT_BUFFER_SIZE = UINT32_C(16) * 1024;

    #define CUDA_CHECK(ans) { launcher::gpuAssert((ans), __FILE__, __LINE__); }
    void gpuAssert(CUresult code, const char *file, int line) {
        if (code != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(code, &errStr);
            std::cerr << "CUDA error: " << errStr << " at " << file << ":" << line << std::endl;
            exit(1);
        }
    }

    struct LaunchParameters {
        // internal
        std::string kernel_name;
        std::string kernel_ptx;
        uint64_t threads_total;
        uint64_t threads_per_batch;
        // user-provided
        uint32_t threads_per_block;
        uint32_t device_id;
        // non-kernel
        int32_t start_batch;
        int32_t end_batch;
    };

    int launch_kernel(const LaunchParameters& config) {
        // TODO
        // Use cuda driver api to turn the bunch of provided parameters
        // into a functional kernel, then launch it with the provided
        // configuration parameters.

        CUDA_CHECK(cuInit(0));
        CUdevice device;
        CUDA_CHECK(cuDeviceGet(&device, config.device_id));
        CUcontext context;
        CUDA_CHECK(cuCtxCreate(&context, 0, device));
        CUmodule module;
        CUDA_CHECK(cuModuleLoadData(&module, config.kernel_ptx.c_str()));
        CUfunction kernel;
        CUDA_CHECK(cuModuleGetFunction(&kernel, module, config.kernel_name));

        CUdeviceptr d_result_array, d_result_count;
        CUDA_CHECK(cuMemAlloc(&d_result_array, RESULT_BUFFER_SIZE * sizeof(uint64_t)));
        CUDA_CHECK(cuMemAlloc(&d_result_count, sizeof(uint32_t)));

        // Cleanup
        cuMemFree(d_result_array);
        cuMemFree(d_result_count);
        cuCtxDestroy(context);

        return 0;
    }

    int read_ptx(const char* ptx_filename, LaunchParameters& config) {
        std::ifstream file(ptx_filename);
        if (!file) {
            std::cerr << "Fatal error: Failed to open file '" << ptx_filename << "'." << std::endl;
            return 1;
        }

        std::ostringstream ss;
        ss << file.rdbuf();
        if (file.fail() && !file.eof()) {
            std::cerr << "Fatal error: Failed while reading file '" << ptx_filename << "'." << std::endl;
            return 1;
        }
        config.kernel_ptx = ss.str();
        return 0;
    }

    int parse_args(int argc, char** argv, LaunchParameters& config) {
        bool have_name = false, have_ptx = false;

        for (int i = 1; i < argc-1; i++) {
            if (strcmp(argv[i], "--kernel-name") == 0) {
                config.kernel_name = std::string(argv[i+1]);
                have_name = true;
            }
            else if (strcmp(argv[i], "--kernel-ptx") == 0) {
                if (read_ptx(argv[i+1], config))
                    return 1;
                have_ptx = true;
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

        return have_ptx && have_name ? 0 : 1;
    }
};


int main(int argc, char** argv) {
    // default config
    launcher::LaunchParameters config {
        std::string(), // kernel_name
        std::string(), // kernel_ptx
        UINT64_C(1) << 48, // threads_total
        UINT64_C(1) << 32, // threads_per_batch
        256, // threads per block
        0,   // device id
        launcher::UNSPECIFIED, // (optional) start batch
        launcher::UNSPECIFIED  // (optional) end batch
    };

    if (launcher::parse_args(argc, argv, config)) {
        std::cerr << "Fatal error: kernel name or PTX file invalid or not provided." << std::endl;
        return 1;
    }

    // calculate or keep provided start and end batches.
    if (config.start_batch == launcher::UNSPECIFIED || config.start_batch < 0) {
        std::cerr << "Start batch unspecified or too small, defaulting to start-batch=0" << std::endl;
        config.start_batch = 0;
    }
    const int32_t end_exclusive = (config.threads_total + config.threads_per_batch - 1) / config.threads_per_batch;
    if (config.start_batch == launcher::UNSPECIFIED || config.end_batch > end_exclusive) {
        std::cerr << "End batch unspecified or too large, defaulting to end-batch=" << end_exclusive << std::endl;
        config.end_batch = end_exclusive;
    }

    return launch_kernel(config);
}

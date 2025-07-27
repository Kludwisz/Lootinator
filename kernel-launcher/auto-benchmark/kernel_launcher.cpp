/*
This file will be compiled as a standalone application.
It's going to be an multi-feature executable that will enable:
- automated benchmarking of a provided list of kernels (core feature)
- launching any single kernel with provided parameters (feature of simple launcher)
- exporting either a single kernel or a fully automated benchmark-based
  kernel runner as CUDA source files (additional feature, for remote computing)
- (extra) launching any single kernel with workload split evenly(?) across several GPUs
*/

#include <cuda.h>  // driver API
#include <nvrtc.h> // runtime compilation API

#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include <iostream>
#include <sstream>
#include <fstream>


namespace launcher {
    // in nvrtc-compiled code, the stdint.h header is not directly available, 
    // so this approach should be better for portability across standard systems
    typedef unsigned long long u64;
    typedef unsigned int u32;
    typedef int i32;

    constexpr i32 UNSPECIFIED = -1;
    constexpr u32 RESULT_BUFFER_SIZE = 16u * 1024u; // max results per kernel launch
    constexpr u32 WARMUP_LAUNCH_COUNT = 2u;   // number of runs before actual benchmarking starts
    constexpr u32 BENCHMARK_BATCH_SCALE = 4u; // real batch size is this times bigger than test batch size
    constexpr float BENCHMARK_MAX_ELAPSED_TIME = 100.0f; // elapsed milliseconds of single run that stop the benchmark

    #define CUDA_CHECK(ans) do { launcher::gpuAssert((ans), __FILE__, __LINE__); } while(false)
    void gpuAssert(CUresult code, const char *file, int line) {
        if (code != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(code, &errStr);
            std::cerr << "CUDA error: " << errStr << " at " << file << ":" << line << std::endl;
            exit(1);
        }
    }

    #define NVRTC_CHECK(ans) do { launcher::nvrtcAssert((ans), __FILE__, __LINE__); } while(false)
    void nvrtcAssert(nvrtcResult res, const char* file, int line) {
        if (res != NVRTC_SUCCESS) {
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(res) << " at " << file << ":" << line << std::endl;
            exit(1);
        }
    }

    #define DEBUG(bool_var) if (bool_var) std::cerr << "Debug: "

    enum AppMode {
        NONE,
        BENCHMARK,
        RUN_SINGLE
    }

    struct AppParameters {
        AppMode mode;
        bool debug_info;
        bool generate_cuda_source;
    }

    struct LaunchParameters {
        // internal
        std::string kernel_name;
        std::string kernel_ptx;
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

    struct KernelData {
        CUdevice device;
        CUcontext context;
        CUmodule module;
        CUfunction kernel;

        CUdeviceptr d_result_array, d_result_count;
        CUdeviceptr d_shared_mem_contents;
        u32 shared_mem_bytes;

        void* kernel_args[5];

        KernelData(const LaunchParameters& config) {
            // initialize all the necessary data structures
            CUDA_CHECK(cuInit(0));
            CUDA_CHECK(cuDeviceGet(&device, config.device_id));
            CUDA_CHECK(cuCtxCreate(&context, 0, device));
            CUDA_CHECK(cuModuleLoadData(&module, config.kernel_ptx.c_str()));
            CUDA_CHECK(cuModuleGetFunction(&kernel, module, config.kernel_name.c_str()));

            CUDA_CHECK(cuMemAlloc(&d_result_array, RESULT_BUFFER_SIZE * sizeof(u64)));
            CUDA_CHECK(cuMemAlloc(&d_result_count, sizeof(u32)));
            CUDA_CHECK(cuMemAlloc(&d_shared_mem_contents, config.kernel_shared_memory.size() * sizeof(u32)));

            u32 h_shared_mem_contents_length = config.kernel_shared_memory.size();
            shared_mem_bytes = h_shared_mem_contents_length * sizeof(u32);
            CUDA_CHECK(cuMemcpyHtoD(d_shared_mem_contents, config.kernel_shared_memory.data(), h_shared_mem_contents_length * sizeof(u32)));

            // __global__ kernel_name(u64* result_array, u32* result_count, u32* shared_mem_contents, u32 shared_mem_contents_length, u64 offset)
            kernel_args = {
                &d_result_array,
                &d_result_count,
                &d_shared_mem_contents,
                &h_shared_mem_contents_length,
                nullptr
            };
        }

        ~KernelData() {
            // Cleanup
            cuMemFree(d_result_array);
            cuMemFree(d_result_count);
            cuMemFree(d_shared_mem_contents);
            cuModuleUnload(module);
            cuCtxDestroy(context);
        }
    }

    struct BenchmarkResults {
        bool success;
        float ms_per_batch;
        float ms_total_estimate;
    };

    int finalize_config(LaunchParameters& config) {
        // assert that grid size is ok
        const launcher::u32 n_blocks = static_cast<launcher::u32>(config.threads_per_batch / config.threads_per_block);
        const launcher::u64 n_blocks_long = config.threads_per_batch / config.threads_per_block;
        if (static_cast<launcher::u64>(n_blocks) != n_blocks_long) {
            std::cerr << "Fatal error (launch_kernel): number of blocks exceeded 2^32 for provided thread block size: " << config.threads_per_block << std::endl;
            return 1;
        }

        // calculate or keep provided start and end batches.
        if (config.start_batch == launcher::UNSPECIFIED || config.start_batch < 0) {
            DEBUG(app_params.debug_info) << "Start batch unspecified or too small, defaulting to start-batch=0" << std::endl;
            config.start_batch = 0;
        }
        launcher::i32 end_exclusive = (config.threads_total + config.threads_per_batch - 1) / config.threads_per_batch;
        if (config.end_batch == launcher::UNSPECIFIED || config.end_batch > end_exclusive) {
            DEBUG(app_params.debug_info) << "End batch unspecified or too large, defaulting to end-batch=" << end_exclusive << std::endl;
            config.end_batch = end_exclusive;
        }
    }

    // Uses cuda driver api to turn the provided parameters into a functional kernel and launches it.
    int launch_kernel(const KernelData& kdata, const LaunchParameters& config, const AppParameters& app_params) {      
        u64 h_result_array[RESULT_BUFFER_SIZE];
        u32 h_shared_mem_contents_length = config.kernel_shared_memory.size();

        for (i32 batch = config.start_batch; batch < config.end_batch; batch++) {
            DEBUG(app_params.debug_info) << "Running batch #" << batch << " of range [" << config.start_batch << ", " << config.end_batch << ")" << std::endl;
            
            // reset result buffer
            u32 h_result_count = 0;
            CUDA_CHECK(cuMemcpyHtoD(kdata.d_result_count, &h_result_count, sizeof(u32)));
            
            u32 n_blocks = static_cast<u32>(config.threads_total / config.threads_per_block);
            u64 thread_offset = batch * config.threads_per_batch;
            kdata.kernel_args[4] = &thread_offset;
            CUDA_CHECK(cuLaunchKernel(
                kdata.kernel,
                n_blocks, 1, 1, // grid size
                config.threads_per_block, 1, 1, // block size
                kdata.shared_mem_bytes,
                NULL, // use current context to launch
                kdata.kernel_args,
                NULL // no extra parameters
            ));
            CUDA_CHECK(cuCtxSynchronize());

            // copy results back to host, print to stdout
            CUDA_CHECK(cuMemcpyDtoH(&h_result_count, kdata.d_result_count, sizeof(u32)));
            CUDA_CHECK(cuMemcpyDtoH(h_result_array, kdata.d_result_array, h_result_count * sizeof(u64)));

            for (u32 i = 0; i < h_result_count; i++) {
                std::cout << h_result_array[i] << '\n';
            }
            std::cout << std::flush;
            DEBUG(app_params.debug_info) << "Got " <<  h_result_count << " results.\n";
        }

        return 0;
    }

    BenchmarkResults benchmark_kernel(const LaunchParameters& config, const AppParameters& app_params) {
        LaunchParameters work_config = config;
        AppParameters work_app_params = app_params;
        KernelData kdata(work_config);
        BenchmarkResults results{false, 0.0f, 0.0f};

        const i32 middle_batch = (config.start_batch + config.end_batch) / 2;
        work_config.threads_per_batch /= BENCHMARK_BATCH_SCALE;

        // warmup (to get more accurate measurements)
        work_config.start_batch = middle_batch;
        work_config.end_batch = middle_batch + 1;
        work_app_params.debug_info = false;
        for (u32 i = 0; i < WARMUP_LAUNCH_COUNT; i++) {
            if (launch_kernel(kdata, work_config, work_app_params)) {
                return results;
            }
        }

        // benchmarking with auto-tuning
        i32 batches = 1;
        float elapsed_ms = 0.0f;
        work_app_params.debug_info = app_params.debug_info;
        
        while (elapsed_ms < 100.0f) {
            work_config.end_batch = work_config.start_batch + batches;

            auto t0 = std::chrono::high_resolution_clock::now();
            if (launch_kernel(kdata, work_config, work_app_params)) {
                return results;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            float elapsed_ms = (t1-t0).count() * 1e-6f;
            if (!(elapsed_ms < 100.0f))
                batches *= 2;
        }

        // return the results
        results.success = true;
        results.ms_per_batch = elapsed_ms * BENCHMARK_BATCH_SCALE / batches;
        results.ms_total_estimate = results.ms_per_real_batch * (config.end_batch - config.start_batch);
        return results;
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

        u32 val;
        while (file >> val) {
            config.kernel_shared_memory.push_back(val);
        }
        return 0;
    }

    // bad code but it works
    int parse_args(int argc, char** argv, AppParameters& app_params, std::vector<LaunchParameters>& launch_params) {
        // default app config
        app_config.mode = AppMode::NONE;
        app_config.debug_info = false;
        app_config.generate_cuda_source = false;

        for (int i = 1; i < argc; i++) {
            // flag args
            if (strcmp(argv[i], "--debug") == 0) {
                app_config.debug_info = true;
            }
            else if (strcmp(argv[i], "--get-cuda-source") == 0) {
                app_config.generate_cuda_source = true;
            }
            else if (strcmp(argv[i], "--benchmark") == 0) {
                if (app_params.mode != AppMode::NONE) {
                    std::cerr << "Illegal args: --benchmark and --run-single cannot be specified at once!\n";
                    return 1;
                }
                app_params.mode = AppMode::BENCHMARK;
            }
            else if (strcmp(argv[i], "--run-single") == 0) {
                if (app_params.mode != AppMode::NONE) {
                    std::cerr << "Illegal args: --benchmark and --run-single cannot be specified at once!\n";
                    return 1;
                }
                app_params.mode = AppMode::RUN_SINGLE;
            }
            if (i >= argc-1) break;
            
            // args with value
            if (strcmp(argv[i], "--use-config") == 0) {
                // TODO parse config from json
            }
        }

        if (app_params.mode == AppMode::NONE) {
            std::cerr << "Error: Must specify operation mode: either --benchmark or --run-single\n";
            return 1;
        }
        if (launch_params.empty()) {
            std::cerr << "Error: Invalid or unspecified configuration file. Did you forget to --use-config (filename.json)?\n";
            return 1;
        }

        return 0;
    }
};


int main(int argc, char** argv) {
    std::vector<launcher::LaunchParameters> kernel_params;
    launcher::AppParameters app_params;

    if (launcher::parse_args(argc, argv, app_params, kernel_params)) {
        return 1;
    }

    if (app_params.mode == launcher::AppMode::BENCHMARK) {
        DEBUG(app_params.debug_info) << "selected benchmark mode.\n";
    }
    else { // app mode = run single
        DEBUG(app_params.debug_info) << "selected run-single mode.\n";
    }
}

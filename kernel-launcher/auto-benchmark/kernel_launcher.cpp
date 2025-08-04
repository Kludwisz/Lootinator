/*
This will be compiled as a standalone application.
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
#include <algorithm>

#include <iostream>
#include <sstream>
#include <fstream>

#include "../../external/json/single_include/nlohmann/json.hpp"

#include "launcher_data.h"

namespace launcher {
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
            CUDA_CHECK(cuModuleLoadData(&module, config.kernel_code.c_str()));
            CUDA_CHECK(cuModuleGetFunction(&kernel, module, config.kernel_name.c_str()));

            CUDA_CHECK(cuMemAlloc(&d_result_array, RESULT_BUFFER_SIZE * sizeof(u64)));
            CUDA_CHECK(cuMemAlloc(&d_result_count, sizeof(u32)));
            CUDA_CHECK(cuMemAlloc(&d_shared_mem_contents, config.kernel_shared_memory.size() * sizeof(u32)));

            u32 h_shared_mem_contents_length = config.kernel_shared_memory.size();
            shared_mem_bytes = h_shared_mem_contents_length * sizeof(u32);
            CUDA_CHECK(cuMemcpyHtoD(d_shared_mem_contents, config.kernel_shared_memory.data(), h_shared_mem_contents_length * sizeof(u32)));

            // __global__ kernel_name(u64* result_array, u32* result_count, u32* shared_mem_contents, u32 shared_mem_contents_length, u64 offset)
            kernel_args[0] = &d_result_array;
            kernel_args[1] = &d_result_count;
            kernel_args[2] = &d_shared_mem_contents;
            kernel_args[3] = nullptr;
            kernel_args[4] = nullptr;
        }

        ~KernelData() {
            // Cleanup
            cuMemFree(d_result_array);
            cuMemFree(d_result_count);
            cuMemFree(d_shared_mem_contents);
            cuModuleUnload(module);
            cuCtxDestroy(context);
        }
    };

    struct BenchmarkResults {
        std::string kernel_name;
        bool success;
        float ms_per_batch;
        float ms_total_estimate;
    };
}

namespace launcher {
    extern int generate_benchmarker_source(std::vector<launcher::LaunchParameters> kernel_configs);
    extern int generate_runner_source(launcher::LaunchParameters& kernel_config);
}

namespace launcher {
    int finalize_config(LaunchParameters& config, const AppParameters& app_params) {
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

        return 0;
    }

    // Uses cuda driver api to turn the provided parameters into a functional kernel and launches it.
    int launch_kernel(KernelData& kdata, const LaunchParameters& config, const AppParameters& app_params) {      
        u64 h_result_array[RESULT_BUFFER_SIZE];
        u32 h_shared_mem_contents_length = config.kernel_shared_memory.size();

        for (i32 batch = config.start_batch; batch < config.end_batch; batch++) {
            //DEBUG(app_params.debug_info) << "Running batch #" << batch << " of range [" << config.start_batch << ", " << config.end_batch << ")" << std::endl;
            
            // reset result buffer
            u32 h_result_count = 0;
            CUDA_CHECK(cuMemcpyHtoD(kdata.d_result_count, &h_result_count, sizeof(u32)));
            
            u32 n_blocks = static_cast<u32>(config.threads_per_batch / config.threads_per_block);
            u64 thread_offset = batch * config.threads_per_batch;
            kdata.kernel_args[3] = &h_shared_mem_contents_length;
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
            if (app_params.mode == AppMode::BENCHMARK)
                continue;

            for (u32 i = 0; i < h_result_count; i++) {
                std::cout << h_result_array[i] << '\n';
            }
            std::cout << std::flush;
            //DEBUG(app_params.debug_info) << "Got " <<  h_result_count << " results.\n";
        }

        DEBUG(app_params.debug_info) << "launch_kernel finished. work done: " << config.start_batch << " " << config.end_batch << "\n";
        return 0;
    }

    // Performs an automated benchmark of the provided kernel.
    BenchmarkResults benchmark_kernel(const LaunchParameters& config, const AppParameters& app_params) {
        LaunchParameters work_config = config;
        AppParameters work_app_params = app_params;
        KernelData kdata(work_config);
        BenchmarkResults results{config.kernel_name, false, 0.0f, 0.0f};

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
        DEBUG(app_params.debug_info) << " benchmarking kernel " << config.kernel_name << " - running benchmark";
        
        while (elapsed_ms < 100.0f) {
            work_config.end_batch = work_config.start_batch + batches;

            auto t0 = std::chrono::high_resolution_clock::now();
            if (launch_kernel(kdata, work_config, work_app_params)) {
                return results;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            elapsed_ms = (t1-t0).count() * 1e-6f;
            if (elapsed_ms < 100.0f)
                batches *= 2;
        }

        // return the results
        results.success = true;
        results.ms_per_batch = elapsed_ms * BENCHMARK_BATCH_SCALE / batches;
        results.ms_total_estimate = results.ms_per_batch * (config.end_batch - config.start_batch);
        return results;
    }

    // writes 'time_ms' to 'fout' in a human-friendly format 
    void write_human_readable_time(std::ofstream& fout, const float time_ms) {
        float seconds = time_ms / 1000.0f;
        float minutes = seconds / 60.0f;
        float hours = minutes / 60.0f;
        int full_hours = static_cast<int>(std::floor(hours));
        minutes -= full_hours * 60.0f;
        int full_minutes = static_cast<int>(std::floor(minutes));
        seconds -= full_minutes * 60.0f + full_hours * 3600.0f;
        int full_seconds = static_cast<int>(std::floor(seconds));

        fout << full_hours << " hours, " << full_minutes << " minutes, " << full_seconds << " seconds";
    }

    // Benchmarks all kernels provided in the vector of launch parameters. Prints results to BENCHMARK_RESULTS_FILE.
    void benchmark_all(const AppParameters& app_params, std::vector<launcher::LaunchParameters> kernel_configs) {
        std::vector<BenchmarkResults> result_vec;
        
        for (const auto& config : kernel_configs) {
            DEBUG(app_params.debug_info) << " benchmarking kernel " << config.kernel_name << " - initializing data\n";
            BenchmarkResults res = benchmark_kernel(config, app_params);
            DEBUG(app_params.debug_info) << " benchmarking kernel " << config.kernel_name << " - done\n";

            result_vec.push_back(res);
            if (!res.success) {
                std::cerr << "Warning: Benchmark failed for kernel " << config.kernel_name << ", results will be ignored\n";
            }
        }

        // sort ascending by estimated time to run full seedspace
        std::sort(result_vec.begin(), result_vec.end(), [](BenchmarkResults& l, BenchmarkResults& r){ return l.ms_total_estimate < r.ms_total_estimate; });
        // print results to file
        std::ofstream fout(BENCHMARK_RESULTS_FILE);
        for (const auto& res : result_vec) {
            if (!res.success)
                continue;
            fout << res.kernel_name << ":\n";
            fout << "Estimated batch runtime: " << res.ms_per_batch << "ms (";
            write_human_readable_time(fout, res.ms_per_batch);
            fout << ")\nEstimated total runtime: " << res.ms_total_estimate << "ms (";
            write_human_readable_time(fout, res.ms_total_estimate);
            fout << ")\n\n";
        }

        DEBUG(app_params.debug_info) << " benchmark_all completed.\n";
    }

    // Compiles the CUDA kernel in file 'source_filename' and stores the PTX in the 'config' structure
    // OR stores the CUDA source code in the 'config' structure, depending on whether 'generate_sources' was set.
    int compile_nvrtc(std::string source_filename, LaunchParameters& config, const bool generate_sources) {
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
        if (generate_sources) {
            config.kernel_code = kernel_source;
            return 0;
        }

        nvrtcProgram prog;
        NVRTC_CHECK(nvrtcCreateProgram(&prog, kernel_source.c_str(), "internal.cu", 0, nullptr, nullptr));
        NVRTC_CHECK(nvrtcCompileProgram(prog, 0, nullptr));
        size_t ptxSize;
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
        std::string ptx(ptxSize, '\0');
        NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
        nvrtcDestroyProgram(&prog);

        config.kernel_code = ptx;
        return 0;
    }

    // Parses the shared memory contents from file 'shared_mem_filename'
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
        std::cerr << "Debug: Shared memory size: " << config.kernel_shared_memory.size() << "\n";
        return 0;
    }

    // Parses the config file containing information about all the provided kernels:
    // filepaths of shared memories, kernel sources, total & per-batch thread counts, etc.
    int parse_config_json(const char* config_filename, AppParameters& app_params, std::vector<LaunchParameters>& launch_params) {
        try {
            std::ifstream fin(config_filename);
            nlohmann::json data = nlohmann::json::parse(fin);
            u32 device_id = 0;

            if (data.contains("debug"))
                app_params.debug_info = data["debug"];
            if (data.contains("device_id"))
                device_id = data["device_id"];
            if (data.contains("generate_cuda_source"))
                app_params.generate_cuda_source = data["generate_cuda_source"];

            DEBUG(app_params.debug_info) << "App params parsed succesfully.\n";

            // parsing kernel data (kernels field is required)
            nlohmann::json kernel_data = data["kernels"];
            for (auto& node : kernel_data) {
                LaunchParameters lp;
                lp.start_batch = lp.end_batch = launcher::UNSPECIFIED;
                lp.device_id = device_id;

                // all fields below are required 
                lp.kernel_name = node["kernel_name"];
                lp.threads_total = node["threads_total"];
                lp.threads_per_batch = node["threads_per_batch"];
                lp.threads_per_block = node["threads_per_block"];
                lp.kernel_source_file = node["source_code_filepath"];
                std::string shmem_file = node["shared_memory_filepath"];
                // these two are optional
                if (node.contains("start_batch")) lp.start_batch = node["start_batch"];
                if (node.contains("start_batch")) lp.end_batch = node["end_batch"];

                DEBUG(app_params.debug_info) << "Compiling kernel " <<  lp.kernel_name << "...\n";

                if (read_shared_memory(shmem_file.c_str(), lp)) {
                    std::cerr << "Error: failed to read shared memory file '" << shmem_file << "' for kernel " << lp.kernel_name << "\n";
                    return 1;
                }
                if (compile_nvrtc(lp.kernel_source_file, lp, app_params.generate_cuda_source)) {
                    std::cerr << "Error: failed to compile source code file '" << lp.kernel_source_file << "' for kernel " << lp.kernel_name << "\n";
                    return 1;
                }
                if (finalize_config(lp, app_params)) {
                    return 1;
                }

                launch_params.push_back(lp);
            }
        }
        catch (std::exception ex) {
            std::cerr << ex.what() << "\n";
            return 1;
        }
        return 0;
    }

    // parses command line arguments. bad code but it works
    int parse_args(int argc, char** argv, AppParameters& app_params, std::vector<LaunchParameters>& launch_params) {
        // default app config
        app_params.mode = AppMode::NONE;
        app_params.debug_info = false;
        app_params.generate_cuda_source = false;

        for (int i = 1; i < argc; i++) {
            // flag args
            if (strcmp(argv[i], "--benchmark") == 0) {
                if (app_params.mode == AppMode::RUN_SINGLE) {
                    std::cerr << "Error: Illegal args: --benchmark and --run-single cannot be specified at once!\n";
                    return 1;
                }
                app_params.mode = AppMode::BENCHMARK;
            }
            else if (strcmp(argv[i], "--run-single") == 0) {
                if (app_params.mode == AppMode::BENCHMARK) {
                    std::cerr << "Error: Illegal args: --benchmark and --run-single cannot be specified at once!\n";
                    return 1;
                }
                app_params.mode = AppMode::RUN_SINGLE;
            }
            if (i >= argc-1) break;
            
            // args with value
            if (strcmp(argv[i], "--use-config") == 0) {
                if (parse_config_json(argv[i+1], app_params, launch_params)) {
                    std::cerr << "Error: Invalid configuration file: " << argv[i+1] << ".\n";
                    return 1;
                }
            }
        }

        if (app_params.mode == AppMode::NONE) {
            std::cerr << "Error: Must specify operation mode: either --benchmark or --run-single\n";
            return 1;
        }
        if (launch_params.empty()) {
            std::cerr << "Error: Unspecified configuration file. Did you forget to --use-config (filename.json)?\n";
            return 1;
        }

        return 0;
    }
};


int main(int argc, char** argv) {
    std::vector<launcher::LaunchParameters> kernel_configs;
    launcher::AppParameters app_params;

    if (launcher::parse_args(argc, argv, app_params, kernel_configs)) {
        return 1;
    }

    if (app_params.mode == launcher::AppMode::BENCHMARK) {
        DEBUG(app_params.debug_info) << "selected benchmark mode.\n";

        if (app_params.generate_cuda_source) {
            return launcher::generate_benchmarker_source(kernel_configs);
        }
        else {
            launcher::benchmark_all(app_params, kernel_configs);
            return 0;
        }
    }
    else { // app mode = run single
        DEBUG(app_params.debug_info) << "selected run-single mode.\n";
        launcher::LaunchParameters config = kernel_configs.at(0);
        if (launcher::finalize_config(config, app_params))
            return 1;

        if (app_params.generate_cuda_source) {
            return launcher::generate_runner_source(config);
        }
        else {
            launcher::KernelData kdata(config);
            return launcher::launch_kernel(kdata, config, app_params);
        }
    }
}

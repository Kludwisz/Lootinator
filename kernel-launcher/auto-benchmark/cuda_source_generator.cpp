#include <iostream>
#include <sstream>
#include <fstream>

#include "launcher_data.h"

namespace launcher {
    int generate_benchmarker_source(std::vector<launcher::LaunchParameters> kernel_configs);
    int generate_runner_source(launcher::LaunchParameters& kernel_config);

    void print_preamble(std::ostream& out) {
        out << 
R"(#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <chrono>
#include <cstdint>
#include <cinttypes>
#include <iostream>

typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u64;
typedef int64_t i64;

constexpr u64 JRAND_MULTIPLIER = 0x5deece66d;
constexpr u64 MASK_48 = ((1ULL << 48) - 1);

#define CUDA_CHECK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(false)
void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

__device__ inline void setSeed(u64* rand, u64 value){ *rand = (value ^ JRAND_MULTIPLIER) & MASK_48; }
__device__ inline int next(u64* rand, const int bits){ *rand = (*rand * JRAND_MULTIPLIER + 11) & MASK_48; return (int)((i64)*rand >> (48 - bits)); }
__device__ inline int nextInt(u64* rand, const int n){ if ((n-1 & n) == 0) {u64 x = n * (u64)next(rand, 31); return (int)((i64)x >> 31);} else {return (int)(next(rand, 31) % n);} }
__device__ inline float nextFloat(u64* rand){ return next(rand, 24) / (float)(1 << 24); }
)" << "\n\n";
    }

    void print_shared_mem(std::ostream& out, const launcher::LaunchParameters& kernel_config) {
        out << "{";
        const std::vector<u32>& shmem = kernel_config.kernel_shared_memory;
        for (size_t i = 0; i < shmem.size(); i++) {
            out << kernel_config.kernel_shared_memory[i] << (i == shmem.size()-1 ? "};" : ",");
        }
    }

    void print_kernels(std::ostream& out, const std::vector<launcher::LaunchParameters>& kernel_configs) {
        for (int k = 0; k < kernel_configs.size(); k++) {
            // each kernel gets its own namespace to avoid device helper conflicts
            out << "namespace kernel" << k << " {\n";
            // shared header is stripped, for standard CUDA another header will be used
            // and we don't want to declare it for each kernel individually 
            size_t header_end = kernel_configs[k].kernel_code.find(HEADER_END_INDICATOR) + HEADER_END_INDICATOR_LENGTH;
            out << kernel_configs[k].kernel_code.substr(header_end + 1);
            out << "\n} //namespace\n";
        }
    }

    void print_kernel_launchers(std::ostream& out, const std::vector<launcher::LaunchParameters>& kernel_configs) {
        for (int k = 0; k < kernel_configs.size(); k++) {
            const auto& conf = kernel_configs.at(k);
            out << "namespace kernel" << k << " {";
            out << 
R"(
void launch(
    const uint32_t num_blocks, const uint32_t threads_per_block, const uint32_t shared_mem_bytes,
    uint64_t* result_array, uint32_t* result_count,
    uint32_t* shared_mem_contents, uint32_t shared_mem_contents_length, 
    uint64_t offset) 
{
    )" << conf.kernel_name << R"(<<< num_blocks, threads_per_block, shared_mem_bytes >>> (
        result_array, result_count, shared_mem_contents, shared_mem_contents_length, offset
    );
}} //namespace
)";
        }
    }

    void print_benchmarker(std::ostream& out, const std::vector<launcher::LaunchParameters>& kernel_configs) {
        out << 
R"(
typedef void (*launch_function)(uint32_t, uint32_t, uint32_t, uint64_t*, uint32_t*, uint32_t*, uint32_t, uint64_t);

int main() {
    std::vector<launch_function> launchers;
    std::vector<std::vector<uint32_t>> shared_mems;
)";

        for (int i = 0; i < kernel_configs.size(); i++) {
            out << "    launchers.push_back(kernel" << i << "::launch); shared_mems.push_back(std::vector<uint32_t>());\n";
        }
        for (int i = 0; i < kernel_configs.size(); i++) {
            out << "    {uint32_t shmem[] = ";
            print_shared_mem(out, kernel_configs.at(i));
            out << " for (int k = 0; k < " << kernel_configs.at(i).kernel_shared_memory.size() 
                << "; k++) shared_mems[" << i << "].push_back(shmem[k]);}\n";
        }

        // TODO for each kernel do a full cuda device reset, synchronize.
        // Then set up the result buffer, shared memory buffer, etc,
        // launch the kernel repeatedly like the precompiled version of the
        // benchmarker does.
        // Store the estimated kernel performances, then find the kernel with
        // best performance and launch it (informing the user which kernel
        // was chosen!!!)

        out << "    return 0;\n}\n";
    }

    int generate_runner_source(launcher::LaunchParameters& kernel_config) {
        std::vector<launcher::LaunchParameters> single_kernel;
        single_kernel.push_back(kernel_config);

        std::ofstream fout(launcher::SOURCE_CODE_OUTPUT_FILE);
        print_preamble(fout);
        print_kernels(fout, single_kernel);
        print_kernel_launchers(fout, single_kernel);

        fout << "int main() { return kernel0::launch(); }"; // TODO unfinished still, need to set up kernel data
        return 0;
    }

    int generate_benchmarker_source(std::vector<launcher::LaunchParameters> kernel_configs) {
        std::ofstream fout(launcher::SOURCE_CODE_OUTPUT_FILE);
        print_preamble(fout);
        print_kernels(fout, kernel_configs);
        print_kernel_launchers(fout, kernel_configs);

        print_benchmarker(fout, kernel_configs);
        return 0;
    }
}
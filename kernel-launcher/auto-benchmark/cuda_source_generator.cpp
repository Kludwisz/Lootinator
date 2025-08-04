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
#include <algorithm>
#include <cstdint>
#include <cinttypes>

typedef uint32_t u32;
typedef int32_t i32;
typedef uint64_t u64;
typedef int64_t i64;

constexpr u64 JRAND_MULTIPLIER = 0x5deece66d;
constexpr u64 MASK_48 = ((1ULL << 48) - 1);

#define CUDA_CHECK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while(false)
void gpuAssert(CUresult code, const char *file, int line) {
    if (code != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(code, &errStr);
        std::cerr << "CUDA error: " << errStr << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

__device__ inline void setSeed(u64* rand, u64 value){ *rand = (value ^ JRAND_MULTIPLIER) & MASK_48; }
__device__ inline int next(u64* rand, const int bits){ *rand = (*rand * JRAND_MULTIPLIER + 11) & MASK_48; return (int)((i64)*rand >> (48 - bits)); }
__device__ inline int nextInt(u64* rand, const int n){ if ((n-1 & n) == 0) {u64 x = n * (u64)next(rand, 31); return (int)((i64)x >> 31);} else {return (int)(next(rand, 31) % n);} }
__device__ inline float nextFloat(u64* rand){ return next(rand, 24) / (float)(1 << 24); }
)";
    }

    void print_shared_mem(std::ostream& out, const launcher::LaunchParameters& kernel_config) {
        out << "{";
        const std::vector<u32>& shmem = kernel_config.kernel_shared_memory;
        for (size_t i = 0; i < shmem.size(); i++) {
            out << kernel_config.kernel_shared_memory[i] << (i == shmem.size()-1 ? "};\n" : ",");
        }
    }

    void print_kernels(std::ostream& out, const std::vector<launcher::LaunchParameters>& kernel_configs) {
        for (int k = 0; k < kernel_configs.size(); k++) {
            // each kernel gets its own namespace to avoid device helper conflicts
            out << "namespace kernel" << k << " {\n";
            out << kernel_configs[k].kernel_code;
            out << "} //namespace\n";
        }
    }

    void print_kernel_launchers(std::ostream& out, const std::vector<launcher::LaunchParameters>& kernel_configs) {
        // TODO generate kernelN::launch for each provided kernel
    }

    void print_benchmarker(std::ostream& out, const std::vector<launcher::LaunchParameters>& kernel_configs) {
        // TODO generate benchmarker: vector of function pointers to kernel run functions 
    }

    int generate_runner_source(launcher::LaunchParameters& kernel_config) {
        std::vector<launcher::LaunchParameters> single_kernel;
        single_kernel.push_back(kernel_config);

        std::ofstream fout(launcher::SOURCE_CODE_OUTPUT_FILE);
        print_preamble(fout);
        print_kernels(fout, single_kernel);
        print_kernel_launchers(fout, single_kernel);

        fout << "int main() { return kernel0::launch(); }";
        return 1;
    }

    int generate_benchmarker_source(std::vector<launcher::LaunchParameters> kernel_configs) {
        std::ofstream fout(launcher::SOURCE_CODE_OUTPUT_FILE);
        print_preamble(fout);
        print_kernels(fout, kernel_configs);
        print_kernel_launchers(fout, kernel_configs);

        print_benchmarker(fout, kernel_configs);
        return 1;
    }
}
#include "lootinator/template/kernel_template.h"


namespace loot {
    KernelTemplate::KernelTemplate(const TemplateParameters& params, const std::string& kernel_name) : Template(params) {
        this->kernel_name = kernel_name;
    }

    void KernelTemplate::generate(std::ostream& out) const {
        generate_device_helpers(out);
        generate_loot_processors(out);
        generate_kernel_header(out);
        generate_kernel_body(out);
    }

    // Generates PRNG functions for Java Random that will be shared by
    // all loot finding / loot cracking kernels:
    void KernelTemplate::generate_device_helpers(std::ostream& out) const {
        out <<  
R"(__device__ inline void setSeed(uint64_t* rand, uint64_t value){ *rand = (value ^ 0x5deece66d) & ((1ULL << 48) - 1); }
__device__ inline int next(uint64_t* rand, const int bits){ *rand = (*rand * 0x5deece66d + 0xb) & ((1ULL << 48) - 1); return (int)((int64_t)*rand >> (48 - bits)); }
__device__ inline int nextInt(uint64_t* rand, const int n){ if ((n-1 & n) == 0) {uint64_t x = n * (uint64_t)next(rand, 31); return (int)((int64_t)x >> 31);} else {return (int)(next(rand, 31) % n);} }
__device__ inline float nextFloat(uint64_t* rand){ return next(rand, 24) / (float)(1 << 24); }
)";
    }

    void KernelTemplate::generate_loot_processors(std::ostream& out) const {
        // TODO
        // Based on the loot functions listed in the computed loot table, generate CUDA
        // __device__ function representations of all the functions.
    }

    // Generates a kernel header - copies necessary data to __shared__ memory and creates unique thread index tid.
    void KernelTemplate::generate_kernel_header(std::ostream& out) const {
        out << 
R"(
extern "C" {
    typedef unsigned long long u64;
    typedef unsigned int u32;
    typedef int i32;

    __global__ void)" << kernel_name << R"((
        u64* result_array, u32* result_count, 
        u32* shared_mem_contents, u32 shared_mem_contents_length, 
        u64 offset)
    {
        extern __shared__ u32 data[];
        if (threadIdx.x < shared_mem_contents_length) {
            for (int i = threadIdx.x; i < shared_mem_contents_length; i += blockDim.x) {
                data[i] = shared_mem_contents[i];
            }
        }
        __syncthreads();

        const u64 tid = blockIdx.x * blockDim.x + threadIdx.x + offset;
)";
    }

    void KernelTemplate::generate_kernel_terminator(std::ostream& out) const {
        out <<
R"(
    } //end kernel
}     //end extern "C"
)";
    }
}
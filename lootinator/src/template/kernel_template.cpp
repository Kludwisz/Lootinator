#include "lootinator/template/kernel_template.h"

namespace loot {
    KernelTemplate::KernelTemplate(const TemplateParameters& params, const std::string& kernel_name) : Template(params) {
        this->kernel_name = kernel_name;
    }

    void KernelTemplate::generate(std::ostream& out) const {
        //generate_preamble(out);
        //generate_loot_lookup_table(out);
        generate_loot_processors(out);
        generate_kernel_header(out);
        generate_kernel_body(out);
        //generate_kernel_launch_information(out);
        //generate_host_controller(out);
    }

//     // Generates a preamble containing code that will be shared by
//     // all loot finding / loot cracking kernels:
//     // 1. Includes for common CUDA / C++ headers. 
//     // 2. GPU error handling macro 
//     // 3. __managed__ memory storage for found loot seeds.
//     // 4. __device__ PRNG functions for Java Random (Xoroshiro support can come in the future).
//     void KernelTemplate::generate_preamble(std::ostream& out) const {
//         // 1. Includes for common CUDA / C++ headers.
//         out <<  
// R"(#include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include <cinttypes>
// #include <cstdlib>
// #include <cstdint>
// #include <ostream>
// #include <chrono>
// )";

//         // 2. GPU error handling macro.
//         out <<  
// R"(#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
// inline void gpuAssert(cudaError_t code, const char* file, int line) {
//     if (code != cudaSuccess) {
//         std::fprintf(stderr, "GPU ERROR: %s (code %d) in file %s, line %d\n", cudaGetErrorString(code), code, file, line);
//         std::exit(code);
//     }
// }
// )";

//         // 3. __managed__ memory storage for found loot seeds.
//         out << 
// "constexpr size_t RESULT_ARRAY_CAPACITY = " << KernelTemplate::DEFAULT_RESULT_ARRAY_CAPACITY << R"(;
// __managed__ uint64_t results[RESULT_ARRAY_CAPACITY];
// __managed__ uint32_t resultCount = 0;
// )";

//         // 4. __device__ PRNG functions.
//         out <<  
// R"(__device__ inline void setSeed(uint64_t* rand, uint64_t value){ *rand = (value ^ 0x5deece66d) & ((1ULL << 48) - 1); }
// __device__ inline int next(uint64_t* rand, const int bits){ *rand = (*rand * 0x5deece66d + 0xb) & ((1ULL << 48) - 1); return (int)((int64_t)*rand >> (48 - bits)); }
// __device__ inline int nextInt(uint64_t* rand, const int n){ if ((n-1 & n) == 0) {uint64_t x = n * (uint64_t)next(rand, 31); return (int)((int64_t)x >> 31);} else {return (int)(next(rand, 31) % n);} }
// __device__ inline float nextFloat(uint64_t* rand){ return next(rand, 24) / (float)(1 << 24); }
// )";
//     }

//     void KernelTemplate::generate_loot_lookup_table(std::ostream& out) const {
//         // TODO
//         // Copy the precomputed loot table entry indices and place them in device (or managed) memory
//     }

    void KernelTemplate::generate_loot_processors(std::ostream& out) const {
        // TODO
        // Based on the loot functions listed in the computed loot table, generate CUDA
        // __device__ function representations of all the functions.
    }

    // Generates a kernel header - copies necessary data to __shared__ memory and creates unique thread index tid.
    void KernelTemplate::generate_kernel_header(std::ostream& out) const {
        out << 
R"(
__global__ void)" << kernel_name << R"((const uint64_t offset, uint32_t* globalLootData, uint64_t* managedResultArray, uint64_t* managedResultCount)
{
    __shared__ uint32_t lootData[SHARED_MEMORY_ELEMS];
    for (int i = threadIdx.x; i < SHARED_MEMORY_ELEMS; i += blockDim.x)
    lootData[i] = globalLootData[i];
    __syncthreads();
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x + offset;
)";
    }

//     // Generates kernel configuration constants, such as the number of threads per block,
//     // or number of batches in the entire seedspace.
//     void KernelTemplate::generate_kernel_launch_information(std::ostream& out) const {
//         out <<  
// R"(constexpr uint32_t NUM_BATCHES = UINT32_C(1) << 16;
// constexpr uint64_t THREADS_PER_BATCH = UINT64_C(1) << 32;
// constexpr uint32_t THREADS_PER_BLOCK = UINT32_C(256);
// constexpr uint32_t NUM_BLOCKS = THREADS_PER_BATCH / THREADS_PER_BLOCK;
// )";
//     }

//     // Generates host-side code for launching the kernel in approppriate batches. 
//     // The controller additionally provides progress information and estimated time of completion. 
//     void KernelTemplate::generate_host_controller(std::ostream& out) const {
//         out << 
// R"(int main() {
//     GPU_ASSERT(cudaSetDevice(0));
//     for (uint32_t batch_no = 0; batch_no < NUM_BATCHES) {
//         const auto t1 = std::chrono::steady_clock::now();
//         )" << kernel_name << R"( <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (THREADS_PER_BATCH * batch_no);
//         GPU_ASSERT(cudaDeviceSynchronize());
//         for (uint32_t i = 0; i < result_count; i++) {
//             std::cout << results[i] << '\n';
//         }
//         std::cout << std::flush;
//         const auto t2 = std::chrono::steady_clock::now();
//         const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
//         const double eta_minutes = (double)elapsed_ms * (NUM_BATCHES - 1 - batch_no) / 60000.0;
//         std::cerr << "ETA: " << eta_minutes << " minutes. Tasks done: " << (batch_no+1) << '/' << (NUM_BATCHES+1) << std::endl;
//     }
//     GPU_ASSERT(cudaDeviceReset());
//     return 0;
// }
// )";
//     }
}
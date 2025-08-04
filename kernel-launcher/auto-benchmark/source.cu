#include "cuda_runtime.h"
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
namespace kernel0 {
extern "C" {
    __global__ void state_prediction_rolls(
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
        u64 base_state = tid * (5U << 17);

        for (u32 rem = 2U; rem < 5U; rem++) {
            u64 state = base_state + rem<<17;
            u64* rand = &state;

            int counter = 0;
            for (u32 r = 0; r < rem+4; r++) {
                int item = data[nextInt(rand, 28)];
                if (item == 3)
                    counter += nextInt(rand, 2) + 1;
                else if (item < 2) {
                    state = (state * JRAND_MULTIPLIER + 11) & MASK_48;
                }
            }

            if (counter >= 11) {
                u32 ix = atomicAdd(result_count, 1);
                result_array[ix] = tid ^ JRAND_MULTIPLIER;
            }
        }
    }
}
} //namespace
namespace kernel1 {
extern "C" {
    __global__ void naive_bruteforce(
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

        u64 state = tid;
        u64* rand = &state;

        int rolls = 4 + nextInt(rand, 5);
        int counter = 0;
        for (int r = 0; r < rolls; r++) {
            int item = data[nextInt(rand, 28)];
            if (item == 3)
                counter += nextInt(rand, 2) + 1;
            else if (item < 2) {
                state = (state * JRAND_MULTIPLIER + 11) & MASK_48;
            }
        }

        if (counter >= 11) {
            u32 ix = atomicAdd(result_count, 1);
            result_array[ix] = tid ^ JRAND_MULTIPLIER;
        }
    }
}
} //namespace
namespace kernel2 {
extern "C" {
    __global__ void state_prediction_item(
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
        u64 state = tid * (28U << 17) + (25U << 17);
        u64* rand = &state;
        int counter = 1 + nextInt(rand, 2);

        for (int r = 0; r < 7; r++) {
            int item = data[nextInt(rand, 28)];
            if (item == 3)
                counter += nextInt(rand, 2) + 1;
            else if (item < 2) {
                state = (state * JRAND_MULTIPLIER + 11) & MASK_48;
            }
        }
        if (counter < 11)
            return;

        state = tid * (28U << 17) + (25U << 17);
        for (int back = 0; back < 10; back++) {
            state = (state * (-35320271006875LL) - 174426972345687LL) & MASK_48;
            u64 state2 = state;

            int rolls = nextInt(&state2, 5) + 4;
            int counter2 = 0;

            for (int r = 0; r < rolls; r++) {
                int item = data[nextInt(&state2, 28)];
                if (item == 3)
                    counter += nextInt(&state2, 2) + 1;
                else if (item < 2) {
                    state2 = (state2 * JRAND_MULTIPLIER + 11) & MASK_48;
                }
            }
            if (counter2 >= 11) {
                u32 ix = atomicAdd(result_count, 1);
                result_array[ix] = tid ^ JRAND_MULTIPLIER;
            }
        }
    }
}
} //namespace
namespace kernel0 {
void launch(
    const uint32_t num_blocks, const uint32_t threads_per_block, const uint32_t shared_mem_bytes,
    uint64_t* result_array, uint32_t* result_count,
    uint32_t* shared_mem_contents, uint32_t shared_mem_contents_length, 
    uint64_t offset) 
{
    state_prediction_rolls<<< num_blocks, threads_per_block, shared_mem_bytes >>> (
        result_array, result_count, shared_mem_contents, shared_mem_contents_length, offset
    );
}} //namespace
namespace kernel1 {
void launch(
    const uint32_t num_blocks, const uint32_t threads_per_block, const uint32_t shared_mem_bytes,
    uint64_t* result_array, uint32_t* result_count,
    uint32_t* shared_mem_contents, uint32_t shared_mem_contents_length, 
    uint64_t offset) 
{
    naive_bruteforce<<< num_blocks, threads_per_block, shared_mem_bytes >>> (
        result_array, result_count, shared_mem_contents, shared_mem_contents_length, offset
    );
}} //namespace
namespace kernel2 {
void launch(
    const uint32_t num_blocks, const uint32_t threads_per_block, const uint32_t shared_mem_bytes,
    uint64_t* result_array, uint32_t* result_count,
    uint32_t* shared_mem_contents, uint32_t shared_mem_contents_length, 
    uint64_t offset) 
{
    state_prediction_item<<< num_blocks, threads_per_block, shared_mem_bytes >>> (
        result_array, result_count, shared_mem_contents, shared_mem_contents_length, offset
    );
}} //namespace

typedef void (*launch_function)(uint32_t, uint32_t, uint32_t, uint64_t*, uint32_t*, uint32_t*, uint32_t, uint64_t);

int main() {
    std::vector<launch_function> launchers;
    std::vector<std::vector<uint32_t>> shared_mems;
    launchers.push_back(kernel0::launch); shared_mems.push_back(std::vector<uint32_t>());
    launchers.push_back(kernel1::launch); shared_mems.push_back(std::vector<uint32_t>());
    launchers.push_back(kernel2::launch); shared_mems.push_back(std::vector<uint32_t>());
    {uint32_t shmem[] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,4,5}; for (int k = 0; k < 28; k++) shared_mems[0].push_back(shmem[k]);}
    {uint32_t shmem[] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,4,5}; for (int k = 0; k < 28; k++) shared_mems[1].push_back(shmem[k]);}
    {uint32_t shmem[] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,4,5}; for (int k = 0; k < 28; k++) shared_mems[2].push_back(shmem[k]);}
    return 0;
}

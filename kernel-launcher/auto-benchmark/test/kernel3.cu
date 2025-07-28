typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;

constexpr u64 JRAND_MULTIPLIER = 0x5deece66d;
constexpr u64 MASK_48 = ((1ULL << 48) - 1);

__device__ inline void setSeed(u64* rand, u64 value){ *rand = (value ^ JRAND_MULTIPLIER) & MASK_48; }
__device__ inline int next(u64* rand, const int bits){ *rand = (*rand * JRAND_MULTIPLIER + 11) & MASK_48; return (int)((i64)*rand >> (48 - bits)); }
__device__ inline int nextInt(u64* rand, const int n){ if ((n-1 & n) == 0) {u64 x = n * (u64)next(rand, 31); return (int)((i64)x >> 31);} else {return (int)(next(rand, 31) % n);} }
__device__ inline float nextFloat(u64* rand){ return next(rand, 24) / (float)(1 << 24); }

extern "C" {
    __global__ void state_prediction_item(
        u64* result_array, u32* result_count, 
        u32* shared_mem_contents, u32 shared_mem_contents_length, 
        u64 offset)
    {
        extern __shared__ u32 data[];
        if (threadIdx.x < shared_mem_contents_length) {
            for (int i = threadIdx.x; i < 28; i += blockDim.x) {
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
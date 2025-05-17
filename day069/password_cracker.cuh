#ifndef PASSWORD_CRACKER_CUH
#define PASSWORD_CRACKER_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in error macro

#define MAX_PW_LEN 6

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        /* exit(EXIT_FAILURE); */ /* Exiting from a .cuh might be problematic in tests, consider alternatives */ \
    }

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ unsigned int fnv1a_hash_bytes(const unsigned char* data, int length);

// output_password is a device pointer
// The function should find the password and write it to output_password
void solve(unsigned int target_hash, int password_length, int R, char* output_password);

#endif // PASSWORD_CRACKER_CUH

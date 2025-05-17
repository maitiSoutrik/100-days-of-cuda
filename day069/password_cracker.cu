#include "password_cracker.cuh"
#include <cuda_runtime.h>
#include <cmath> // For pow
#include <vector>
#include <iostream> // For debugging, remove later

// FNV-1a hash function (device implementation)
__device__ unsigned int fnv1a_hash_bytes(const unsigned char* data, int length) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    for (int i = 0; i < length; i++) {
        hash = (hash ^ data[i]);
        hash = hash * FNV_PRIME;
    }
    return hash;
}

// Kernel to crack password
__global__ void password_cracker_kernel(
    unsigned int target_hash, 
    int password_length, 
    int R, 
    char* output_password_device, 
    int* d_found_flag,
    unsigned long long total_passwords) 
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_passwords) {
        return;
    }

    // Early exit if password already found by another thread
    if (*d_found_flag == 1) {
        return;
    }

    unsigned char current_candidate_bytes[MAX_PW_LEN + 1]; // +1 for null terminator (though not hashed)
    unsigned long long temp_idx = idx;

    // Generate password candidate from index (base 26 conversion)
    for (int i = password_length - 1; i >= 0; --i) {
        current_candidate_bytes[i] = 'a' + (temp_idx % 26);
        temp_idx /= 26;
    }
    // Null-terminate for safety if we were to print, though not strictly needed for hashing fixed length
    // current_candidate_bytes[password_length] = '\0'; 

    // Perform R rounds of hashing
    unsigned int current_hash = 0;
    unsigned char hash_input_bytes[4]; // To store the bytes of the previous hash output

    // First round: hash the password string
    current_hash = fnv1a_hash_bytes(current_candidate_bytes, password_length);

    // Subsequent R-1 rounds: hash the 4-byte output of the previous hash
    for (int round = 1; round < R; ++round) {
        // Convert previous hash (uint32_t) to 4 bytes (little-endian)
        hash_input_bytes[0] = (current_hash >> 0) & 0xFF;
        hash_input_bytes[1] = (current_hash >> 8) & 0xFF;
        hash_input_bytes[2] = (current_hash >> 16) & 0xFF;
        hash_input_bytes[3] = (current_hash >> 24) & 0xFF;
        current_hash = fnv1a_hash_bytes(hash_input_bytes, 4);
    }

    // Check if the computed hash matches the target hash
    if (current_hash == target_hash) {
        // Attempt to set the found flag. Only one thread should succeed.
        if (atomicCAS(d_found_flag, 0, 1) == 0) { 
            // This thread is the first to find the password
            for (int i = 0; i < password_length; ++i) {
                output_password_device[i] = current_candidate_bytes[i];
            }
            output_password_device[password_length] = '\0'; // Null-terminate the output
        }
    }
}

// Host function to orchestrate password cracking
void solve(unsigned int target_hash, int password_length, int R, char* output_password_device) {
    if (password_length <= 0 || password_length > MAX_PW_LEN) {
        // Handle invalid password length if necessary, or assume valid input per constraints
        // For now, let's proceed, but in a real scenario, add error handling.
        // A printf here could be useful for debugging on host.
        return; 
    }

    int* d_found_flag;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_found_flag, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_found_flag, 0, sizeof(int))); // Initialize found flag to 0 (false)

    unsigned long long total_possible_passwords = 1;
    for(int i = 0; i < password_length; ++i) {
        total_possible_passwords *= 26;
    }
    
    // Kernel launch parameters
    // Jetson Nano has limited SMs, so smaller block sizes might be better.
    // Max threads per block is 1024.
    // Max blocks depends on compute capability (5.3 for Nano is 65535 in x-dim)
    int threads_per_block = 256; // A common choice, can be tuned
    // Ensure num_blocks doesn't overflow standard int if total_possible_passwords is huge,
    // though unsigned long long for total_possible_passwords handles large search spaces.
    // (total_possible_passwords + threads_per_block - 1) / threads_per_block ensures enough blocks.
    unsigned long long num_blocks_ull = (total_possible_passwords + threads_per_block - 1) / threads_per_block;
    
    // Check if num_blocks_ull exceeds maximum grid dimension for Jetson Nano (sm_53)
    // Max grid dim X is 2^31 - 1 for compute capability >= 3.0, but practically limited by available resources.
    // For sm_53, maxGridSize[0] is 2147483647.
    // However, launching too many blocks can be inefficient or lead to errors.
    // Let's cap it for practical purposes or consider iterative launches if needed.
    // For 26^6 (~308M) passwords, num_blocks_ull would be ~1.2M. This should be fine.
    dim3 num_blocks((unsigned int)num_blocks_ull, 1, 1);
    dim3 block_size(threads_per_block, 1, 1);

    // Launch kernel
    password_cracker_kernel<<<num_blocks, block_size>>>(
        target_hash, 
        password_length, 
        R, 
        output_password_device, 
        d_found_flag,
        total_possible_passwords
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    // Wait for kernel to complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // d_output_password (passed as output_password_device) is already on the device and written to by the kernel.
    // The calling code will be responsible for copying it back to host if needed.

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_found_flag));
}
